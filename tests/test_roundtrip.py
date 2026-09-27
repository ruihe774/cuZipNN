import ctypes
import math
import zlib

import pytest
import torch

from cuzipnn import (
    _codec,
    _kernels,
    compress_tensor,
    compress_tensor_with_crc32,
    decompress_tensor,
    hash_tensor_with_crc32,
)

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
FLOAT_DTYPES = [
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
OTHER_DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.complex64,
]
CHUNK = 4096  # small chunks so moderate tensors span many chunks
# Smallest useful threshold: every chunk's budget rounds down to 0 bytes, so all are stored raw.
_ALL_RAW = 1e-9


def _bits(t):
    return t.contiguous().view(_INT_VIEW[t.element_size()])


def _assert_roundtrip(x, **kw):
    # Exercise the compressed path unless a test asks otherwise; many inputs here are tiny.
    kw.setdefault("min_compress_bytes", 0)
    blob = compress_tensor(x, **kw)
    assert not blob.is_cuda and blob.is_pinned() == kw.get("pin_memory", False)
    assert blob.dtype == torch.uint8 and blob.dim() == 1
    y = decompress_tensor(blob)
    assert y.is_cuda and y.dtype == x.dtype and y.shape == x.shape
    assert torch.equal(_bits(x.cuda()), _bits(y))
    return blob


def _weights(dtype, n, device="cuda"):
    return (torch.randn(n, device=device) * 0.02).to(dtype)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n", [1, 7, CHUNK - 1, CHUNK, CHUNK + 1, 10 * CHUNK + 13, 1_000_003])
def test_float_sizes(dtype, n):
    _assert_roundtrip(_weights(dtype, n), chunk_bytes=CHUNK)


@pytest.mark.parametrize("dtype", OTHER_DTYPES)
def test_other_dtypes(dtype):
    if dtype == torch.bool:
        x = torch.rand(100_003, device="cuda") < 0.3
    elif dtype == torch.complex64:
        x = torch.randn(100_003, dtype=dtype, device="cuda")
    else:
        x = torch.randint(-100, 100, (100_003,), device="cuda").to(dtype)
    _assert_roundtrip(x, chunk_bytes=CHUNK)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pinned(dtype):
    _assert_roundtrip(_weights(dtype, 10 * CHUNK + 13), chunk_bytes=CHUNK, pin_memory=True)
    _assert_roundtrip(torch.empty(0, dtype=dtype), pin_memory=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_default_chunk_large(dtype):
    blob = _assert_roundtrip(_weights(dtype, 5_000_000))
    assert blob.numel() < 0.95 * 5_000_000 * torch.finfo(dtype).bits // 8


@pytest.mark.parametrize("shape", [(), (0,), (3, 0, 5), (1,), (17, 33, 5), (2, 3, 4, 5, 6)])
def test_shapes(shape):
    _assert_roundtrip(
        _weights(torch.bfloat16, math.prod(shape)).reshape(shape), chunk_bytes=CHUNK
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_special_values(dtype):
    info = torch.finfo(dtype)
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            info.max,
            -info.max,
            info.tiny,
            info.tiny / 4,
            -info.tiny / 8,
        ],
        dtype=dtype,
    )
    _assert_roundtrip(specials.repeat(1000).cuda(), chunk_bytes=CHUNK)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_all_bit_patterns(dtype):
    # Every bit pattern (incl. NaN payloads) survives, via random bits reinterpreted as floats too.
    bits = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, (300_001,), dtype=dtype, device="cuda"
    )
    float_view = {
        torch.int8: torch.float8_e5m2,
        torch.int16: torch.bfloat16,
        torch.int32: torch.float32,
    }[dtype]
    blob = _assert_roundtrip(bits.view(float_view), chunk_bytes=CHUNK)
    # Random bytes are incompressible: every chunk must fall back to raw, so the overhead stays tiny.
    assert blob.numel() < bits.numel() * bits.element_size() * 1.01


def test_all_zeros_compresses_well():
    blob = _assert_roundtrip(torch.zeros(1_000_000, dtype=torch.bfloat16, device="cuda"))
    # nvCOMP's rANS still spends ~0.5 bit/symbol on a constant chunk (it has no RLE special case).
    assert blob.numel() < 0.1 * 2_000_000


def test_non_contiguous_input():
    x = _weights(torch.bfloat16, 300 * 400).reshape(300, 400)
    _assert_roundtrip(x.t(), chunk_bytes=CHUNK)
    _assert_roundtrip(x[::3, 1::2], chunk_bytes=CHUNK)


def test_unaligned_fp8_view():
    x = _weights(torch.float8_e4m3fn, 100_001)
    _assert_roundtrip(x[3:], chunk_bytes=CHUNK)


def test_cpu_input_and_cpu_blob():
    x = _weights(torch.bfloat16, 100_000, device="cpu")
    blob = compress_tensor(x, chunk_bytes=CHUNK)
    y = decompress_tensor(blob.cpu())
    assert y.is_cuda and torch.equal(_bits(x), _bits(y).cpu())


@pytest.mark.parametrize("n", [0, 100_000])
def test_decompress_to_cpu(n):
    x = _weights(torch.bfloat16, n)
    blob = compress_tensor(x, chunk_bytes=CHUNK)
    for b in (blob, blob.cuda()):
        y = decompress_tensor(b, device="cpu")
        assert y.device.type == "cpu" and y.shape == x.shape


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int8, torch.complex64])
@pytest.mark.parametrize("min_compress_bytes", [0, _codec.DEFAULT_MIN_COMPRESS_BYTES])
def test_decompress_0dim_to_cpu(dtype, min_compress_bytes):
    # A 0-dim tensor cannot be viewed as a dtype of another element size.
    x = torch.tensor(1.5, device="cuda").to(dtype)
    blob = compress_tensor(x, min_compress_bytes=min_compress_bytes)
    assert _is_stored(blob) == (min_compress_bytes > 0)
    for b in (blob, blob.cuda()):
        y = decompress_tensor(b, device="cpu")
        assert y.device.type == "cpu" and y.dtype == dtype and y.shape == ()
        assert torch.equal(_bits(y), _bits(x).cpu())
        assert torch.equal(_bits(x.cpu()), _bits(y))


def test_unaligned_blob_view():
    x = _weights(torch.float32, 100_000)
    blob = compress_tensor(x, chunk_bytes=CHUNK)
    padded = torch.empty(blob.numel() + 3, dtype=torch.uint8, device="cuda")
    padded[3:] = blob
    assert torch.equal(_bits(decompress_tensor(padded[3:])), _bits(x))


def test_deterministic():
    x = _weights(torch.bfloat16, 1_000_003)
    assert torch.equal(
        compress_tensor(x, chunk_bytes=CHUNK), compress_tensor(x, chunk_bytes=CHUNK)
    )


def test_deterministic_despite_stale_device_memory():
    # nvCOMP's rANS leaves some bytes within comp_bytes unwritten, and each stream's last chunk
    # (2307 bytes here, stored raw for the mantissa stream) is copied through to its 8-byte
    # boundary; neither may pick up stale memory.
    x = _weights(torch.bfloat16, 4_000_003)
    blobs = []
    for fill in (0x00, 0xA5):
        torch.cuda.empty_cache()
        # Freed straight back to the cache, so the next call's buffers start out dirty.
        torch.full((256 << 20,), fill, dtype=torch.uint8, device="cuda")
        blobs.append(compress_tensor(x, chunk_bytes=CHUNK))
    assert torch.equal(*blobs)


def test_deterministic_despite_bytes_past_view():
    # A single-byte tensor is copied straight from its own memory, through to the 8-byte boundary.
    base = torch.randint(0, 256, (CHUNK + 16,), dtype=torch.uint8, device="cuda")
    x = base[: CHUNK + 3]
    blobs = []
    for fill in (0x00, 0xA5):
        base[CHUNK + 3 :] = fill
        blobs.append(_assert_roundtrip(x, chunk_bytes=CHUNK, passthrough_threshold=_ALL_RAW))
    assert torch.equal(*blobs)


def test_non_default_stream():
    x = _weights(torch.bfloat16, 1_000_003)
    ref = compress_tensor(x, chunk_bytes=CHUNK)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        blob = compress_tensor(x, chunk_bytes=CHUNK)
    assert torch.equal(blob, ref)


def test_tiny_threshold_stores_everything_raw():
    x = _weights(torch.bfloat16, 100_000)
    blob = _assert_roundtrip(x, chunk_bytes=CHUNK, passthrough_threshold=_ALL_RAW)
    assert blob.numel() >= 200_000


@pytest.mark.parametrize("passthrough_threshold", [0.0, -0.5, 1.01])
def test_bad_threshold(passthrough_threshold):
    with pytest.raises(ValueError, match="passthrough_threshold"):
        compress_tensor(
            _weights(torch.bfloat16, 100), passthrough_threshold=passthrough_threshold
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("sub_chunk_bytes", [1, 1000, 8 * 1024, 32 * 1024, 1 << 30])
def test_sub_chunk_bytes(dtype, sub_chunk_bytes):
    # decompress_tensor needs no matching argument: nvCOMP reads the sub-chunk count from the bitstream.
    _assert_roundtrip(_weights(dtype, 1_000_003), sub_chunk_bytes=sub_chunk_bytes)


@pytest.mark.parametrize(
    "chunk_bytes,sub_chunk_bytes,count",
    [
        (128 << 10, 8 << 10, 16),
        (128 << 10, 7 << 10, 32),
        (16 << 10, 8 << 10, 4),
        (1 << 24, 1, 64),
    ],
)
def test_sub_chunk_count(chunk_bytes, sub_chunk_bytes, count):
    assert _codec._sub_chunks(chunk_bytes, sub_chunk_bytes) == count


def test_larger_sub_chunks_compress_better():
    x = _weights(torch.bfloat16, 2_000_000)
    small, large = (compress_tensor(x, sub_chunk_bytes=s).numel() for s in (2 << 10, 32 << 10))
    assert large < small


def test_bad_sub_chunk_bytes():
    with pytest.raises(ValueError):
        compress_tensor(_weights(torch.bfloat16, 100), sub_chunk_bytes=0)


def test_bad_blob():
    with pytest.raises(ValueError):
        decompress_tensor(torch.zeros(64, dtype=torch.uint8))


def _mixed_int8(n_chunks):
    # Alternating incompressible and compressible chunks, plus a short raw tail.
    parts = [
        torch.randint(0, 256, (CHUNK,), dtype=torch.uint8)
        if i % 2 == 0
        else torch.zeros(CHUNK, dtype=torch.uint8)
        for i in range(n_chunks)
    ]
    return torch.cat(parts + [torch.randint(0, 256, (13,), dtype=torch.uint8)]).cuda()


@pytest.mark.parametrize("layout", ["offset", "strided", "offset_in_misaligned_storage"])
def test_unaligned_cpu_blob_view(layout):
    x = _mixed_int8(7)
    blob = compress_tensor(x, chunk_bytes=CHUNK, min_compress_bytes=0)
    if layout == "offset":
        view = torch.cat([torch.zeros(3, dtype=torch.uint8), blob])[3:]
    elif layout == "strided":
        view = torch.stack([blob, blob], 1)[:, 0]
    else:
        # The storage starts 3 bytes into the buffer, so the view's address is 4-byte aligned
        # but its storage offset (1) is not.
        buf = bytearray(blob.numel() + 4)
        torch.frombuffer(buf, dtype=torch.uint8)[4:].copy_(blob)
        view = torch.frombuffer(memoryview(buf)[3:], dtype=torch.uint8)[1:]
    assert torch.equal(decompress_tensor(view), x)


def test_truncated_data_section():
    blob = compress_tensor(_mixed_int8(7), chunk_bytes=CHUNK, min_compress_bytes=0)
    with pytest.raises(ValueError, match="truncated"):
        decompress_tensor(blob[:-8])


def _zipnn_reference_streams(x: torch.Tensor, reorder: bool) -> torch.Tensor:
    """Reorder + byte grouping as in ZipNN's csrc/data_manipulation_dtype{16,32}.c."""
    k = x.element_size()
    w = 8 * k
    u = x.to(torch.int64) & ((1 << w) - 1)
    if reorder:
        sign = (u >> (w - 1)) & 1
        exp = (u >> (w - 9)) & 0xFF
        mant = u & ((1 << (w - 9)) - 1)
        u = (exp << (w - 8)) | (sign << (w - 9)) | mant
    return torch.stack([((u >> (8 * b)) & 0xFF).to(torch.uint8) for b in range(k)])


@pytest.mark.parametrize(
    "dtype,reorder", [(torch.bfloat16, True), (torch.float32, True), (torch.float16, False)]
)
def test_split_matches_zipnn(dtype, reorder):
    x = torch.randn(10_001, device="cuda").to(dtype)
    n, k = x.numel(), x.element_size()
    planar = torch.empty(k * n, dtype=torch.uint8, device="cuda")
    _kernels.split(x, planar, n, reorder)
    ref = _zipnn_reference_streams(x.cpu().view(_INT_VIEW[k]), reorder)
    assert torch.equal(planar.view(k, n).cpu(), ref)
    # And against ZipNN's exact C bit formulas for the 16-bit case.
    if dtype == torch.bfloat16:
        v = x.cpu().view(torch.int16).to(torch.int64) & 0xFFFF
        c = ((v << 1) & 0xFF00) | ((v >> 8) & 0x80) | (v & 0x7F)
        assert torch.equal(ref[1], (c >> 8).to(torch.uint8))
        assert torch.equal(ref[0], (c & 0xFF).to(torch.uint8))


@pytest.mark.parametrize(
    "dtype,stride",
    [
        (torch.float32, 800_000_000),
        (torch.float32, 800_000_008),
        (torch.float64, 320_000_000),
        (torch.float64, 320_000_008),
    ],
)
def test_split_large_stride(dtype, stride):
    # (K - 1) * stride >= 2**31 while stride itself fits in int32: stream offsets must not wrap.
    # Strides that are not multiples of 16 skip Triton's divisibility specialization.
    x = torch.randn(4099, device="cuda").to(dtype)
    n, k = x.numel(), x.element_size()
    planar = torch.zeros(k * stride, dtype=torch.uint8, device="cuda")
    _kernels.split(x, planar, stride, False)
    ref = _zipnn_reference_streams(x.cpu().view(_INT_VIEW[k]), False)
    for b in range(k):
        assert torch.equal(planar[b * stride : b * stride + n].cpu(), ref[b])


def test_pinned_blob_reusable_after_return():
    # The caller may overwrite a pinned blob as soon as decompress_tensor returns.
    x = _weights(torch.bfloat16, 1_000_003)
    blob = compress_tensor(x, chunk_bytes=CHUNK, pin_memory=True)
    # Warm up first: loading the Triton module on the first launch synchronizes the device.
    decompress_tensor(blob)
    torch.cuda._sleep(1 << 30)  # keep the GPU busy so the upload is still queued on return
    y = decompress_tensor(blob)
    blob.zero_()
    assert torch.equal(_bits(y), _bits(x))


def _is_stored(blob):
    return bool(blob[7].item() & _codec._FLAG_STORED)


@pytest.mark.parametrize("dtype", FLOAT_DTYPES + OTHER_DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (3, 0, 5), (1,), (7,), (17, 33, 5)])
def test_stored_roundtrip(dtype, shape):
    x = torch.randn(shape, device="cuda").to(dtype)
    nbytes = x.numel() * x.element_size()
    blob = _assert_roundtrip(x, min_compress_bytes=nbytes + 1)
    assert _is_stored(blob)
    assert blob.numel() == _codec._FIXED.size + 8 * len(shape) + nbytes
    y = decompress_tensor(blob, device="cpu")
    assert y.device.type == "cpu" and torch.equal(_bits(y), _bits(x.cpu()))


def test_stored_boundary():
    x = _weights(torch.bfloat16, 1000)
    assert _is_stored(compress_tensor(x, min_compress_bytes=2001))
    assert not _is_stored(_assert_roundtrip(x, min_compress_bytes=2000))


def test_default_min_compress_bytes():
    small = _weights(torch.bfloat16, _codec.DEFAULT_MIN_COMPRESS_BYTES // 2 - 1)
    large = _weights(torch.bfloat16, _codec.DEFAULT_MIN_COMPRESS_BYTES // 2)
    assert _is_stored(compress_tensor(small))
    assert not _is_stored(compress_tensor(large))
    for x in (small, large):
        assert torch.equal(_bits(decompress_tensor(compress_tensor(x))), _bits(x))


def test_stored_non_contiguous_and_cpu_input():
    x = _weights(torch.float32, 30 * 40).reshape(30, 40)
    for v in (x.t(), x[::3, 1::2], x.cpu().t()):
        blob = compress_tensor(v, pin_memory=True)
        assert _is_stored(blob) and blob.is_pinned()
        assert torch.equal(_bits(decompress_tensor(blob)), _bits(v.cuda()))
        assert torch.equal(_bits(decompress_tensor(blob.cuda())), _bits(v.cuda()))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stored_unaligned_blob_view(device):
    x = _weights(torch.float64, 101)
    blob = compress_tensor(x)
    padded = torch.empty(blob.numel() + 3, dtype=torch.uint8, device=device)
    padded[3:] = blob
    assert torch.equal(_bits(decompress_tensor(padded[3:])), _bits(x))


def test_stored_truncated():
    blob = compress_tensor(_weights(torch.bfloat16, 100))
    assert _is_stored(blob)
    with pytest.raises(ValueError, match="truncated"):
        decompress_tensor(blob[:-1])


def test_bad_min_compress_bytes():
    with pytest.raises(ValueError, match="min_compress_bytes"):
        compress_tensor(_weights(torch.bfloat16, 100), min_compress_bytes=-1)


def _crc32_ref(t, chunk=_codec.DEFAULT_CHECKSUM_CHUNK_BYTES):
    t = t.detach().cpu().contiguous()
    b = ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())
    return [zlib.crc32(b[i : i + chunk]) for i in range(0, len(b), chunk)]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _weights(torch.bfloat16, 1_000_003), id="bf16"),
        pytest.param(lambda: _weights(torch.float32, 300_001).cpu(), id="fp32-cpu"),
        pytest.param(lambda: _mixed_int8(7), id="int8-mixed"),
        pytest.param(lambda: _weights(torch.float16, 1000).reshape(10, 100).t(), id="fp16-t"),
        pytest.param(lambda: torch.empty(3, 0, 5, device="cuda"), id="empty"),
    ],
)
@pytest.mark.parametrize("min_compress_bytes", [0, 1 << 30])
@pytest.mark.parametrize("checksum_chunk_bytes", [8, 1000, 16 * 1024, 1 << 30])
def test_crc32_matches_zlib(make, min_compress_bytes, checksum_chunk_bytes):
    # 8 divides every blob size (no short last chunk); 1 << 30 exceeds it (one short chunk).
    x = make()
    kw = {"chunk_bytes": CHUNK, "min_compress_bytes": min_compress_bytes}
    blob, crc = compress_tensor_with_crc32(x, checksum_chunk_bytes=checksum_chunk_bytes, **kw)
    assert torch.equal(blob, compress_tensor(x, **kw))
    assert _is_stored(blob) == (min_compress_bytes > 0)
    assert crc.dtype == torch.uint32 and crc.device.type == "cpu" and crc.dim() == 1
    assert crc.tolist() == _crc32_ref(blob, checksum_chunk_bytes)


def test_crc32_default_chunk():
    assert _codec.DEFAULT_CHECKSUM_CHUNK_BYTES == 16 * 1024
    blob, crc = compress_tensor_with_crc32(_weights(torch.bfloat16, 1_000_003))
    assert crc.numel() == math.ceil(blob.numel() / (16 * 1024)) > 1
    assert crc.tolist() == _crc32_ref(blob)


def test_hash_blob_matches_checksums():
    x = _weights(torch.bfloat16, 1_000_003)
    blob, crc = compress_tensor_with_crc32(x, chunk_bytes=CHUNK, pin_memory=True)
    assert blob.is_pinned()
    padded = torch.empty(blob.numel() + 3, dtype=torch.uint8, device="cuda")
    padded[3:] = blob
    for view in (blob, blob.cuda(), padded[3:], torch.stack([blob, blob], 1)[:, 0]):
        assert torch.equal(hash_tensor_with_crc32(view), crc)
    assert torch.equal(_bits(decompress_tensor(blob)), _bits(x))


def test_hash_tensor():
    x = _weights(torch.float32, 30 * 40).reshape(30, 40)
    for t in (x, x.t(), x.cpu().t(), x[::3, 1::2], torch.tensor(1.5), torch.empty(0)):
        crc = hash_tensor_with_crc32(t, checksum_chunk_bytes=1000)
        assert crc.dtype == torch.uint32 and crc.device.type == "cpu"
        assert crc.tolist() == _crc32_ref(t, 1000)


def test_crc32_detects_corruption():
    blob, crc = compress_tensor_with_crc32(
        _weights(torch.bfloat16, 1_000_003), chunk_bytes=CHUNK
    )
    for pos in (0, 16 * 1024 - 1, 16 * 1024, blob.numel() // 2, blob.numel() - 1):
        bad = blob.clone()
        bad[pos] ^= 1
        diff = (hash_tensor_with_crc32(bad).to(torch.int64) != crc.to(torch.int64)).nonzero()
        assert diff.view(-1).tolist() == [pos // (16 * 1024)]


def test_crc32_non_default_stream():
    x = _weights(torch.bfloat16, 1_000_003)
    ref = compress_tensor_with_crc32(x, chunk_bytes=CHUNK)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        blob, crc = compress_tensor_with_crc32(x, chunk_bytes=CHUNK)
        assert torch.equal(hash_tensor_with_crc32(blob), crc)
    assert torch.equal(blob, ref[0]) and torch.equal(crc, ref[1])


@pytest.mark.parametrize("checksum_chunk_bytes", [0, -1])
def test_bad_checksum_chunk_bytes(checksum_chunk_bytes):
    x = _weights(torch.bfloat16, 100)
    with pytest.raises(ValueError, match="checksum_chunk_bytes"):
        compress_tensor_with_crc32(x, checksum_chunk_bytes=checksum_chunk_bytes)
    with pytest.raises(ValueError, match="checksum_chunk_bytes"):
        hash_tensor_with_crc32(x, checksum_chunk_bytes=checksum_chunk_bytes)
