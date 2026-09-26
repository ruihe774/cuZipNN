import math

import pytest
import torch

from cuzipnn import _codec, _kernels, compress_tensor, decompress_tensor

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


def _bits(t):
    return t.contiguous().view(_INT_VIEW[t.element_size()])


def _assert_roundtrip(x, **kw):
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
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    blobs = []
    for fill in (0x00, 0xA5):
        torch.cuda.empty_cache()
        # Freed straight back to the cache, so the next call's buffers start out dirty. The cache
        # is per stream, and compress_tensor allocates on the caller's stream unless that is the
        # legacy default one, so dirty and compress on the same side stream.
        with torch.cuda.stream(s):
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
        blobs.append(_assert_roundtrip(x, chunk_bytes=CHUNK, threshold=0.0))
    assert torch.equal(*blobs)


def test_non_default_stream():
    x = _weights(torch.bfloat16, 1_000_003)
    ref = compress_tensor(x, chunk_bytes=CHUNK)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        blob = compress_tensor(x, chunk_bytes=CHUNK)
    assert torch.equal(blob, ref)


def test_threshold_zero_stores_everything_raw():
    x = _weights(torch.bfloat16, 100_000)
    blob = _assert_roundtrip(x, chunk_bytes=CHUNK, threshold=0.0)
    assert blob.numel() >= 200_000


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
    blob = compress_tensor(x, chunk_bytes=CHUNK)
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
    blob = compress_tensor(_mixed_int8(7), chunk_bytes=CHUNK)
    with pytest.raises(ValueError, match="truncated"):
        decompress_tensor(blob[:-8])


def test_raw_chunk_with_wrong_size():
    blob = compress_tensor(_mixed_int8(7), chunk_bytes=CHUNK)
    sizes_off, _, _ = _codec._layout(1, 0, 0)
    # Shrink the raw 13-byte tail (chunk 7) within its 8-byte padding, so no chunk moves; moving
    # the ANS chunks would hand nvCOMP garbage, which it does not survive.
    stored = blob[sizes_off + 4 * 7 : sizes_off + 4 * 8].view(torch.int32)
    assert stored.item() == 13
    stored -= 1
    with pytest.raises(ValueError, match="raw chunk"):
        decompress_tensor(blob)


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
