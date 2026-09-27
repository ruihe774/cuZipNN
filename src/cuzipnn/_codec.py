"""ZipNN on the GPU: bit reordering and byte grouping in Triton, entropy coding with nvCOMP ANS.

Blob layout (little-endian, every section 8-byte aligned):

    0   magic "ZNNG" | u8 version | u8 dtype code | u8 ndim | u8 flags
    8   u32 chunk_bytes | u32 n_comp
    16  u64 numel
    24  i64 shape[ndim]

flags: bit 0 = exponent bit reordering, bit 1 = stored uncompressed.

A stored blob (tensors smaller than min_compress_bytes) has chunk_bytes = n_comp = 0, and the
header is followed directly by the tensor's bytes in row-major order. Otherwise it continues:

    ..  i32 stored_bytes[K * nch]  bytes stored for each (stream, chunk), ANS-compressed or raw
    ..  i32 comp_idx[n_comp]       indices of the ANS-compressed chunks
    ..  data                       chunks in (stream, chunk) order, each padded to 8 bytes

K is the element size in bytes (one stream per byte), and nch = ceil(numel / chunk_bytes).
Chunk j of every stream covers elements [j * chunk_bytes, (j + 1) * chunk_bytes).
"""

import ctypes
import struct
from collections.abc import Buffer

import torch
import triton

from . import _kernels, _nvcomp

_MAGIC = b"ZNNG"
_VERSION = 2
_FIXED = struct.Struct("<4sBBBBIIQ")
_FLAG_REORDER = 1
_FLAG_STORED = 2
_MAX_NDIM = 64

DEFAULT_CHUNK_BYTES = 128 * 1024
DEFAULT_SUB_CHUNK_BYTES = 8 * 1024
DEFAULT_PASSTHROUGH_THRESHOLD = 0.95
DEFAULT_MIN_COMPRESS_BYTES = 64 * 1024
DEFAULT_CHECKSUM_CHUNK_BYTES = 16 * 1024

_DTYPES = {
    1: torch.bool,
    2: torch.uint8,
    3: torch.int8,
    4: torch.int16,
    5: torch.int32,
    6: torch.int64,
    7: torch.float16,
    8: torch.bfloat16,
    9: torch.float32,
    10: torch.float64,
    11: torch.float8_e4m3fn,
    12: torch.float8_e5m2,
    13: torch.uint16,
    14: torch.uint32,
    15: torch.uint64,
    16: torch.complex64,
}
_CODES = {dtype: code for code, dtype in _DTYPES.items()}
# As in ZipNN, only bfloat16 and float32 get the exponent bit reordering.
_REORDER = {torch.bfloat16, torch.float32}


def _align8[T: (int, torch.Tensor)](x: T) -> T:
    return (x + 7) & ~7


def _raw_bytes(
    k: int, nch: int, n: int, chunk_bytes: int, threshold: float, device: torch.device | str
) -> torch.Tensor:
    """Uncompressed size of each (stream, chunk): chunk_bytes, except for each stream's last chunk."""
    raw = torch.full((k, nch), chunk_bytes * threshold, dtype=torch.int64, device=device)
    raw[:, -1] = (n - (nch - 1) * chunk_bytes) * threshold
    return raw.view(-1)


def _sub_chunks(chunk_bytes: int, sub_chunk_bytes: int) -> int:
    """nvCOMP's sub-chunk count: chunk_bytes / sub_chunk_bytes, rounded up to a power of 2 in [4, 64]."""
    count = 1 << (triton.cdiv(chunk_bytes, sub_chunk_bytes) - 1).bit_length()
    return min(max(count, _nvcomp.MIN_SUB_CHUNKS), _nvcomp.MAX_SUB_CHUNKS)


def _layout(ndim: int, total: int, n_comp: int) -> tuple[int, int, int]:
    sizes_off = 24 + 8 * ndim
    idx_off = sizes_off + 4 * total
    data_off = _align8(idx_off + 4 * n_comp)
    return sizes_off, idx_off, data_off


def _gpu(x: torch.Tensor) -> torch.device:
    return x.device if x.is_cuda else torch.device("cuda", torch.cuda.current_device())


def _crc32(data: torch.Tensor, checksum_chunk_bytes: int) -> torch.Tensor:
    """Enqueue the CRC-32 of each checksum_chunk_bytes-byte chunk of `data`, a contiguous uint8
    CUDA tensor. The last chunk may be shorter. Returns uint32 on the GPU."""
    nbytes = data.numel()
    m = triton.cdiv(nbytes, checksum_chunk_bytes)
    crc = torch.empty(m, dtype=torch.uint32, device=data.device)
    if m:
        # A short last chunk just gets its own size in the same batch: one launch, no padding.
        sizes = torch.full((m,), checksum_chunk_bytes, dtype=torch.int64)
        sizes[-1] = nbytes - (m - 1) * checksum_chunk_bytes
        base = data.data_ptr()
        ptrs = torch.arange(base, base + nbytes, checksum_chunk_bytes, dtype=torch.int64)
        tbl = torch.stack([ptrs, sizes]).to(data.device, non_blocking=True)
        _nvcomp.crc32(tbl[0], tbl[1], min(checksum_chunk_bytes, nbytes), crc)
    return crc


def _fill_stored(
    blob: torch.Tensor, offset: int, x: torch.Tensor, checksum_chunk_bytes: int | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy x's bytes into blob[offset:], after blob's header. With checksum_chunk_bytes, the blob
    is assembled and hashed on the GPU before it is copied back."""
    if checksum_chunk_bytes is None:
        blob[offset:].view(x.dtype).view(x.shape).copy_(x)
        return blob, None
    device = _gpu(x)
    with torch.cuda.device(device):
        buf = torch.empty(blob.numel(), dtype=torch.uint8, device=device)
        buf[:offset].copy_(blob[:offset], non_blocking=True)
        buf[offset:].view(x.dtype).view(x.shape).copy_(x, non_blocking=True)
        crc = _crc32(buf, checksum_chunk_bytes)
        blob.copy_(buf)
        return blob, crc.cpu()


def compress_tensor(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    sub_chunk_bytes: int = DEFAULT_SUB_CHUNK_BYTES,
    passthrough_threshold: float = DEFAULT_PASSTHROUGH_THRESHOLD,
    min_compress_bytes: int = DEFAULT_MIN_COMPRESS_BYTES,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Compress a tensor losslessly on the GPU; returns a 1-D uint8 tensor in CPU memory."""
    blob, _ = _compress(
        tensor,
        chunk_bytes=chunk_bytes,
        sub_chunk_bytes=sub_chunk_bytes,
        passthrough_threshold=passthrough_threshold,
        min_compress_bytes=min_compress_bytes,
        checksum_chunk_bytes=None,
        pin_memory=pin_memory,
    )
    return blob


def compress_tensor_with_crc32(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    sub_chunk_bytes: int = DEFAULT_SUB_CHUNK_BYTES,
    passthrough_threshold: float = DEFAULT_PASSTHROUGH_THRESHOLD,
    min_compress_bytes: int = DEFAULT_MIN_COMPRESS_BYTES,
    checksum_chunk_bytes: int = DEFAULT_CHECKSUM_CHUNK_BYTES,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """compress_tensor, plus the CRC-32 (as zlib.crc32) of each checksum_chunk_bytes-byte chunk
    of the blob, the last one possibly shorter, as a 1-D uint32 tensor in CPU memory.

    The blob is the same as compress_tensor's. The checksums are computed on the GPU before the
    blob is copied back; hash_tensor_with_crc32(blob) reproduces them, so a blob can be verified
    before it is passed to decompress_tensor.
    """
    blob, checksums = _compress(
        tensor,
        chunk_bytes=chunk_bytes,
        sub_chunk_bytes=sub_chunk_bytes,
        passthrough_threshold=passthrough_threshold,
        min_compress_bytes=min_compress_bytes,
        checksum_chunk_bytes=checksum_chunk_bytes,
        pin_memory=pin_memory,
    )
    assert checksums is not None
    return blob, checksums


def hash_tensor_with_crc32(
    tensor: torch.Tensor,
    *,
    checksum_chunk_bytes: int = DEFAULT_CHECKSUM_CHUNK_BYTES,
) -> torch.Tensor:
    """CRC-32 (as zlib.crc32) of each checksum_chunk_bytes-byte chunk of the tensor's bytes in
    row-major order, the last one possibly shorter, computed on the GPU. Returns a 1-D uint32
    tensor in CPU memory."""
    if checksum_chunk_bytes <= 0:
        raise ValueError("checksum_chunk_bytes must be positive")
    x = tensor.detach()
    device = _gpu(x)
    with torch.cuda.device(device):
        x = x.to(device, non_blocking=True, memory_format=torch.contiguous_format).contiguous()
        return _crc32(x.view(-1).view(torch.uint8), checksum_chunk_bytes).cpu()


def _compress(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int,
    sub_chunk_bytes: int,
    passthrough_threshold: float,
    min_compress_bytes: int,
    checksum_chunk_bytes: int | None,
    pin_memory: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if tensor.dtype not in _CODES:
        raise TypeError(f"unsupported dtype {tensor.dtype}")
    if chunk_bytes % 8 or not 0 < chunk_bytes <= _nvcomp.MAX_CHUNK_BYTES:
        raise ValueError(
            f"chunk_bytes must be a multiple of 8 in (0, {_nvcomp.MAX_CHUNK_BYTES}]"
        )
    if passthrough_threshold <= 0 or passthrough_threshold > 1:
        raise ValueError("passthrough_threshold must be in (0, 1]")
    if sub_chunk_bytes <= 0:
        raise ValueError("sub_chunk_bytes must be positive")
    if min_compress_bytes < 0:
        raise ValueError("min_compress_bytes must be non-negative")
    if checksum_chunk_bytes is not None and checksum_chunk_bytes <= 0:
        raise ValueError("checksum_chunk_bytes must be positive")
    sub_chunks = _sub_chunks(chunk_bytes, sub_chunk_bytes)
    if tensor.dim() > _MAX_NDIM:
        raise ValueError(f"at most {_MAX_NDIM} dimensions are supported")

    x = tensor.detach()

    k, n, reorder = x.element_size(), x.numel(), x.dtype in _REORDER
    nch = triton.cdiv(n, chunk_bytes)
    total = k * nch

    def header(n_comp: int, size: int | None = None, stored: bool = False) -> torch.Tensor:
        shape_spec = struct.Struct(f"<{tensor.dim()}q")
        header_size = _FIXED.size + shape_spec.size
        assert header_size == 24 + 8 * tensor.dim()  # sanity check
        r = torch.empty(
            header_size if size is None else size,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        b = (ctypes.c_byte * r.numel()).from_address(r.data_ptr())
        _FIXED.pack_into(
            b,
            0,
            _MAGIC,
            _VERSION,
            _CODES[x.dtype],
            tensor.dim(),
            _FLAG_STORED if stored else _FLAG_REORDER if reorder else 0,
            0 if stored else chunk_bytes,
            n_comp,
            n,
        )
        shape_spec.pack_into(b, _FIXED.size, *tensor.shape)
        return r

    if n * k < min_compress_bytes:
        header_size = _FIXED.size + 8 * tensor.dim()
        blob = header(0, size=header_size + n * k, stored=True)
        return _fill_stored(blob, header_size, x, checksum_chunk_bytes)

    if n == 0:
        blob = header(0)
        return _fill_stored(blob, blob.numel(), x, checksum_chunk_bytes)

    device = _gpu(x)
    with torch.cuda.device(device):
        x = (
            x.to(device, non_blocking=True, memory_format=torch.contiguous_format)
            .contiguous()
            .view(-1)
        )

        # Bit reordering + byte grouping into K contiguous streams of nch chunks each.
        if k == 1:
            planar = x.view(torch.uint8)
            if planar.data_ptr() % 8:
                planar = planar.clone()
        else:
            planar = torch.empty(total * chunk_bytes, dtype=torch.uint8, device=device)
            _kernels.split(x, planar, nch * chunk_bytes, reorder)

        # Entropy-code every (stream, chunk) in one nvCOMP batch.
        in_start = planar.data_ptr()
        in_ptrs = torch.arange(
            in_start,
            in_start + total * chunk_bytes,
            chunk_bytes,
            dtype=torch.int64,
            device=device,
        )
        slot = _align8(_nvcomp.max_compressed_chunk_bytes(chunk_bytes, sub_chunks))
        # The blob is packed into this buffer at the end. Stored chunks are never larger than
        # raw ones, so it also fits the header and tables.
        head = _align8(24 + 8 * tensor.dim() + 8 * total)
        # Zeroed: nvCOMP's rANS leaves some bytes within comp_bytes unwritten, which would
        # otherwise leak stale device memory into the blob and make it non-deterministic.
        comp = torch.zeros(
            max(total * slot, head + total * chunk_bytes), dtype=torch.uint8, device=device
        )
        out_start = comp.data_ptr()
        out_ptrs = torch.arange(
            out_start, out_start + total * slot, slot, dtype=torch.int64, device=device
        )
        raw_bytes = _raw_bytes(k, nch, n, chunk_bytes, 1, device)
        comp_bytes = torch.empty(total, dtype=torch.int64, device=device)
        statuses = torch.empty(total, dtype=torch.int32, device=device)
        _nvcomp.compress(
            in_ptrs,
            raw_bytes,
            chunk_bytes,
            k * n,
            out_ptrs,
            comp_bytes,
            statuses,
            sub_chunks,
        )

        raw_bytes = _raw_bytes(k, nch, n, chunk_bytes, 1, "cpu")
        threshold_bytes = _raw_bytes(k, nch, n, chunk_bytes, passthrough_threshold, "cpu")
        # Keep a chunk compressed only if it saves enough; otherwise store it raw (ZipNN's threshold rule).
        comp_bytes = torch.where(
            statuses == _nvcomp.NVCOMP_SUCCESS, comp_bytes, _nvcomp.MAX_CHUNK_BYTES
        ).to("cpu")
        use = comp_bytes < threshold_bytes
        stored = torch.where(use, comp_bytes, raw_bytes)
        padded = _align8(stored)
        ends = torch.cumsum(padded, 0)
        comp_idx = use.nonzero().view(-1)
        n_comp = comp_idx.numel()

        sizes_off, idx_off, data_off = _layout(tensor.dim(), total, n_comp)
        size = data_off + int(ends[-1])
        prefix = header(n_comp, size=data_off)
        prefix[sizes_off:idx_off].view(torch.int32).copy_(stored)
        prefix[idx_off : idx_off + 4 * n_comp].view(torch.int32).copy_(comp_idx)
        prefix[idx_off + 4 * n_comp :].zero_()

        # Pack in two passes, so that no pass reads from the buffer it writes to. First move the
        # ANS chunks out of comp: into their own planar region, which is dead once the chunk is
        # compressed, or for K = 1 (planar may be the caller's tensor) into a temporary buffer.
        in_ptrs = torch.arange(
            in_start, in_start + total * chunk_bytes, chunk_bytes, dtype=torch.int64
        )
        out_ptrs = torch.arange(out_start, out_start + total * slot, slot, dtype=torch.int64)
        comp_stored = torch.where(use, stored, 0)
        if k == 1:
            comp_padded = _align8(comp_stored)
            temp = torch.empty(int(comp_padded.sum()), dtype=torch.uint8, device=device)
            home = temp.data_ptr() + torch.cumsum(comp_padded, 0) - comp_padded
        else:
            home = in_ptrs
        # Then pack every chunk into comp, zero-filling each chunk's padding.
        dst = out_start + data_off + ends - padded
        tbl = torch.stack(
            [out_ptrs, home, comp_stored, torch.where(use, home, in_ptrs), dst, stored]
        ).to(device, non_blocking=True)
        _kernels.copy_chunks(tbl[0], tbl[1], tbl[2], chunk_bytes, pad=False)
        _kernels.copy_chunks(tbl[3], tbl[4], tbl[5], chunk_bytes, pad=True)
        # Only after the first pass, which reads ANS chunks from the slots this overwrites.
        comp[:data_off].copy_(prefix, non_blocking=True)

        crc = None
        if checksum_chunk_bytes is not None:
            crc = _crc32(comp[:size], checksum_chunk_bytes)
        blob = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=pin_memory)
        blob.copy_(comp[:size])
        return blob, None if crc is None else crc.cpu()


def _parse_header(
    prefix: Buffer,
) -> tuple[torch.dtype, tuple[int, ...], int, int, int, int]:
    prefix = memoryview(prefix)
    if len(prefix) < _FIXED.size:
        raise ValueError("blob is too short")
    magic, version, code, ndim, flags, chunk_bytes, n_comp, n = _FIXED.unpack_from(prefix)
    if magic != _MAGIC or version != _VERSION or code not in _DTYPES:
        raise ValueError("not a cuzipnn blob")
    if len(prefix) < _FIXED.size + 8 * ndim:
        raise ValueError("blob is too short")
    shape = struct.unpack_from(f"<{ndim}q", prefix, _FIXED.size)
    return _DTYPES[code], shape, flags, chunk_bytes, n_comp, n


def decompress_tensor(
    blob: torch.Tensor, *, device: torch.device | str | int | None = None
) -> torch.Tensor:
    """Decompress a blob produced by compress_tensor; the blob may be on the CPU or the GPU.

    The result is placed on `device` (default: the blob's device if it is on the GPU, else the current CUDA device).
    Decoding always runs on the GPU; a non-CUDA `device` such as "cpu" receives a copy of the result.

    Passing a blob that is not from `compress_tensor` results in undefined behavior.
    """
    if blob.dtype != torch.uint8 or blob.dim() != 1:
        raise ValueError("blob must be a 1-D uint8 tensor")

    blob = blob.detach()
    prefix_len = min(blob.numel(), _FIXED.size + 8 * _MAX_NDIM)
    prefix_tensor = (
        blob[:prefix_len].to("cpu", memory_format=torch.contiguous_format).contiguous()
    )
    prefix = (ctypes.c_byte * prefix_len).from_address(prefix_tensor.data_ptr())
    dtype, shape, flags, chunk_bytes, n_comp, n = _parse_header(prefix)
    reorder = bool(flags & _FLAG_REORDER)

    device = (
        torch.device(device)
        if device is not None
        else blob.device
        if blob.is_cuda
        else torch.device("cuda", torch.cuda.current_device())
    )

    if flags & _FLAG_STORED:
        # Stored uncompressed: copy the bytes after the header straight to the target device.
        header_size = _FIXED.size + 8 * len(shape)
        out = torch.empty(shape, dtype=dtype, device=device)
        nbytes = out.numel() * out.element_size()
        if blob.numel() < header_size + nbytes:
            raise ValueError("blob is truncated")
        out.view(-1).view(torch.uint8).copy_(blob[header_size : header_size + nbytes])
        return out

    # Decode on the GPU regardless; a non-CUDA target gets a copy at the end.
    out_device = device
    if device.type != "cuda":
        device = torch.device("cuda", torch.cuda.current_device())

    if n == 0:
        return torch.empty(shape, dtype=dtype, device=out_device)
    out = torch.empty(shape, dtype=dtype, device=device)

    k = out.element_size()
    nch = triton.cdiv(n, chunk_bytes)
    total = k * nch
    sizes_off, idx_off, data_off = _layout(len(shape), total, n_comp)
    if blob.numel() < data_off:
        raise ValueError("blob is truncated")
    if k == 1:
        table = (
            blob[sizes_off:idx_off]
            .to("cpu", memory_format=torch.contiguous_format)
            .contiguous()
        )
        if table.storage_offset() % 4:
            table = table.clone()
        padded_cpu = _align8(table.view(torch.int32).to(torch.int64))
        # nvCOMP reads the compressed chunks, so the data section must be complete before it runs.
        if blob.numel() < data_off + padded_cpu.sum().item():
            raise ValueError("blob is truncated")

    with torch.cuda.device(device):
        x = blob.to(
            device, non_blocking=True, memory_format=torch.contiguous_format
        ).contiguous()
        if x.data_ptr() % 8:
            x = x.clone()
        # A pinned blob is uploaded asynchronously, but the caller may reuse it once we return.
        uploaded = torch.cuda.Event() if blob.is_pinned() else None
        if uploaded is not None:
            uploaded.record()

        stored = x[sizes_off:idx_off].view(torch.int32).to(torch.int64)
        comp_idx = x[idx_off : idx_off + 4 * n_comp].view(torch.int32).to(torch.int64)
        padded = _align8(stored)
        src = x.data_ptr() + data_off + (torch.cumsum(padded, 0) - padded)
        raw = _raw_bytes(k, nch, n, chunk_bytes, 1, device)

        # Single-stream dtypes decode straight into the output; others go through a planar buffer.
        planar = (
            out.view(-1).view(torch.uint8)
            if k == 1
            else torch.empty(total * chunk_bytes, dtype=torch.uint8, device=device)
        )
        planar_base = planar.data_ptr()
        slots = torch.arange(
            planar_base,
            planar_base + total * chunk_bytes,
            chunk_bytes,
            dtype=torch.int64,
            device=device,
        )

        if n_comp:
            _nvcomp.decompress(
                src[comp_idx],
                stored[comp_idx],
                raw[comp_idx],
                chunk_bytes,
                n_comp * chunk_bytes,
                slots[comp_idx],
            )

        if k == 1:
            # nvCOMP decodes the ANS chunks into their slots; copy the raw chunks into theirs.
            raw_only = raw.index_fill(0, comp_idx, 0)
            _kernels.copy_chunks(src, slots, raw_only, chunk_bytes, pad=False)
        else:
            # Each stream's chunk comes from the planar buffer if ANS-decoded, else straight from the blob.
            tbl = src.index_copy(0, comp_idx, slots[comp_idx]) - planar_base
            _kernels.merge(tbl, planar, out.view(-1), chunk_bytes, reorder)

        if uploaded is not None:
            uploaded.synchronize()
        return out.to(out_device)
