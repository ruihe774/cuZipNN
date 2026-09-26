"""ZipNN on the GPU: bit reordering and byte grouping in Triton, entropy coding with nvCOMP ANS.

Blob layout (little-endian, every section 8-byte aligned):

    0   magic "ZNNG" | u8 version | u8 dtype code | u8 ndim | u8 flags
    8   u32 chunk_bytes | u32 n_comp
    16  u64 numel
    24  i64 shape[ndim]
    ..  i32 stored_bytes[K * nch]  bytes stored for each (stream, chunk), ANS-compressed or raw
    ..  i32 comp_idx[n_comp]       indices of the ANS-compressed chunks
    ..  data                       chunks in (stream, chunk) order, each padded to 8 bytes

K is the element size in bytes (one stream per byte), and nch = ceil(numel / chunk_bytes).
Chunk j of every stream covers elements [j * chunk_bytes, (j + 1) * chunk_bytes).
"""

import ctypes
import struct

import torch
import triton

from . import _cudart, _kernels, _nvcomp

_MAGIC = b"ZNNG"
_VERSION = 1
_FIXED = struct.Struct("<4sBBBBIIQ")
_FLAG_REORDER = 1
_MAX_NDIM = 64

DEFAULT_CHUNK_BYTES = 128 * 1024
DEFAULT_SUB_CHUNK_BYTES = 8 * 1024
DEFAULT_THRESHOLD = 0.95

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


def compress_tensor(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    sub_chunk_bytes: int = DEFAULT_SUB_CHUNK_BYTES,
    threshold: float = DEFAULT_THRESHOLD,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Compress a tensor losslessly on the GPU; returns a 1-D uint8 tensor in CPU memory."""
    if tensor.dtype not in _CODES:
        raise TypeError(f"unsupported dtype {tensor.dtype}")
    if chunk_bytes % 8 or not 0 < chunk_bytes <= _nvcomp.MAX_CHUNK_BYTES:
        raise ValueError(
            f"chunk_bytes must be a multiple of 8 in (0, {_nvcomp.MAX_CHUNK_BYTES}]"
        )
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in (0, 1]")
    if sub_chunk_bytes <= 0:
        raise ValueError("sub_chunk_bytes must be positive")
    sub_chunks = _sub_chunks(chunk_bytes, sub_chunk_bytes)
    if tensor.dim() > _MAX_NDIM:
        raise ValueError(f"at most {_MAX_NDIM} dimensions are supported")

    x = tensor.detach()

    k, n, reorder = x.element_size(), x.numel(), x.dtype in _REORDER
    nch = triton.cdiv(n, chunk_bytes)
    total = k * nch

    def header(n_comp: int, size: int | None = None) -> torch.Tensor:
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
            _FLAG_REORDER if reorder else 0,
            chunk_bytes,
            n_comp,
            n,
        )
        shape_spec.pack_into(b, _FIXED.size, *tensor.shape)
        return r

    if n == 0:
        return header(0)

    device = x.device if x.is_cuda else torch.device("cuda", torch.cuda.current_device())
    current_stream = torch.cuda.current_stream(device)
    if current_stream.cuda_stream != 0:
        stream = current_stream
    else:
        stream = torch.cuda.Stream(device)
        stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
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
        # Zeroed: nvCOMP's rANS leaves some bytes within comp_bytes unwritten, which would
        # otherwise leak stale device memory into the blob and make it non-deterministic.
        comp = torch.zeros(total * slot, dtype=torch.uint8, device=device)
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
            stream,
        )

        raw_bytes = _raw_bytes(k, nch, n, chunk_bytes, 1, "cpu")
        threshold_bytes = _raw_bytes(k, nch, n, chunk_bytes, threshold, "cpu")
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
        blob = header(n_comp, size=data_off + int(ends[-1]))
        blob[sizes_off:idx_off].view(torch.int32).copy_(stored)
        blob[idx_off : idx_off + 4 * n_comp].view(torch.int32).copy_(comp_idx)
        blob[idx_off + 4 * n_comp : data_off].zero_()

        in_ptrs = torch.arange(
            in_start,
            in_start + total * chunk_bytes,
            chunk_bytes,
            dtype=torch.int64,
            device="cpu",
        )
        out_ptrs = torch.arange(
            out_start, out_start + total * slot, slot, dtype=torch.int64, device="cpu"
        )
        src = torch.where(use, out_ptrs, in_ptrs)
        dst = blob.data_ptr() + data_off + (ends - padded)
        # One copy per run of chunks that are back to back in both the source and the blob
        # (consecutive raw chunks), which collapses incompressible streams into a few copies.
        head = torch.ones(total, dtype=torch.bool)
        head[1:] = src[:-1] + padded[:-1] != src[1:]
        last = head.roll(-1)
        _cudart.memcpy_batch(dst[head], src[head], dst[last] + padded[last] - dst[head], stream)
        stream.synchronize()

        # The copies run through each chunk's 8-byte padding, which picks up whatever follows it
        # in the source; keep only the stored bytes of each chunk's last word.
        tail = stored % 8
        part = tail.nonzero().view(-1)
        words = blob[data_off:].view(torch.int64)
        words[ends[part] // 8 - 1] &= (1 << 8 * tail[part]) - 1
        return blob


def _parse_header(
    prefix: bytes,
) -> tuple[torch.dtype, tuple[int, ...], bool, int, int, int]:
    if len(prefix) < _FIXED.size:
        raise ValueError("blob is too short")
    magic, version, code, ndim, flags, chunk_bytes, n_comp, n = _FIXED.unpack_from(prefix)
    if magic != _MAGIC or version != _VERSION or code not in _DTYPES:
        raise ValueError("not a cuzipnn blob")
    if len(prefix) < _FIXED.size + 8 * ndim:
        raise ValueError("blob is too short")
    shape = struct.unpack_from(f"<{ndim}q", prefix, _FIXED.size)
    return _DTYPES[code], shape, bool(flags & _FLAG_REORDER), chunk_bytes, n_comp, n


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
    prefix = ctypes.string_at(prefix_tensor.data_ptr(), prefix_len)
    dtype, shape, reorder, chunk_bytes, n_comp, n = _parse_header(prefix)

    device = (
        torch.device(device)
        if device is not None
        else blob.device
        if blob.is_cuda
        else torch.device("cuda", torch.cuda.current_device())
    )
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
    # For k == 1: (stored bytes, padded bytes, comp_idx) of every chunk, on the CPU.
    cpu_table: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if k == 1:
        table = (
            blob[sizes_off : idx_off + 4 * n_comp]
            .to("cpu", memory_format=torch.contiguous_format)
            .contiguous()
        )
        if table.storage_offset() % 4:
            table = table.clone()
        stored_cpu = table[: 4 * total].view(torch.int32).to(torch.int64)
        padded_cpu = _align8(stored_cpu)
        # nvCOMP reads the compressed chunks, so the data section must be complete before it runs.
        if blob.numel() < data_off + padded_cpu.sum().item():
            raise ValueError("blob is truncated")
        cpu_table = stored_cpu, padded_cpu, table[4 * total :].view(torch.int32)

    current_stream = torch.cuda.current_stream(device)
    if current_stream.cuda_stream != 0:
        stream = current_stream
    else:
        stream = torch.cuda.Stream(device)
        stream.wait_stream(current_stream)

    with torch.cuda.stream(stream):
        x = blob.to(
            device, non_blocking=True, memory_format=torch.contiguous_format
        ).contiguous()
        if x.data_ptr() % 8:
            x = x.clone()
        # A pinned blob is uploaded asynchronously, but the caller may reuse it once we return.
        uploaded = torch.cuda.Event() if blob.is_pinned() else None
        if uploaded is not None:
            uploaded.record(stream)

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
                stream,
            )

        if cpu_table is not None:
            stored_cpu, padded_cpu, comp_idx_cpu = cpu_table
            # Copy the raw chunks into place; nvCOMP decodes the ANS chunks into their slots.
            # The copy tables are built on the CPU, from a small copy of the blob's chunk table.
            is_raw = torch.ones(total, dtype=torch.bool, device="cpu")
            is_raw[comp_idx_cpu] = False
            src_cpu = x.data_ptr() + data_off + (torch.cumsum(padded_cpu, 0) - padded_cpu)
            raw_src = src_cpu[is_raw]
            raw_dst = torch.arange(
                planar_base,
                planar_base + total * chunk_bytes,
                chunk_bytes,
                dtype=torch.int64,
                device="cpu",
            )[is_raw]
            raw_size = padded_cpu[is_raw]
            if raw_size.numel():
                # One copy per run of raw chunks that are back to back in both the blob and the output.
                head = torch.ones(raw_size.numel(), dtype=torch.bool, device="cpu")
                head[1:] = raw_src[:-1] + raw_size[:-1] != raw_src[1:]
                last = head.roll(-1)
                _cudart.memcpy_batch(
                    raw_dst[head],
                    raw_src[head],
                    raw_dst[last] + raw_size[last] - raw_dst[head],
                    stream,
                )
        else:
            # Each stream's chunk comes from the planar buffer if ANS-decoded, else straight from the blob.
            tbl = src.index_copy(0, comp_idx, slots[comp_idx])
            _kernels.merge(tbl, out.view(-1), chunk_bytes, reorder)

        current_stream.wait_stream(stream)
        if uploaded is not None:
            uploaded.synchronize()
        return out.to(out_device)
