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

import struct

import torch
import triton

from . import _kernels, _nvcomp

_MAGIC = b"ZNNG"
_VERSION = 1
_FIXED = struct.Struct("<4sBBBBIIQ")
_FLAG_REORDER = 1
_MAX_NDIM = 64

DEFAULT_CHUNK_BYTES = 128 * 1024
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


def _align8(x):
    return (x + 7) & ~7


def _raw_bytes(k: int, nch: int, n: int, chunk_bytes: int, device) -> torch.Tensor:
    """Uncompressed size of each (stream, chunk): chunk_bytes, except for each stream's last chunk."""
    raw = torch.full((k, nch), chunk_bytes, dtype=torch.int64, device=device)
    raw[:, -1] = n - (nch - 1) * chunk_bytes
    return raw.flatten()


def _layout(ndim: int, total: int, n_comp: int):
    sizes_off = 24 + 8 * ndim
    idx_off = sizes_off + 4 * total
    data_off = _align8(idx_off + 4 * n_comp)
    return sizes_off, idx_off, data_off


def compress_tensor(
    tensor: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    threshold: float = DEFAULT_THRESHOLD,
) -> torch.Tensor:
    """Compress a tensor losslessly on the GPU; returns a 1-D uint8 CUDA tensor."""
    if tensor.dtype not in _CODES:
        raise TypeError(f"unsupported dtype {tensor.dtype}")
    if chunk_bytes % 8 or not 0 < chunk_bytes <= _nvcomp.MAX_CHUNK_BYTES:
        raise ValueError(
            f"chunk_bytes must be a multiple of 8 in (0, {_nvcomp.MAX_CHUNK_BYTES}]"
        )
    if tensor.dim() > _MAX_NDIM:
        raise ValueError(f"at most {_MAX_NDIM} dimensions are supported")

    x = tensor.detach()
    device = x.device if x.is_cuda else torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        x = x.to(device).flatten()
        k, n, reorder = x.element_size(), x.numel(), x.dtype in _REORDER
        nch = triton.cdiv(n, chunk_bytes)
        total = k * nch
        header = _FIXED.pack(
            _MAGIC,
            _VERSION,
            _CODES[x.dtype],
            tensor.dim(),
            _FLAG_REORDER if reorder else 0,
            chunk_bytes,
            0,
            n,
        ) + struct.pack(f"<{tensor.dim()}q", *tensor.shape)

        if n == 0:
            return torch.frombuffer(bytearray(header), dtype=torch.uint8).to(device)

        # Bit reordering + byte grouping into K contiguous streams of nch chunks each.
        if k == 1 and x.data_ptr() % 8 == 0:
            planar = x.view(torch.uint8)
        else:
            planar = torch.empty(total * chunk_bytes, dtype=torch.uint8, device=device)
            _kernels.split(x, planar, nch * chunk_bytes, reorder)

        # Entropy-code every (stream, chunk) in one nvCOMP batch.
        idx = torch.arange(total, dtype=torch.int64, device=device)
        raw = _raw_bytes(k, nch, n, chunk_bytes, device)
        in_ptrs = planar.data_ptr() + idx * chunk_bytes
        slot = _align8(_nvcomp.max_compressed_chunk_bytes(chunk_bytes))
        comp = torch.empty(total * slot, dtype=torch.uint8, device=device)
        out_ptrs = comp.data_ptr() + idx * slot
        comp_bytes = torch.empty(total, dtype=torch.int64, device=device)
        statuses = torch.empty(total, dtype=torch.int32, device=device)
        _nvcomp.compress(in_ptrs, raw, chunk_bytes, k * n, out_ptrs, comp_bytes, statuses)

        # Keep a chunk compressed only if it saves enough; otherwise store it raw (ZipNN's threshold rule).
        use = (statuses == _nvcomp.NVCOMP_SUCCESS) & (comp_bytes < raw * threshold)
        stored = torch.where(use, comp_bytes, raw)
        padded = _align8(stored)
        starts = torch.cumsum(padded, 0) - padded
        n_comp, data_bytes = torch.stack([use.sum(), starts[-1] + padded[-1]]).tolist()
        comp_idx = torch.nonzero_static(use, size=n_comp).flatten()

        sizes_off, idx_off, data_off = _layout(tensor.dim(), total, n_comp)
        blob = torch.empty(data_off + data_bytes, dtype=torch.uint8, device=device)
        header = header[:12] + struct.pack("<I", n_comp) + header[16:]
        blob[:sizes_off].copy_(torch.frombuffer(bytearray(header), dtype=torch.uint8))
        blob[sizes_off:idx_off].view(torch.int32).copy_(stored)
        blob[idx_off : idx_off + 4 * n_comp].view(torch.int32).copy_(comp_idx)
        blob[idx_off + 4 * n_comp : data_off].zero_()

        src = torch.where(use, out_ptrs, in_ptrs)
        dst = blob.data_ptr() + data_off + starts
        _kernels.copy_chunks(src, dst, stored, chunk_bytes, pad=True)
        return blob


def _parse_header(prefix: bytes):
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
    """
    if blob.dtype != torch.uint8 or blob.dim() != 1:
        raise ValueError("blob must be a 1-D uint8 tensor")
    prefix_len = min(blob.numel(), _FIXED.size + 8 * _MAX_NDIM)
    prefix = bytes(blob[:prefix_len].cpu().tolist())
    dtype, shape, reorder, chunk_bytes, n_comp, n = _parse_header(prefix)

    if device is None:
        device = (
            blob.device if blob.is_cuda else torch.device("cuda", torch.cuda.current_device())
        )
    else:
        device = torch.device(device)
    with torch.cuda.device(device):
        out = torch.empty(shape, dtype=dtype, device=device)
        if n == 0:
            return out
        blob = blob.to(device, non_blocking=True)
        if blob.data_ptr() % 8:
            blob = blob.clone()

        k = out.element_size()
        nch = triton.cdiv(n, chunk_bytes)
        total = k * nch
        sizes_off, idx_off, data_off = _layout(len(shape), total, n_comp)
        if blob.numel() < data_off:
            raise ValueError("blob is truncated")

        stored = blob[sizes_off:idx_off].view(torch.int32).to(torch.int64)
        comp_idx = blob[idx_off : idx_off + 4 * n_comp].view(torch.int32).to(torch.int64)
        padded = _align8(stored)
        src = blob.data_ptr() + data_off + torch.cumsum(padded, 0) - padded
        raw = _raw_bytes(k, nch, n, chunk_bytes, device)

        # Single-stream dtypes decode straight into the output; others go through a planar buffer.
        planar = (
            out.flatten().view(torch.uint8)
            if k == 1
            else torch.empty(total * chunk_bytes, dtype=torch.uint8, device=device)
        )
        slots = (
            planar.data_ptr()
            + torch.arange(total, dtype=torch.int64, device=device) * chunk_bytes
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
            # Copy the raw chunks into place; the ANS chunks are already there.
            _kernels.copy_chunks(
                src, slots, raw.index_fill(0, comp_idx, 0), chunk_bytes, pad=False
            )
        else:
            # Each stream's chunk comes from the planar buffer if ANS-decoded, else straight from the blob.
            tbl = src.index_copy(0, comp_idx, slots[comp_idx])
            _kernels.merge(tbl, out.flatten(), chunk_bytes, reorder)
        return out
