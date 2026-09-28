"""Triton kernels for ZipNN's bit reordering, byte grouping, and chunk packing."""

import torch
import triton
import triton.language as tl

_SPLIT_BLOCK = 1024
_MERGE_BLOCK = 4096
_COPY_BLOCK = 1024  # 8-byte words
_UNSIGNED = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32, 8: torch.uint64}


def _word_type(k: int) -> tl.dtype:
    return tl.uint64 if k == 8 else tl.uint32


@triton.jit
def _reorder(u, W: tl.constexpr):
    # ZipNN bit reorder: `s eeeeeeee m...` -> `eeeeeeee s m...`, putting the whole exponent in the MSByte.
    exp = (u >> (W - 9)) & 0xFF
    sign = u >> (W - 1)
    mant = u & ((1 << (W - 9)) - 1)  # pyright: ignore[reportOperatorIssue]
    return (exp << (W - 8)) | (sign << (W - 9)) | mant


@triton.jit
def _revert(u, W: tl.constexpr):
    exp = u >> (W - 8)
    sign = (u >> (W - 9)) & 1
    mant = u & ((1 << (W - 9)) - 1)  # pyright: ignore[reportOperatorIssue]
    return (sign << (W - 1)) | (exp << (W - 9)) | mant


@triton.jit
def _split_kernel(
    x_ptr,
    planar_ptr,
    n,
    stride,
    K: tl.constexpr,
    WT: tl.constexpr,
    REORDER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Byte grouping: byte b of element e goes to planar[b * stride + e]."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    u = tl.load(x_ptr + offs, mask=mask, other=0).to(WT)
    if REORDER:
        u = _reorder(u, 8 * K)
    # stride is an i32 argument when it fits, but (K - 1) * stride may not.
    stride = stride.to(tl.int64)
    for b in tl.static_range(K):  # pyright: ignore[reportGeneralTypeIssues]
        tl.store(planar_ptr + b * stride + offs, (u >> (8 * b)).to(tl.uint8), mask=mask)


@triton.jit
def _merge_streams(
    src_tbl,
    base_ptr,
    out_ptr,
    j,
    nch,
    offs,
    e,
    mask,
    K: tl.constexpr,
    WT: tl.constexpr,
    REORDER: tl.constexpr,
):
    u = tl.zeros(offs.shape, WT)
    for b in tl.static_range(K):  # pyright: ignore[reportGeneralTypeIssues]
        # Every chunk starts 16-byte aligned; the hint lets Triton vectorize the byte loads.
        src = base_ptr + tl.multiple_of(tl.load(src_tbl + b * nch + j), 16)
        u |= tl.load(src + offs, mask=mask, other=0).to(WT) << (8 * b)
    if REORDER:
        u = _revert(u, 8 * K)
    tl.store(out_ptr + e, u.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _merge_kernel(
    src_tbl,
    base_ptr,
    out_ptr,
    n,
    nch,
    C: tl.constexpr,
    K: tl.constexpr,
    WT: tl.constexpr,
    REORDER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Inverse of _split_kernel for chunk j; stream b's chunk j is at base_ptr + src_tbl[b * nch + j]."""
    j = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    e = j.to(tl.int64) * C + offs
    # Only the last chunk can be partial. Masking the others by C alone keeps their accesses
    # vectorized even when n is not a multiple of the vector width.
    if j < nch - 1:
        _merge_streams(src_tbl, base_ptr, out_ptr, j, nch, offs, e, offs < C, K, WT, REORDER)
    else:
        _merge_streams(
            src_tbl, base_ptr, out_ptr, j, nch, offs, e, (offs < C) & (e < n), K, WT, REORDER
        )


@triton.jit
def _copy_chunks_kernel(src_tbl, dst_tbl, nbytes_ptr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    """Copy chunk i (nbytes[i] bytes) from src_tbl[i] to dst_tbl[i]; both addresses are 16-byte aligned.

    With PAD, the destination is zero-filled up to the next multiple of 16 bytes.
    """
    i = tl.program_id(0)
    t = tl.program_id(1)
    nb = tl.load(nbytes_ptr + i)
    if t * BLOCK * 8 < nb:
        src = tl.load(src_tbl + i)
        dst = tl.load(dst_tbl + i)
        # Whole 16-byte units as pairs of words; the rest, up to 15 bytes, byte by byte.
        nw = nb // 16 * 2
        w = t * BLOCK + tl.arange(0, BLOCK)
        m = w < nw
        # The hint must go on the uint64 pointers; one on the integer address is lost in the cast.
        src64 = tl.multiple_of(src.to(tl.pointer_type(tl.uint64)), 16)
        dst64 = tl.multiple_of(dst.to(tl.pointer_type(tl.uint64)), 16)
        v = tl.load(src64 + w, mask=m)
        tl.store(dst64 + w, v, mask=m)
        if (nb > nw * 8) & (t == nw // BLOCK):
            k = nw * 8 + tl.arange(0, 16)
            b = tl.load(src.to(tl.pointer_type(tl.uint8)) + k, mask=k < nb, other=0)
            if PAD:
                tl.store(dst.to(tl.pointer_type(tl.uint8)) + k, b)
            else:
                tl.store(dst.to(tl.pointer_type(tl.uint8)) + k, b, mask=k < nb)


def split(x: torch.Tensor, planar: torch.Tensor, stride: int, reorder: bool) -> None:
    n = x.numel()
    k = x.element_size()
    _split_kernel[(triton.cdiv(n, _SPLIT_BLOCK),)](
        x.view(_UNSIGNED[k]),
        planar,
        n,
        stride,
        K=k,
        WT=_word_type(k),
        REORDER=reorder,
        BLOCK=_SPLIT_BLOCK,
        # ~4 elements per thread for k=1 down to 1 for k>=4; ~2-4% faster than a fixed 8 for 4/8-byte dtypes on GB10.
        num_warps=min(8 * k, 32),  # pyright: ignore[reportCallIssue]
    )


def merge(
    src_tbl: torch.Tensor,
    base: torch.Tensor,
    out: torch.Tensor,
    chunk_bytes: int,
    reorder: bool,
) -> None:
    """src_tbl holds int64 byte offsets from base.data_ptr(), each a multiple of 16."""
    n = out.numel()
    k = out.element_size()
    nch = triton.cdiv(n, chunk_bytes)
    grid = (nch, triton.cdiv(min(chunk_bytes, n), _MERGE_BLOCK))
    _merge_kernel[grid](
        src_tbl,
        base,
        out.view(_UNSIGNED[k]),
        n,
        nch,
        C=chunk_bytes,
        K=k,
        WT=_word_type(k),
        REORDER=reorder,
        BLOCK=_MERGE_BLOCK,
        # 4 elements per thread; 2-5% faster than 8 warps for 2/4/8-byte dtypes on GB10.
        num_warps=32,  # pyright: ignore[reportCallIssue]
    )


def copy_chunks(
    src_tbl: torch.Tensor,
    dst_tbl: torch.Tensor,
    nbytes: torch.Tensor,
    max_bytes: int,
    pad: bool,
) -> None:
    """src_tbl/dst_tbl hold int64 device addresses; a chunk with nbytes 0 is skipped."""
    grid = (src_tbl.numel(), triton.cdiv(max_bytes, 8 * _COPY_BLOCK))
    # 8 words per thread; BLOCK 256-2048 x 1-16 warps are all within ~4% of this on GB10.
    _copy_chunks_kernel[grid](src_tbl, dst_tbl, nbytes, PAD=pad, BLOCK=_COPY_BLOCK, num_warps=4)  # pyright: ignore[reportCallIssue]
