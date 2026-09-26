"""Triton kernels for ZipNN's bit reordering, byte grouping, and chunk packing."""

import torch
import triton
import triton.language as tl

_UNSIGNED = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32, 8: torch.uint64}


@triton.jit
def _reorder(u, W: tl.constexpr):
    # ZipNN bit reorder: `s eeeeeeee m...` -> `eeeeeeee s m...`, putting the whole exponent in the MSByte.
    exp = (u >> (W - 9)) & 0xFF
    sign = u >> (W - 1)
    mant = u & ((1 << (W - 9)) - 1)
    return (exp << (W - 8)) | (sign << (W - 9)) | mant


@triton.jit
def _revert(u, W: tl.constexpr):
    exp = u >> (W - 8)
    sign = (u >> (W - 9)) & 1
    mant = u & ((1 << (W - 9)) - 1)
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
    for b in tl.static_range(K):
        tl.store(planar_ptr + b * stride + offs, (u >> (8 * b)).to(tl.uint8), mask=mask)


@triton.jit
def _merge_kernel(
    src_tbl,
    out_ptr,
    n,
    nch,
    C: tl.constexpr,
    K: tl.constexpr,
    WT: tl.constexpr,
    REORDER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Inverse of _split_kernel for chunk j; src_tbl[b * nch + j] is the address of stream b's chunk j."""
    j = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    e = j.to(tl.int64) * C + offs
    mask = (offs < C) & (e < n)
    u = tl.zeros((BLOCK,), WT)
    for b in tl.static_range(K):
        src = tl.load(src_tbl + b * nch + j).to(tl.pointer_type(tl.uint8))
        u |= tl.load(src + offs, mask=mask, other=0).to(WT) << (8 * b)
    if REORDER:
        u = _revert(u, 8 * K)
    tl.store(out_ptr + e, u.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _copy_chunks_kernel(src_tbl, dst_tbl, nbytes_ptr, PAD: tl.constexpr, BLOCK: tl.constexpr):
    """Copy chunk i (nbytes[i] bytes) from src_tbl[i] to dst_tbl[i]; both addresses are 8-byte aligned.

    With PAD, the destination is zero-filled up to the next multiple of 8 bytes.
    """
    i = tl.program_id(0)
    t = tl.program_id(1)
    nb = tl.load(nbytes_ptr + i)
    if t * BLOCK * 8 < nb:
        src = tl.load(src_tbl + i)
        dst = tl.load(dst_tbl + i)
        nw = nb // 8
        w = t * BLOCK + tl.arange(0, BLOCK)
        m = w < nw
        v = tl.load(src.to(tl.pointer_type(tl.uint64)) + w, mask=m)
        tl.store(dst.to(tl.pointer_type(tl.uint64)) + w, v, mask=m)
        if (nb > nw * 8) & (t == nw // BLOCK):
            k = nw * 8 + tl.arange(0, 8)
            b = tl.load(src.to(tl.pointer_type(tl.uint8)) + k, mask=k < nb, other=0)
            if PAD:
                tl.store(dst.to(tl.pointer_type(tl.uint8)) + k, b)
            else:
                tl.store(dst.to(tl.pointer_type(tl.uint8)) + k, b, mask=k < nb)


_SPLIT_BLOCK = 1024
_MERGE_BLOCK = 4096
_COPY_BLOCK = 1024  # 8-byte words


def _word_type(k: int):
    return tl.uint64 if k == 8 else tl.uint32


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
        num_warps=8,
    )


def merge(src_tbl: torch.Tensor, out: torch.Tensor, chunk_bytes: int, reorder: bool) -> None:
    n = out.numel()
    k = out.element_size()
    nch = triton.cdiv(n, chunk_bytes)
    grid = (nch, triton.cdiv(min(chunk_bytes, n), _MERGE_BLOCK))
    _merge_kernel[grid](
        src_tbl,
        out.view(_UNSIGNED[k]),
        n,
        nch,
        C=chunk_bytes,
        K=k,
        WT=_word_type(k),
        REORDER=reorder,
        BLOCK=_MERGE_BLOCK,
        num_warps=8,
    )


def copy_chunks(
    src_tbl: torch.Tensor,
    dst_tbl: torch.Tensor,
    nbytes: torch.Tensor,
    max_bytes: int,
    pad: bool,
) -> None:
    grid = (src_tbl.numel(), triton.cdiv(max_bytes, 8 * _COPY_BLOCK))
    _copy_chunks_kernel[grid](src_tbl, dst_tbl, nbytes, PAD=pad, BLOCK=_COPY_BLOCK, num_warps=4)
