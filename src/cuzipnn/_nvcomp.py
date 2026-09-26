"""Minimal ctypes bindings for nvCOMP's batched ANS (rANS entropy coder) C API."""

import ctypes
import importlib

import torch

NVCOMP_TYPE_CHAR = 0
NVCOMP_SUCCESS = 0

# From nvcomp/ans.h
MAX_CHUNK_BYTES = 1 << 24
DECOMPRESS_INPUT_ALIGNMENT = 8
# Explicit max_sub_chunk_count must be a power of 2 in this range (0 means auto).
MIN_SUB_CHUNKS = 4
MAX_SUB_CHUNKS = 64


class _CompressOpts(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),  # nvcompANSType_t: nvcomp_rANS == 0
        ("data_type", ctypes.c_int),
        ("max_sub_chunk_count", ctypes.c_uint8),
        ("reserved", ctypes.c_char * 55),
    ]


class _DecompressOpts(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),  # NVCOMP_DECOMPRESS_BACKEND_DEFAULT == 0
        ("data_type", ctypes.c_int),
        ("max_sub_chunk_count", ctypes.c_uint8),
        ("reserved", ctypes.c_char * 55),
    ]


# 0 = autodetect the sub-chunk count from the bitstream, so any blob decodes.
_DECOMPRESS_OPTS = _DecompressOpts(0, NVCOMP_TYPE_CHAR, 0)


def _compress_opts(sub_chunks: int) -> _CompressOpts:
    return _CompressOpts(0, NVCOMP_TYPE_CHAR, sub_chunks)


# Not `import nvidia.libnvcomp`: importing nvidia.nvcomp first deletes that attribute from the namespace package.
_lib = importlib.import_module("nvidia.libnvcomp").load_library()
if _lib is None:
    raise ImportError("could not load libnvcomp.so.5")

_size_t = ctypes.c_size_t
_ptr = ctypes.c_void_p
_size_p = ctypes.POINTER(ctypes.c_size_t)


def _bind(name, *argtypes):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


_max_output = _bind(
    "nvcompBatchedANSCompressGetMaxOutputChunkSize", _size_t, _CompressOpts, _size_p
)
_compress_temp = _bind(
    "nvcompBatchedANSCompressGetTempSizeAsync",
    _size_t,
    _size_t,
    _CompressOpts,
    _size_p,
    _size_t,
)
_compress = _bind(
    "nvcompBatchedANSCompressAsync",
    _ptr,
    _ptr,
    _size_t,
    _size_t,
    _ptr,
    _size_t,
    _ptr,
    _ptr,
    _CompressOpts,
    _ptr,
    _ptr,
)
_decompress_temp = _bind(
    "nvcompBatchedANSDecompressGetTempSizeAsync",
    _size_t,
    _size_t,
    _DecompressOpts,
    _size_p,
    _size_t,
)
_decompress = _bind(
    "nvcompBatchedANSDecompressAsync",
    _ptr,
    _ptr,
    _ptr,
    _ptr,
    _size_t,
    _ptr,
    _size_t,
    _ptr,
    _DecompressOpts,
    _ptr,
    _ptr,
)


def _check(status, what):
    if status != NVCOMP_SUCCESS:
        raise RuntimeError(f"nvCOMP {what} failed with status {status}")


def max_compressed_chunk_bytes(chunk_bytes: int, sub_chunks: int) -> int:
    out = _size_t()
    _check(
        _max_output(chunk_bytes, _compress_opts(sub_chunks), ctypes.byref(out)),
        "GetMaxOutputChunkSize",
    )
    return out.value


def compress(
    in_ptrs: torch.Tensor,
    in_bytes: torch.Tensor,
    chunk_bytes: int,
    total_bytes: int,
    out_ptrs: torch.Tensor,
    out_bytes: torch.Tensor,
    statuses: torch.Tensor,
    sub_chunks: int,
    stream: int,
) -> None:
    """Asynchronously ANS-compress a batch of chunks on `stream`. All tensor arguments live on the GPU.

    in_ptrs/out_ptrs are int64 device addresses; in_bytes/out_bytes are int64 (size_t);
    statuses is int32 and receives one nvcompStatus_t per chunk. Each chunk is coded as
    `sub_chunks` independently decodable sub-chunks (0 lets nvCOMP choose).
    """
    n = in_ptrs.numel()
    opts = _compress_opts(sub_chunks)
    temp_bytes = _size_t()
    _check(
        _compress_temp(n, chunk_bytes, opts, ctypes.byref(temp_bytes), total_bytes),
        "CompressGetTempSize",
    )
    temp = torch.empty(max(temp_bytes.value, 1), dtype=torch.uint8, device=in_ptrs.device)
    _check(
        _compress(
            in_ptrs.data_ptr(),
            in_bytes.data_ptr(),
            chunk_bytes,
            n,
            temp.data_ptr(),
            temp_bytes.value,
            out_ptrs.data_ptr(),
            out_bytes.data_ptr(),
            opts,
            statuses.data_ptr(),
            stream,
        ),
        "CompressAsync",
    )


def decompress(
    in_ptrs: torch.Tensor,
    in_bytes: torch.Tensor,
    out_capacity: torch.Tensor,
    chunk_bytes: int,
    total_bytes: int,
    out_ptrs: torch.Tensor,
    stream: int,
) -> None:
    """Asynchronously decompress a batch of ANS chunks on `stream`. All tensor arguments live on the GPU."""
    n = in_ptrs.numel()
    device = in_ptrs.device
    temp_bytes = _size_t()
    _check(
        _decompress_temp(
            n, chunk_bytes, _DECOMPRESS_OPTS, ctypes.byref(temp_bytes), total_bytes
        ),
        "DecompressGetTempSize",
    )
    temp = torch.empty(max(temp_bytes.value, 1), dtype=torch.uint8, device=device)
    actual = torch.empty(n, dtype=torch.int64, device=device)
    statuses = torch.empty(n, dtype=torch.int32, device=device)
    _check(
        _decompress(
            in_ptrs.data_ptr(),
            in_bytes.data_ptr(),
            out_capacity.data_ptr(),
            actual.data_ptr(),
            n,
            temp.data_ptr(),
            temp_bytes.value,
            out_ptrs.data_ptr(),
            _DECOMPRESS_OPTS,
            statuses.data_ptr(),
            stream,
        ),
        "DecompressAsync",
    )
