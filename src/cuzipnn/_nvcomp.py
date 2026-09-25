"""Minimal ctypes bindings for nvCOMP's batched ANS (rANS entropy coder) C API."""

import ctypes
import importlib

import torch

NVCOMP_TYPE_CHAR = 0
NVCOMP_SUCCESS = 0

# From nvcomp/ans.h
MAX_CHUNK_BYTES = 1 << 24
DECOMPRESS_INPUT_ALIGNMENT = 8


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


_COMPRESS_OPTS = _CompressOpts(0, NVCOMP_TYPE_CHAR, 0)
_DECOMPRESS_OPTS = _DecompressOpts(0, NVCOMP_TYPE_CHAR, 0)

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


def max_compressed_chunk_bytes(chunk_bytes: int) -> int:
    out = _size_t()
    _check(_max_output(chunk_bytes, _COMPRESS_OPTS, ctypes.byref(out)), "GetMaxOutputChunkSize")
    return out.value


def _stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def compress(
    in_ptrs: torch.Tensor,
    in_bytes: torch.Tensor,
    chunk_bytes: int,
    total_bytes: int,
    out_ptrs: torch.Tensor,
    out_bytes: torch.Tensor,
    statuses: torch.Tensor,
) -> None:
    """Asynchronously ANS-compress a batch of chunks. All tensor arguments live on the GPU.

    in_ptrs/out_ptrs are int64 device addresses; in_bytes/out_bytes are int64 (size_t);
    statuses is int32 and receives one nvcompStatus_t per chunk.
    """
    n = in_ptrs.numel()
    temp_bytes = _size_t()
    _check(
        _compress_temp(n, chunk_bytes, _COMPRESS_OPTS, ctypes.byref(temp_bytes), total_bytes),
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
            _COMPRESS_OPTS,
            statuses.data_ptr(),
            _stream(in_ptrs.device),
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
) -> None:
    """Asynchronously decompress a batch of ANS chunks. All tensor arguments live on the GPU."""
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
            _stream(device),
        ),
        "DecompressAsync",
    )
