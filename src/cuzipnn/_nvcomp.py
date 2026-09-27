"""Minimal ctypes bindings for nvCOMP's batched ANS (rANS entropy coder) and CRC32 C APIs."""

import ctypes
import importlib
from typing import Any

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


# From nvcomp/crc32.h
class _CRC32Spec(ctypes.Structure):
    _fields_ = [
        ("poly", ctypes.c_uint32),
        ("init", ctypes.c_uint32),
        ("ref_in", ctypes.c_bool),
        ("ref_out", ctypes.c_bool),
        ("xorout", ctypes.c_uint32),
        ("reserved", ctypes.c_char * 16),
    ]


class _CRC32KernelConf(ctypes.Structure):
    _fields_ = [
        ("kernel_kind", ctypes.c_int),
        ("bytes_per_read", ctypes.c_int32),
        ("blocks_per_msg", ctypes.c_int32),
        ("reserved", ctypes.c_char * 20),
    ]


class _CRC32Opts(ctypes.Structure):
    _fields_ = [
        ("spec", _CRC32Spec),
        ("kernel_conf", _CRC32KernelConf),
        ("reserved", ctypes.c_char * 64),
    ]


# nvcompCRC32: standard CRC-32 (PKZIP), the same as zlib.crc32.
_CRC32 = _CRC32Spec(0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF)
_CRC32_ONLY_SEGMENT = 0


# Not `import nvidia.libnvcomp`: importing nvidia.nvcomp first deletes that attribute from the namespace package.
_lib = importlib.import_module("nvidia.libnvcomp").load_library()
if _lib is None:
    raise ImportError("could not load libnvcomp.so.5")

_size_t = ctypes.c_size_t
_ptr = ctypes.c_void_p
_size_p = ctypes.POINTER(ctypes.c_size_t)


def _bind(name: str, *argtypes: type) -> Any:
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
_crc32_conf = _bind(
    "nvcompBatchedCRC32GetHeuristicConf",
    _ptr,
    _size_t,
    ctypes.POINTER(_CRC32KernelConf),
    _size_t,
    _ptr,
)
_crc32 = _bind(
    "nvcompBatchedCRC32Async",
    _ptr,
    _ptr,
    _size_t,
    _ptr,
    _CRC32Opts,
    ctypes.c_int,
    _ptr,
    _ptr,
)


def _check(status: int, what: str) -> None:
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
    stream: torch.cuda.Stream,
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
    stream: torch.cuda.Stream,
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


def crc32(
    in_ptrs: torch.Tensor,
    in_bytes: torch.Tensor,
    max_bytes: int,
    out: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    """Asynchronously compute the CRC-32 of a batch of chunks on `stream`. All tensor arguments live on the GPU.

    in_ptrs are int64 device addresses; in_bytes are int64 (size_t) chunk sizes, the largest of
    which is `max_bytes`; out is uint32 and receives one checksum per chunk.
    """
    n = in_ptrs.numel()
    conf = _CRC32KernelConf()
    # Given max_bytes, this is a host-only lookup that does not synchronize with the device.
    _check(_crc32_conf(None, n, ctypes.byref(conf), max_bytes, None), "CRC32GetHeuristicConf")
    _check(
        _crc32(
            in_ptrs.data_ptr(),
            in_bytes.data_ptr(),
            n,
            out.data_ptr(),
            _CRC32Opts(_CRC32, conf),
            _CRC32_ONLY_SEGMENT,
            None,
            stream,
        ),
        "CRC32Async",
    )
