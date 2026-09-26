"""Minimal ctypes bindings for the CUDA runtime calls that PyTorch does not expose."""

import ctypes
from typing import Any

import torch
from cuda.pathfinder import load_nvidia_dynamic_lib

# Reuses the copy torch already loaded, else searches NVIDIA wheels, conda, and CUDA_HOME.
_lib = ctypes.CDLL(load_nvidia_dynamic_lib("cudart").abs_path)

_SRC_ACCESS_ORDER_STREAM = 0x1


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_int),
        ("srcLocHint", _MemLocation),
        ("dstLocHint", _MemLocation),
        ("flags", ctypes.c_uint),
    ]


_ATTRS = _MemcpyAttributes(_SRC_ACCESS_ORDER_STREAM)
_ATTRS_IDX = (ctypes.c_size_t * 1)(0)

_size_t = ctypes.c_size_t
_ptr = ctypes.c_void_p


def _bind(name: str, *argtypes: type) -> Any:
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


_memcpy_batch = _bind(
    "cudaMemcpyBatchAsync", _ptr, _ptr, _ptr, _size_t, _ptr, _ptr, _size_t, _ptr
)


def _check(err: int, what: str) -> None:
    if err:
        raise RuntimeError(f"{what} failed with cudaError {err}")


def memcpy_batch(
    dsts: torch.Tensor, srcs: torch.Tensor, sizes: torch.Tensor, stream: torch.cuda.Stream
) -> None:
    """Asynchronously copy sizes[i] bytes from srcs[i] to dsts[i]; all three are CPU int64 tensors."""
    for t in (dsts, srcs, sizes):
        assert t.device.type == "cpu" and t.dtype == torch.int64 and t.is_contiguous()
    _check(
        _memcpy_batch(
            dsts.data_ptr(),
            srcs.data_ptr(),
            sizes.data_ptr(),
            dsts.numel(),
            ctypes.byref(_ATTRS),
            _ATTRS_IDX,
            1,
            stream,
        ),
        "cudaMemcpyBatchAsync",
    )
