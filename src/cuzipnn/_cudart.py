"""Minimal ctypes bindings for the CUDA runtime calls that PyTorch does not expose."""

import ctypes

import torch

# torch has already loaded libcudart, so dlopen resolves the soname to that copy.
_lib = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")

# cudaMemcpyBatchAsync rejects the legacy NULL stream; this blocking stream is ordered with it.
STREAM_PER_THREAD = 0x2
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


def _bind(name, *argtypes):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


_memcpy_batch = _bind(
    "cudaMemcpyBatchAsync", _ptr, _ptr, _ptr, _size_t, _ptr, _ptr, _size_t, _ptr
)
_stream_sync = _bind("cudaStreamSynchronize", _ptr)


def _check(err, what):
    if err:
        raise RuntimeError(f"{what} failed with cudaError {err}")


def stream_for(device: torch.device) -> int:
    """The current stream's handle, or the per-thread default stream in place of the NULL stream."""
    return torch.cuda.current_stream(device).cuda_stream or STREAM_PER_THREAD


def memcpy_batch(
    dsts: torch.Tensor, srcs: torch.Tensor, sizes: torch.Tensor, stream: int
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


def stream_synchronize(stream: int) -> None:
    _check(_stream_sync(stream), "cudaStreamSynchronize")
