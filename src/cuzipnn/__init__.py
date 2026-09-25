"""GPU implementation of ZipNN lossless tensor compression, with nvCOMP ANS as the entropy coder."""

from ._codec import compress_tensor, decompress_tensor

__all__ = ["compress_tensor", "decompress_tensor"]
