"""GPU implementation of ZipNN lossless tensor compression, with nvCOMP ANS as the entropy coder."""

from ._codec import (
    compress_tensor,
    compress_tensor_with_crc32,
    decompress_tensor,
    hash_tensor_with_crc32,
)

__all__ = [
    "compress_tensor",
    "compress_tensor_with_crc32",
    "decompress_tensor",
    "hash_tensor_with_crc32",
]
