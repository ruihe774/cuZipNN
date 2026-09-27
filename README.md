# cuZipNN

GPU implementation of [ZipNN](https://github.com/zipnn/zipnn) lossless tensor compression: bit reordering and byte grouping in Triton, entropy coding with [nvCOMP](https://developer.nvidia.com/nvcomp) ANS.

## Installation

```sh
pip install cuzipnn
```

Requires Linux (for Triton), Python 3.12+, an NVIDIA GPU, and a CUDA 13 build of PyTorch 2.9+.

## Usage

```python
import torch
from cuzipnn import compress_tensor, decompress_tensor

t = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
blob = compress_tensor(t)  # 1-D uint8 tensor in CPU memory
out = decompress_tensor(blob, device="cuda")
assert torch.equal(out, t)
```

Supported dtypes: `bool`, `uint8`/`16`/`32`/`64`, `int8`/`16`/`32`/`64`, `float16`, `bfloat16`, `float32`, `float64`, `float8_e4m3fn`, `float8_e5m2`, and `complex64`. The blob records dtype and shape; a non-contiguous input is compressed in row-major order.

## Performance

Whole-model round trips against [ZipNN](https://github.com/zipnn/zipnn) 0.5.4, both with default settings. Ratio is compressed / original size (lower is better). Throughput is original bytes per second (GB = 10⁹ bytes).

| Model | dtype | Size | Ratio<br>ZipNN / cuZipNN | Compress GB/s<br>ZipNN / cuZipNN | Decompress GB/s<br>ZipNN / cuZipNN |
|---|---|---:|---:|---:|---:|
| [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) | BF16 | 15.2 GB | 66.7% / 67.2% | 7.7 / **15.1** | 21.5 / **36.0** |
| [flan-t5-xl](https://huggingface.co/google/flan-t5-xl) | FP32 | 11.4 GB | 83.2% / 83.5% | 6.2 / **12.5** | 21.2 / **29.5** |
| [gpt2-xl](https://huggingface.co/openai-community/gpt2-xl) | FP32 | 6.4 GB | 80.7% / 81.0% | 6.2 / **11.6** | 20.5 / **27.8** |
| [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | FP16 | 3.1 GB | 85.0% / 85.6% | 3.9 / **8.4** | 7.1 / **15.9** |
| [Qwen3-4B-FP8](https://huggingface.co/Qwen/Qwen3-4B-FP8) | FP8 + BF16 | 5.2 GB | 78.1% / 79.7% | 6.0 / **12.2** | 11.8 / **27.2** |

Measured on an NVIDIA GB10 (DGX Spark: 20-core Grace CPU with memory shared with the GPU), one tensor at a time, best of two passes. ZipNN goes from a CPU tensor to bytes and back, using 16 threads. cuZipNN goes from a GPU tensor to a CPU blob and back to the GPU, so its times include both transfers. On a discrete GPU, those transfers go over PCIe instead.

## API

### `compress_tensor(tensor, *, chunk_bytes=131072, sub_chunk_bytes=8192, passthrough_threshold=0.95, min_compress_bytes=65536, pin_memory=False) -> Tensor`

Compresses `tensor` (CPU or CUDA) on the GPU and returns the blob as a 1-D `uint8` CPU tensor.

- `chunk_bytes`: elements per independently coded chunk; a multiple of 8, at most 16 MiB.
- `sub_chunk_bytes`: target size of nvCOMP's ANS sub-chunks. Larger compresses slightly better; smaller decodes with more parallelism.
- `passthrough_threshold`: a chunk stays compressed only if it shrinks below this fraction of its raw size; otherwise it is stored raw. In `(0, 1]`.
- `min_compress_bytes`: tensors smaller than this are stored uncompressed.
- `pin_memory`: return the blob in pinned memory.

### `decompress_tensor(blob, *, device=None) -> Tensor`

Decompresses a blob (on CPU or GPU) back into a tensor on `device`. The default is the blob's device if it is on a GPU, else the current CUDA device. Decoding always runs on the GPU; `device="cpu"` gets a copy of the result.

Passing a blob that was not produced by `compress_tensor` is undefined behavior. Verify untrusted blobs with the CRC-32 functions below first.

### `compress_tensor_with_crc32(tensor, *, ..., checksum_chunk_bytes=16384) -> (Tensor, Tensor)`

Same as `compress_tensor` (same arguments, same blob), and also returns the CRC-32 of every `checksum_chunk_bytes`-byte chunk of the blob (the last one may be shorter) as a 1-D `uint32` CPU tensor. The checksums are computed on the GPU and match `zlib.crc32`.

### `hash_tensor_with_crc32(tensor, *, checksum_chunk_bytes=16384) -> Tensor`

CRC-32 of each `checksum_chunk_bytes`-byte chunk of any tensor's bytes, computed on the GPU. Applied to a blob, it reproduces the checksums from `compress_tensor_with_crc32`:

```python
blob, crc = compress_tensor_with_crc32(t)
...
if not torch.equal(hash_tensor_with_crc32(blob), crc):
    raise ValueError("corrupted blob")
out = decompress_tensor(blob)
```

## Blob format

The layout is documented in [`_codec.py`](src/cuzipnn/_codec.py). Blobs carry a format version, and `decompress_tensor` rejects blobs from other versions. The format is not compatible with the original ZipNN.

## License

[Unlicense](LICENSE)
