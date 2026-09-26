# cuZipNN

GPU implementation of [ZipNN](https://github.com/zipnn/zipnn) lossless tensor compression: bit reordering and byte grouping in Triton, entropy coding with nvCOMP ANS.

```python
from cuzipnn import compress_tensor, decompress_tensor

blob = compress_tensor(t)  # returns a 1-D uint8 CPU tensor
out = decompress_tensor(blob, device="cuda")
```

Supports float (bf16/fp16/fp32/fp64/fp8), integer, bool, and complex64 dtypes.

Requires Python 3.13+, a CUDA 13 GPU, and PyTorch 2.14. Install with `uv sync`.
