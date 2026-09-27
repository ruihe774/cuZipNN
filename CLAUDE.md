# CLAUDE.md

GPU ZipNN tensor codec. Public API is `compress_tensor` / `decompress_tensor` in `src/cuzipnn/_codec.py`.

- `_codec.py`: blob format (documented in module docstring) and orchestration
- `_kernels.py`: Triton byte-split/merge and chunk-copy kernels
- `_nvcomp.py`: minimal ctypes bindings

Commands (use `uv run`):
- `pytest` — tests need a CUDA GPU
- `ruff check && ruff format`
- `pyright`

Bump `_VERSION` in `_codec.py` when the blob layout changes, and update its docstring.
