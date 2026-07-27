# External Dependencies

Everything pulled into this project from outside, with what it is and why.

## Runtime / toolchain

| Thing | Version | What it is | Why |
|-------|---------|-----------|-----|
| Python | 3.11.8 (from `/opt/miniconda3`) | Language runtime | System default was 3.8.1 (EOL, no modern torch wheels). Used miniconda's 3.11 to build a venv. |
| numpy | <2.0 (1.26.4) | Numerics | Array math for reference inference + resource math. Pinned <2 for torch 2.2.2 ABI compatibility. |
| torch | 2.2.2 (CPU) | PyTorch | Ingest models via `torch.fx`; define the quantized golden model for the bit-exact test. 2.2.2 is the last release with macOS x86_64 wheels (this host is Intel/x86_64). |
| onnx | latest | ONNX model format | Optional second frontend (ONNX ingest). |
| pytest | 9.1.1 | Test runner | Acceptance test suite. |
| hypothesis | 6.161.7 | Property-based testing | Randomized monotonicity / invariant tests. |
| pypdf | 6.14.2 | PDF text extraction | One-off: pulled the published resource tables out of the hls4ml papers to build `models/literature.py`. Not needed at runtime or test time. |
| clang++ (Apple clang 17, via `g++`/`clang++`) | 17.0.0 | C++ compiler | Compile emitted HLS C++ as plain C++ (pragmas ignored) for the bit-exact correctness test. |

## Not installed / not required

- **Vitis HLS** — not present on this host (`vitis_hls not found`). Per the contract it is NOT required. An optional `--validate` path shells out to `vitis_hls` only if it exists; all Vitis-dependent tests are skipped otherwise.

## Literature data (for calibration test)

Published FPGA resource numbers for small CNNs/MLPs are stored in `references/literature.json` with source citations. See `MODEL.md` for the error bands and honest discussion. Sources are recorded inline in that JSON.
