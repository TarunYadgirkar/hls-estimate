# hls-estimate

Predict FPGA resource usage (LUT, FF, DSP, BRAM) and latency for a small quantized
PyTorch model, and emit synthesizable Vitis HLS C++ — **without running synthesis**.

**Live tool: [hls-estimate.vercel.app](https://hls-estimate.vercel.app)**

Vitis HLS is *not* required. If it happens to be installed, `--validate` says so and
points at the emitted code; otherwise everything here is analytical.

The analytical model is the contribution, so its assumptions and its measured error
are written down in [MODEL.md](MODEL.md) — including the places it is 161% and 275%
wrong.

---

## Setup

Needs Python 3.9+ (3.11 used here) and a C++ compiler for the bit-exactness test.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

`torch` is pinned to 2.2.2 — the last release with macOS x86_64 wheels. On other
platforms any recent torch works; relax the pin in `pyproject.toml`.

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

## Usage

```bash
.venv/bin/python -m hls_estimate.cli list
```

Estimate a model against a device and get a fit verdict:

```bash
.venv/bin/python -m hls_estimate.cli estimate cnn_small --device zynq-7020
```

```text
target: zynq-7020
layer       kind            MACs    DSP   BRAM      LUT       FF   cycles    %dev
---------------------------------------------------------------------------------
conv0       conv2d        110592      1      6      232      175   110600   2.1%
pool0       maxpool2d          0      0      1      128      108     4104   0.4%
conv1       conv2d        294912      1      8      232      175   294920   2.9%
pool1       maxpool2d          0      0      1      128      108     2056   0.4%
fc          linear          5120      1      7      232      175     5128   2.5%
---------------------------------------------------------------------------------
TOTAL                     410624      3     23      952      741   294920

bottleneck (slowest stage): conv1
verdict: FITS on zynq-7020 (binding resource: BRAM at 8.2%)
```

That is at the default `unroll=1`. Raise the parallelism and the same network stops
fitting — 683 MAC lanes against 220 DSPs.

Add `--json` for machine-readable output, or `--validate` to check for a real Vitis
install.

Write synthesizable HLS C++ (`--tb` appends a test bench `main()`):

```bash
.venv/bin/python -m hls_estimate.cli emit conv_relu_pool -o net.cpp --tb
```

Explore the design space and get the Pareto front:

```bash
.venv/bin/python -m hls_estimate.cli dse cnn_small --device ultra96
```

### As a library

```python
from hls_estimate.frontend import from_torch
from hls_estimate.devices import get_device
from hls_estimate.report import build_report

graph = from_torch(my_module.eval(), input_shape=(1, 3, 32, 32), w_bits=8)
print(build_report(graph, get_device("zynq-7020")).render())
```

`from_torch` traces with `torch.fx`, folds batchnorm into the preceding conv/linear,
and quantizes weights per-tensor symmetrically.

## What is supported

conv2d, linear, relu, maxpool2d, add (residual), batchnorm folded into the preceding
layer, at int4/int8/int16 weights. Devices: `zynq-7020`, `ultra96`, `pynq-z2`.

Out of scope: grouped/depthwise conv, dilation, softmax, attention, RNNs, and
floating-point models. See [SPEC.md](SPEC.md).

## Tuning knobs

Every layer carries `Knobs(unroll, tile, pipeline)`:

- `unroll` — MAC lanes instantiated. Drives DSP up and latency down.
- `tile` — output tile depth. Drives BRAM.
- `pipeline` — II=1 when set, otherwise a sequential initiation interval.

Design space exploration sweeps these and returns only configurations that fit the
device budget.

## The web app

`web/` is a Next.js app that runs the same model in the browser for instant feedback.
The TypeScript port is pinned to the Python source of truth by 118 parity tests
against golden vectors — including byte-for-byte comparison of emitted C++.

```bash
cd web && npm install && npm run dev
npx vitest run          # parity against Python
.venv/bin/python scripts/gen_golden.py   # regenerate goldens after changing the model
```

## Project documents

| File | What is in it |
|---|---|
| [SPEC.md](SPEC.md) | Scope, interfaces, out-of-scope, acceptance criteria |
| [MODEL.md](MODEL.md) | Every modeling assumption and the measured error |
| [DECISIONS.md](DECISIONS.md) | Judgement calls and why |
| [DEPENDENCIES.md](DEPENDENCIES.md) | Everything pulled in from outside |
| [HANDOFF.md](HANDOFF.md) | What works, what is stubbed, what to do next |
| [PROGRESS.md](PROGRESS.md) | Running build log |

## Accuracy, stated plainly

Measured against published hls4ml results: DSP runs ~27% high on fully-parallel
designs and 113% high on a DSP-saturated CNN. LUT/FF coefficients were fitted to one
design and are 161%/275% high on a network they were not fitted to. BRAM and latency
have never been validated against a published number.

Use this to rank configurations and catch designs that are off by an order of
magnitude. Do not quote its LUT count in a paper. Full detail in [MODEL.md](MODEL.md).
