# Handoff

Live: **https://hls-estimate.vercel.app** · Repo: **https://github.com/TarunYadgirkar/hls-estimate**

## Test status

```
.venv/bin/python -m pytest -q     82 passed
cd web && npx vitest run          118 passed
```

No test was weakened, skipped, xfailed or deleted. Two tests were mutation-checked to
prove they are not vacuous:

- Removing the round-half-up term from the C++ emitter fails 5 of 6 bit-exactness cases.
- Changing one TypeScript constant by 0.01 fails 19 of the parity cases.

## What works

**All six acceptance criteria from the brief are met and green.**

| Acceptance test | Status |
|---|---|
| Analytical sanity — DSP matches the closed form exactly | pass |
| Monotonicity invariants over randomized configs (hypothesis) | pass |
| Bit-width scaling follows the documented packing model | pass |
| **Emitted C++ compiles as plain C++ and is bit-exact vs PyTorch** | pass |
| DSE never returns a config exceeding the device budget | pass |
| Calibration against ≥2 published designs, honest error bands | pass |

Beyond the acceptance list:

- **Frontend** — `torch.fx` ingest of a real `nn.Module`, with batchnorm folded into
  the preceding conv/linear and per-tensor symmetric weight quantization. The
  fx-derived graphs are covered by the bit-exactness test too.
- **IR** — conv2d, linear, relu, maxpool2d, add, with per-layer `Knobs(unroll, tile,
  pipeline)`.
- **Resource model** — DSP, BRAM, LUT, FF, latency as closed forms; every assumption
  documented in MODEL.md with its measured error.
- **Codegen** — Vitis HLS C++ with PIPELINE / UNROLL / ARRAY_PARTITION / DATAFLOW,
  plus an optional test bench. Compiles under `clang++ -std=c++17`.
- **DSE** — enumerate, filter to the device budget, return the Pareto front. Search
  truncation is now *disclosed* (warning in Python, banner in the UI) rather than
  silently capped.
- **Report** — per-layer attribution, bottleneck, fit verdict, binding resource.
- **CLI** — `list`, `estimate` (+`--json`, `--validate`), `emit` (+`--tb`), `dse`.
- **Web app** — interactive estimator with a live device floorplan, per-layer table,
  live-updating generated C++, and a clickable Pareto front. Deployed, static,
  security headers set.

## What is stubbed or missing

1. **ONNX frontend — not implemented.** SPEC listed it as the secondary path
   (DECISIONS D6 said it would be reported honestly if it slipped). `onnx` is
   installed but there is no `frontend/onnx.py`. Only `from_torch` exists.
2. **Vitis validation is a probe, not a comparison.** `--validate` reports whether
   `vitis_hls` is on PATH and points at the emitted code. It does not run `csynth`
   and diff the report against the estimate. Vitis was not available on this machine,
   so an automated comparison could not be written *or tested* — writing an untestable
   code path would have been worse than saying this.
3. **`tile` knob is modelled but not swept.** `bram()` honours it and the monotonicity
   test covers it, but `default_knob_grid` only varies `unroll`, so DSE never explores
   tiling.
4. **BRAM and latency are unvalidated** against any published number. See MODEL.md.
5. **Grouped/depthwise conv, dilation** raise `UnsupportedOp` by design.

## The three highest-value next steps

### 1. Model DSP saturation and LUT spillover
The single largest source of error. Real tools push multiplications into LUT fabric
when DSPs run out, and prefer LUTs for multiplies narrower than ~10 bits. We always
charge a DSP, which is why the SVHN CNN is +113% on DSP while its LUT is
correspondingly mispredicted. A first cut: cap DSP at the device budget and convert
the overflow to LUTs at a fitted cost per MAC. This would turn the worst calibration
row into one of the better ones, and it changes the *fit verdict*, which is the
tool's main output.

### 2. Charge logic per kernel position, not per MAC lane, for streamed convolutions
Explains most of the +161% LUT / +275% FF error on the CNN. hls4ml streams a
convolution spatially and reuses one datapath across pixels; our model bills every
lane as independent hardware. Changing the conv LUT/FF term from `f(unroll)` to
`f(out_ch · in_ch · kh · kw / reuse)` matches how these designs are actually built.
Both fixes are testable immediately against the calibration entries already in
`models/literature.py`.

### 3. Validate BRAM against something — anything
It is the only resource with zero empirical grounding, and it is the *binding*
resource on **all 6** bundled examples at default knobs — so the headline verdict this
tool prints is currently decided by its least-trustworthy number. Recommended path: build one small design in Vitis HLS on a machine that has
it, read the real BRAM count out of the synthesis report, and add it to
`models/literature.py` as a first data point. Until then the fit verdict can be driven
by a number nobody has checked.

## Repo map

```
hls_estimate/
  ir.py            IR nodes + Knobs
  model.py         the analytical model (MODEL.md documents it)
  devices.py       device budgets
  dse.py           design space exploration
  report.py        attribution, bottleneck, fit verdict
  cli.py           command line interface
  codegen/         HLS C++ emitters
  frontend/        torch.fx ingest, quantization, batchnorm folding
  models/          bundled examples, golden torch executor, literature data
tests/             the acceptance suite
scripts/           gen_golden.py — emits parity vectors for the web port
web/               Next.js UI (lib/*.ts are parity-pinned ports)
```
