# Decisions Log

Autonomous decisions taken when the spec was ambiguous. Bias: pick the option that
is easier to test, write down why, keep moving.

## D1 — Project location
`~/TarunsCode/hls-estimate`, per the user's documented disk-layout convention
(per-project folder under `~/TarunsCode`). Git repo initialised here.

## D2 — Python 3.11 via miniconda, not system 3.8
System Python is 3.8.1 (EOL; no modern torch wheels). Miniconda ships 3.11.8.
Built an isolated `.venv` from it. Easier to get a working torch than to fight 3.8.

## D3 — torch 2.2.2 (CPU, x86_64)
Host is `x86_64`. torch stopped shipping macOS-Intel wheels after 2.2.2, so 2.2.2
is the newest usable version. CPU-only is fine — we never train, only trace and run
small reference inferences.

## D4 — Quantization semantics we commit to (drives bit-exactness)
The one test that matters is "emitted C++ is bit-exact vs the PyTorch model". To make
that well-defined we OWN the quantization scheme rather than depending on torch's
(fragile, backend-specific) quantized kernels:

- Weights: signed integer, `w_bits` ∈ {4,8}, per-tensor symmetric. Stored as ints.
- Activations: signed integer, `a_bits` (default 8).
- Conv2d/Linear: integer MAC into int32 accumulator, `acc = Σ w_int * x_int (+ bias_int)`.
- Requantize: `y = clamp((acc * mult + (1<<(shift-1))) >> shift, qmin, qmax)` with
  per-layer integer `(mult, shift)` (gemmlowp / TFLite-style fixed-point). Round-half-up.
- ReLU: `max(0, x)` on ints (fused as a clamp lower-bound during requant when present).
- MaxPool: integer max over window.
- Add (residual): integer add of two same-scale tensors, then clamp.
- BatchNorm: expected already folded into conv weights/bias upstream. We expose a
  `fold_batchnorm` helper but the IR has no standalone BN node. Documented in SPEC.

The golden PyTorch model is built from custom `nn.Module`s (`QuantConv2d`, `QuantLinear`,
...) whose `forward` implements exactly the integer math above using int64 tensors.
The emitted C++ mirrors it with `int64_t` accumulation. Same rounding, same clamps →
bit-exact by construction. This is legitimate: it is a real torch model, traceable by
`torch.fx`, and its numeric contract is explicit.

## D5 — DSP packing model (drives bit-width test)
`macs_per_dsp(bits)`: 16→1, 8→1, 4→2, 2→4. i.e. one DSP does one 8-bit MAC, or two
packed 4-bit MACs, along the shared-operand (parallelism) axis. So for equal parallelism
`DSP(int4) = DSP(int8) / 2`. Rationale + caveats in MODEL.md. An `int8_dsp_packing=2`
option (Xilinx dual-int8-on-one-DSP, WP486) is available but OFF by default so the
baseline is the simplest defensible model.

## D6 — Frontend priority: torch.fx first, ONNX second
The bit-exact contract is defined against a torch model, so `torch.fx` ingest is the
primary path and is what the correctness test exercises. ONNX ingest is implemented for
the documented op set but is a secondary convenience; if it slips it goes to HANDOFF as
"stubbed", never faked.

## D7 — Calibration data lives in-repo, not fetched at test time
Tests must be deterministic and offline. Published literature numbers are captured once
into `references/literature.json` (with citations); the calibration test reads that file.
The error band is stated honestly in MODEL.md and asserted in the test.

## D8 — DSE returns a Pareto front over (latency, total-resource-fraction)
Budget = per-device caps on {LUT, FF, DSP, BRAM}. DSE enumerates knob configs, drops any
config exceeding ANY device cap, then returns the non-dominated set trading latency vs.
peak resource utilisation. Enumerate-then-filter (not a fancy optimiser) — easy to test,
and the search spaces here are small.
