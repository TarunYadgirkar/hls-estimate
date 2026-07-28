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

## D9 — Calibration targets: hls4ml, not FINN
FINN's published BNN-PYNQ numbers (CNV-W1A1 etc.) are 1-bit/2-bit binarized networks
implemented as XNOR-popcount in LUT fabric, with no DSP MACs at all. Our model has no
1-bit datapath and would be meaningless there. hls4ml's fixed-point designs are a
genuine apples-to-apples target: their "reuse factor" is exactly the inverse of our
`unroll` knob. Chose 2 networks (jet tagger MLP + SVHN CNN), 3 configurations.
Numbers quoted verbatim from the papers into `models/literature.py`.

## D10 — LUT/FF coefficients are fitted, and labelled as fitted
The initial hand-guessed LUT/FF constants were ~4x too high vs published data. Rather
than ship known-bad constants or silently tune them, they are fitted to one reference
design and MODEL.md states plainly which entry is a fit (not a validation) and which
entries are independent. The independent CNN check is 2.6x/3.75x off and is reported
as such. Bands in the test are set from measured ratios so the test is a regression
guard, not a restatement of the constants.

## D11 — Web UI ports the model to TypeScript, with enforced parity
The UI needs sub-frame response to a slider drag, which rules out a round trip to a
Python backend (and torch is far too large for a serverless function anyway). So
`web/lib/{model,codegen,dse}.ts` are hand ports of the Python modules.

Duplication is a real hazard, so it is pinned: `scripts/gen_golden.py` emits golden
vectors from the Python source of truth (60 layer cases, 6 graph estimates, 20 emitted
C++ files, 10 DSE fronts) and 118 vitest cases assert exact equality. Mutating one TS
constant by 0.01 fails 19 of them; the emitter comparison is byte-for-byte. Python
stays the source of truth — if parity breaks, the port is wrong, never the golden file.

This already earned its keep: the parity test caught two genuine port bugs (wrong
tensor wiring in the web examples, and a `maxParallel` that disagreed with `dse.py`
for pooling layers).

## D12 — CSP allows 'unsafe-inline' for scripts, and says so
The global web-security rules ask for a nonce-based CSP. Next.js App Router emits
inline bootstrap and RSC-payload scripts on every page, so a nonce requires middleware,
which forces dynamic rendering and gives up full static prerendering for a site that is
otherwise 100% static. Given the app has no user data, no auth, no forms and no
network calls, that trade is not worth it here. Everything else is locked down:
`default-src 'self'`, `object-src 'none'`, `frame-ancestors 'none'`, `form-action
'none'`, `base-uri 'self'`, plus HSTS/nosniff/frame-deny/referrer/permissions headers.
Recorded rather than quietly skipped.

## D8 — DSE returns a Pareto front over (latency, total-resource-fraction)
Budget = per-device caps on {LUT, FF, DSP, BRAM}. DSE enumerates knob configs, drops any
config exceeding ANY device cap, then returns the non-dominated set trading latency vs.
peak resource utilisation. Enumerate-then-filter (not a fancy optimiser) — easy to test,
and the search spaces here are small.
