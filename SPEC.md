# hls-estimate — SPEC

Predict FPGA resource usage (LUT, FF, DSP, BRAM) and latency for a small quantized
PyTorch model, and emit synthesizable Vitis HLS C++, **without running synthesis**.

The analytical resource model is the contribution. Every assumption is documented in
`MODEL.md`; this file defines the scope, the interfaces, and the acceptance criteria.

---

## 1. Scope

### In scope
- Ingest a quantized model as a `torch.fx` graph (primary) or ONNX (secondary).
- Supported ops: `conv2d`, `linear`, `relu`, `maxpool2d`, `add` (residual),
  batchnorm **folded** into the preceding conv/linear, quantized int8 and int4 variants.
- Lower to an internal **dataflow IR** with explicit, tunable knobs:
  tiling factors and dataflow/parallelism (unroll) factors per layer.
- Emit synthesizable HLS C++ for Vitis HLS with pragmas: `PIPELINE`, `UNROLL`,
  `ARRAY_PARTITION`, `DATAFLOW`.
- Analytical resource model: DSP, BRAM, LUT, FF, and latency (cycles), per layer and
  aggregated, as closed-form functions of shapes, bit widths, and knobs.
- Design space exploration (DSE): given a device budget, search tiling/unroll configs
  and return the Pareto front of latency vs. resource usage.
- Report: per-layer resource attribution, bottleneck identification, and a fit/no-fit
  verdict against the target device.

### Out of scope
- Training, quantization-aware training, or automatic calibration of scales.
- Non-folded batchnorm as a standalone runtime op (must be folded upstream).
- Ops beyond the list above (softmax, attention, RNNs, transposed conv, grouped/depthwise
  conv with groups≠1, dilation≠1). Depthwise/grouped conv is a documented HANDOFF item.
- Cycle-accurate simulation. Latency is an analytical cycle estimate, not a simulation.
- Running Vitis HLS to produce ground-truth numbers (optional `--validate` path only).
- Floating-point models. Everything is integer-quantized.

---

## 2. Quantization contract (numeric ground truth)

We own the scheme so bit-exactness is well defined (see DECISIONS D4).

- Weights: signed int, `w_bits ∈ {4,8}`, per-tensor symmetric, stored as integers in
  `[-2^(w_bits-1), 2^(w_bits-1)-1]`.
- Activations: signed int, `a_bits` (default 8), range `[qmin, qmax]`.
- Conv2d / Linear: `acc = Σ (w_int · x_int) + bias_int`, accumulated in int64.
- Requantize (per layer, integer fixed-point, round-half-up):
  `y = clamp((acc · mult + (1 << (shift-1))) >> shift, out_qmin, out_qmax)`.
- ReLU: lower-clamp to 0 (may be fused into requant's `out_qmin`).
- MaxPool2d: integer max over the window.
- Add: integer add of two same-scale tensors, then clamp to `[qmin, qmax]`.

The golden model is real `nn.Module`s implementing exactly this in int64; the emitted
C++ mirrors it. Equality is by construction, and the test proves it on random inputs.

---

## 3. Internal IR

`hls_estimate.ir`

- `TensorSpec(shape: tuple[int,...], bits: int, signed: bool)` — logical tensor.
- `Knobs(unroll: int = 1, tile: int = 0, pipeline: bool = True)` — per-layer tuning.
  - `unroll` = MAC parallelism factor (how many multiply units instantiated).
  - `tile` = output-channel (conv) / output-feature (linear) tile depth; `0` = whole layer.
  - `pipeline` = whether the layer loop is pipelined (II=1) vs. sequential.
- Node types (each carries shapes, quant bits, `Knobs`, and requant `(mult, shift)`):
  - `Conv2d(in_ch, out_ch, kh, kw, stride, pad, in_hw, w_bits, a_bits, ...)`
  - `Linear(in_features, out_features, w_bits, a_bits, ...)`
  - `ReLU(...)`, `MaxPool2d(kh, kw, stride, ...)`, `Add(...)`
- `Graph(nodes: list[Node], input_spec, output_spec)` — ordered dataflow.

Invariants the IR guarantees (asserted in code): `unroll ≥ 1`, `unroll` divides the total
MAC-parallel dimension it applies to (or is clamped with a recorded warning),
`tile ≥ 0`, bits ∈ {2,4,8,16}.

---

## 4. Resource & latency model interface

`hls_estimate.model`

Pure functions, no side effects. For a node `n` and its `Knobs k`:

- `dsp(n, k) -> int`
- `bram(n, k) -> int`      (# of 18Kb BRAM primitives)
- `lut(n, k) -> int`
- `ff(n, k) -> int`
- `latency_cycles(n, k) -> int`
- `estimate_graph(graph) -> GraphEstimate` — per-layer table + totals + bottleneck.

Closed forms (full derivation in MODEL.md):

- **MACs** of a layer: conv `= out_ch·out_h·out_w·in_ch·kh·kw`; linear `= out_f·in_f`.
- **DSP** `= ceil(unroll / macs_per_dsp(w_bits))`, with
  `macs_per_dsp = {16:1, 8:1, 4:2, 2:4}` (see D5). Non-MAC ops use 0 DSP.
- **Latency** `≈ ceil(MACs / unroll) · II + depth`, II=1 if pipelined else = inner-loop
  trip count; `depth` = small constant pipeline fill. Strictly decreasing in `unroll`.
- **BRAM**: sum over the layer's buffers (weights, input/line buffer, output tile) of
  `bram18(depth, width_bits) = ceil(width_bits/18) · ceil(depth/1024)`, multiplied by the
  array-partition factor implied by `unroll`. Non-decreasing in `tile` and in `unroll`.
- **LUT / FF**: `base_control + per_lane · unroll · f(bits)` (+ requant logic). Coarse,
  linear, non-decreasing in `unroll`. Honest error bars in MODEL.md.

### Required monotonicity (asserted by tests)
1. `unroll → 2·unroll` ⇒ DSP does not decrease.
2. `tile` increases ⇒ BRAM does not decrease.
3. `unroll` increases ⇒ latency strictly decreases (until fully unrolled).

---

## 5. Codegen interface

`hls_estimate.codegen`

- `emit_graph(graph, name) -> str` — one self-contained `.cpp/.hpp` string of synthesizable
  Vitis HLS C++: a top-level `DATAFLOW` function calling per-layer functions, each with
  `PIPELINE`/`UNROLL`/`ARRAY_PARTITION` pragmas derived from that layer's `Knobs`.
- Emitted code is **plain-C++ compilable**: pragmas are `#pragma HLS ...` (ignored by
  clang/g++), types are `int8_t/int32_t/int64_t`, no Vitis-only headers on the compile
  path (an `#ifdef __VITIS_HLS__` guards `ap_int` / `hls_stream` niceties; the default
  build uses plain integer types).
- `emit_testbench(graph, name) -> str` — a `main()` that reads inputs, runs the network,
  writes outputs; used by the bit-exact test to compare against PyTorch.

### Correctness requirement (the test that matters)
Emitted C++ compiled with `clang++`/`g++` (pragmas ignored) must produce **bit-exact**
output vs. the golden PyTorch model on random inputs, for every supported op.

---

## 6. Device budgets

`hls_estimate.devices` — dict of `DeviceBudget(name, lut, ff, dsp, bram18)`:

- `zynq-7020` (XC7Z020): LUT 53200, FF 106400, DSP 220, BRAM18 280.
- `ultra96` (Zynq UltraScale+ ZU3EG): LUT 70560, FF 141120, DSP 360, BRAM18 432.
- `pynq-z2` == zynq-7020.

(Numbers are the published fabric totals; sources listed in the module + MODEL.md.)

---

## 7. DSE interface

`hls_estimate.dse`

- `explore(graph, device, knob_grid=None) -> list[DesignPoint]` where
  `DesignPoint(config, latency, lut, ff, dsp, bram, fits: bool)`.
- Enumerate per-layer knob combos from `knob_grid` (default: powers-of-two unroll up to
  the layer's MAC-parallel dim; a couple of tile choices).
- **Budget rule (asserted):** every returned point satisfies all four caps of `device`.
  Over-budget configs are dropped, never returned.
- Return the **Pareto front**: non-dominated points over (latency ↓, peak-utilisation ↓).

---

## 8. Report

`hls_estimate.report`

- Per-layer table: MACs, DSP, BRAM, LUT, FF, latency, % of device.
- Totals + **bottleneck** = the layer with the max `max(resource_fraction)`.
- Fit verdict: does the whole design fit the target device? which resource is binding?

---

## 9. CLI

`hls-estimate` (console script):
- `estimate MODEL --device zynq-7020` → report + fit verdict.
- `emit MODEL -o out.cpp` → write HLS C++ (+ testbench with `--tb`).
- `dse MODEL --device ultra96` → Pareto front table.
- `--validate` (optional) → run `vitis_hls` if installed and diff estimates vs. real.

`MODEL` is a Python entry `module:factory` returning a built `Graph` (or a `.onnx` path).
Bundled example models live in `hls_estimate.models` for tests and demos.

---

## 10. Acceptance criteria (tests written first, must fail before implementation)

1. **Analytical sanity** — hand-computed conv layer: predicted DSP equals the closed-form
   `ceil(MACs_parallel / macs_per_dsp)` exactly.
2. **Monotonicity invariants** (randomized, hypothesis): double unroll ⇒ DSP not lower;
   bigger tile ⇒ BRAM not lower; more parallelism ⇒ lower latency.
3. **Bit-width scaling** — int4 vs int8 DSP follows the documented packing model
   (`DSP(int4) == DSP(int8) with macs_per_dsp doubled`).
4. **Bit-exact codegen** — emitted C++ compiles as plain C++ and matches the PyTorch model
   exactly on random inputs, across all supported ops.
5. **Budget enforcement** — DSE never returns a config exceeding the device budget.
6. **Known-model calibration** — for ≥2 small CNNs/MLPs from FINN/hls4ml literature,
   estimates land within the error band stated in MODEL.md (band recorded honestly).

A test is ground truth. It is never weakened, skipped, xfailed, or deleted to pass.
