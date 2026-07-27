# The Analytical Resource Model

This is the contribution. The code is a vehicle for these assumptions; each one is
stated below with its justification, its failure mode, and — where published data
exists — the measured error.

**Summary of honesty:** the DSP model is good in the regime it targets (≈ +27%
systematic overestimate on fully-parallel, non-saturated designs). The LUT/FF model
is a fitted two-parameter guess and is **+160% to +275% wrong** on a CNN outside the
fitting regime. The BRAM model is **not validated against any published number at
all**. Details and numbers below.

---

## 1. Quantization and numeric semantics

The estimator is defined against an explicit integer scheme (SPEC §2):
`acc = Σ w_int·x_int + bias`, then `y = clamp((acc·mult + 2^(shift-1)) >> shift)`.

**Assumption 1.1** — accumulation is exact (int64 in the golden model, `acc_t` in the
emitted C++). Real HLS designs size the accumulator to
`w_bits + a_bits + ceil(log2(MACs))`. We do not model accumulator-width savings, so
we slightly overestimate FF for layers with few MACs.

**Assumption 1.2** — `>>` is an arithmetic (floor) shift, matching
`torch.div(..., rounding_mode='floor')`. This is what shift-based requantization
synthesizes to. Verified by the bit-exactness test, and by a mutation check: removing
the round-half-up term makes 5 of 6 bit-exact tests fail.

---

## 2. DSP model

```
DSP(layer) = ceil(unroll / macs_per_dsp(w_bits))
macs_per_dsp = {16: 1, 8: 1, 4: 2, 2: 4}
```

**Assumption 2.1 — one DSP per MAC lane.** `unroll` is the number of physical
multiply lanes instantiated. Non-MAC ops (ReLU, maxpool, add) use zero DSPs.

**Assumption 2.2 — narrow operands pack along the parallelism axis.** A DSP48 has a
25×18 multiplier. Two int4 operands (or four int2) fit in one multiplier port with
guard bits between them, so one DSP retires 2 (or 4) MACs that share an operand.
This is the standard SIMD-in-DSP trick. Consequence: **int4 uses exactly half the DSP
of int8 at equal parallelism**, which is what the bit-width acceptance test asserts.

**Assumption 2.3 — int8 gets one MAC per DSP, not two.** Xilinx WP486 shows two int8
MACs on one DSP48E2 using the pre-adder. We deliberately do *not* claim that by
default: it needs a shared operand and DSP48E2 (UltraScale+), and is not available on
the DSP48E1 in Zynq-7020. `MACS_PER_DSP[8] = 1` is the conservative choice. Setting
`MACS_PER_DSP[8] = 2` is a one-line change if you target UltraScale+ and know your
dataflow shares an operand.

**Assumption 2.4 — no DSP saturation, no LUT spillover.** Real tools move
multiplications to LUT fabric when DSPs run out, and map narrow (<10-bit)
multiplications to LUTs *by preference*. We always charge a DSP. **This is the single
largest source of error** and it is why the SVHN CNN below is 2.1× off.

**Assumption 2.5 — no constant-folding, no strength reduction.** Multiplications by
0, ±1, or powers of two are free in real synthesis. We charge full price. This is
most of the residual +27% on the jet tagger.

### Measured DSP error

| Reference | Predicted | Published | Ratio |
|---|---|---|---|
| hls4ml jet tagger, uncompressed, 16-bit, RF=1 | 4,256 | 3,329 | **1.28** |
| hls4ml jet tagger, compressed, 16-bit, RF=1 | 1,205 | 954 | **1.26** |
| hls4ml SVHN CNN, 14-bit, RF=1 (DSP-saturated) | 13,552 | 6,377 | **2.13** |

The first two agree to within 2 points of each other across a 3.3× change in model
size, which is real evidence for a *systematic* ~+27% overestimate rather than noise.
**If you need a corrected number, multiply our DSP estimate by 0.78 for
non-saturated designs.** We do not bake that factor in, because it is fitted to one
network from one tool version, and a 2-point fit does not deserve to be a default.

---

## 3. BRAM model

```
bram18(depth, width) = ceil(width / 18) · ceil(depth / 1024)
BRAM(layer) = buffer(weights) + buffer(line/input) + buffer(output tile)
buffer(n, bits, partitions) = p · bram18(ceil(n / p), bits),  p = min(partitions, n)
```

**Assumption 3.1 — an 18Kb BRAM is 1024 × 18 bits.** True for the deepest aspect
ratio. Xilinx BRAM18s can be configured 16K×1 … 512×36; we always assume 1024×18,
which overestimates for narrow-and-deep buffers and underestimates for wide ones.

**Assumption 3.2 — partitioning fragments memory.** `ARRAY_PARTITION cyclic
factor=N` gives N independent arrays, each needing at least one whole BRAM
primitive. So BRAM is non-decreasing in `unroll`. This is what makes the
"more parallelism costs more memory" invariant hold.

**Assumption 3.3 — three buffers per MAC layer.** Weights, an input line buffer
(`in_ch · in_w · kh` for conv — enough for a sliding window), and an output tile.
Real dataflow designs may stream weights from DRAM (fewer BRAMs) or double-buffer
(more). We model neither.

**Assumption 3.4 — no LUTRAM.** Small arrays that Vivado would put in distributed
RAM are charged as whole BRAMs. Overestimates BRAM for tiny layers.

**Validation status: NONE.** Neither reference design reports BRAM in a form we can
compare (the jet tagger paper reports no BRAM; the SVHN CNN's BRAM is dominated by
hls4ml's inter-layer FIFOs, which our IR does not model). **Do not trust the BRAM
number.** It is monotonic and dimensionally sane, and that is all that is tested.

---

## 4. LUT / FF model

```
LUT(mac layer) = 120 + 2.75 · unroll · w_bits + 90
FF (mac layer) = 100 + 0.69 · unroll · (w_bits + a_bits) + 64
LUT(eltwise)   = 120 + 8 · unroll ;  FF(eltwise) = 100 + 8 · unroll
```

**Assumption 4.1 — logic scales linearly in lanes × bits.** Control logic is a fixed
base, the datapath is per-lane. Linearity is what guarantees monotonicity in
`unroll`; it ignores that wide fanout and routing pressure make real designs
super-linear at high parallelism.

**Assumption 4.2 — the coefficients are FITTED, not derived.** `2.75` and `0.69` come
from fitting total logic to the hls4ml jet tagger (263,234 LUT+FF for 4,256 lanes at
16 bits, arXiv:1804.06913 Table 2). The ~2:1 LUT:FF split comes from
arXiv:2101.05108 Table 3, the only reference reporting them separately.
**Two data points, one tool, one network family.** Before calibration the constants
were 4× too high; they could easily still be 2× off in a regime we did not sample.

**Assumption 4.3 — requantization costs a constant.** One multiply-shift-clamp chain
per output, amortized to a flat 90 LUT / 64 FF per layer regardless of `unroll`.
Wrong for heavily unrolled layers, where requant logic replicates per lane.

### Measured LUT/FF error

| Reference | Resource | Predicted | Published | Ratio |
|---|---|---|---|---|
| jet tagger, uncompressed (**fitted here**) | LUT+FF | 282,729 | 263,234 | 1.07 |
| jet tagger, compressed (independent scale) | LUT+FF | 81,121 | 88,797 | **0.91** |
| SVHN CNN 14-bit (fully independent) | LUT | 598,316 | 228,823 | **2.61** |
| SVHN CNN 14-bit (fully independent) | FF | 300,858 | 80,278 | **3.75** |

Read that honestly: **on the network we fitted, we are within 7%. On a network we did
not fit, we are 161% high on LUT and 275% high on FF.** The CNN case fails because
hls4ml streams the convolution spatially and shares one datapath across many pixels,
while our model charges for every lane independently. A model that charged per
*distinct kernel position* rather than per lane would likely close most of that gap;
that is the highest-value next modeling step.

---

## 5. Latency model

```
latency_cycles(layer) = ceil(work / unroll) · II + 8
work = MACs (conv/linear) | numel (relu/add) | in_ch·out_h·out_w·kh·kw (maxpool)
II = 1 if pipelined else 4
graph latency = max over layers      (DATAFLOW: stages run concurrently)
```

**Assumption 5.1 — II=1 is achievable whenever `pipeline` is set.** Real designs miss
II=1 on loop-carried dependencies (notably accumulation into a single register).
Optimistic.

**Assumption 5.2 — pipeline fill/drain is a constant 8 cycles.** Real depth grows with
datapath length and bit width. Negligible for large layers, wrong for tiny ones.

**Assumption 5.3 — DATAFLOW makes graph latency the max, not the sum.** This is
*throughput* (initiation interval), not end-to-end latency of a single inference.
`totals["latency_seq"]` holds the sum if you want the sequential bound. The
distinction matters and the two differ by roughly the number of layers.

**Assumption 5.4 — no memory bandwidth.** Off-chip transfers are free. For any design
that streams weights from DRAM this is badly optimistic.

**Validation status: NONE.** Neither reference reports latency in a form comparable to
our cycle model (the jet tagger is fully pipelined at 75 ns end-to-end; the SVHN
numbers include hls4ml FIFO depths we do not model). Latency is tested only for
monotonicity.

---

## 6. What the acceptance tests actually prove

| Test | What it proves | What it does *not* prove |
|---|---|---|
| Analytical sanity | DSP matches the closed form exactly | that the closed form matches hardware |
| Monotonicity (randomized) | knobs move resources the right way | magnitudes |
| Bit-width scaling | packing model is applied consistently | that packing is achievable |
| **Bit-exact codegen** | **emitted C++ == PyTorch, exactly** | that it synthesizes well |
| Budget enforcement | DSE never returns an over-budget config | that the budget check uses good estimates |
| Calibration | totals land in the stated (measured) bands | accuracy outside those two networks |

The bit-exactness test is the one that carries real weight: a mutation that drops the
rounding term kills 5 of its 6 cases, so it is not vacuous.

---

## 7. Ranked list of what would most improve accuracy

1. **Model DSP saturation and LUT spillover.** Would fix the 2.13× CNN DSP error.
2. **Charge logic per distinct kernel position, not per MAC lane, for streamed
   convolutions.** Would fix most of the 2.6×/3.75× CNN logic error.
3. **Validate BRAM against anything at all.** It is currently unvalidated.
4. **Model accumulator width properly** instead of assuming 64-bit.
5. **Model constant-folding of zero/±1 weights**, which is most of the residual +27%.

## Sources

- J. Duarte et al., "Fast inference of deep neural networks in FPGAs for particle
  physics", JINST 13 (2018) P07027, [arXiv:1804.06913](https://arxiv.org/abs/1804.06913) — Table 2.
- T. Aarrestad et al., "Fast convolutional neural networks on FPGAs with hls4ml",
  Mach. Learn.: Sci. Technol. 2 (2021) 045015,
  [arXiv:2101.05108](https://arxiv.org/abs/2101.05108) — Table 1, Table 3, Table 4, Figure 4.
- Xilinx DS190 (Zynq-7000) and DS891 (Zynq UltraScale+) for device budgets.
- Xilinx WP486 for the dual-int8-per-DSP technique referenced in Assumption 2.3.
