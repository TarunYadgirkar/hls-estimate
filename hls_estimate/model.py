"""Analytical resource + latency model. Pure functions; no synthesis.

All closed forms and the reasoning behind every constant are documented in MODEL.md.
The model is deliberately simple and, above all, *monotonic* in the knobs so that DSE
behaves sanely: more parallelism costs more DSP/LUT/FF and buys lower latency; bigger
tiles cost more BRAM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ir import Conv2d, Linear, ReLU, MaxPool2d, Add

# --- DSP packing model (see MODEL.md §DSP) ---------------------------------
# One DSP performs one 8-bit MAC. Narrower operands pack along the shared-operand
# (parallelism) axis: two int4 MACs, or four int2 MACs, per DSP.
MACS_PER_DSP = {2: 4, 4: 2, 8: 1, 16: 1}

# --- BRAM primitive --------------------------------------------------------
BRAM18_BITS = 18 * 1024          # 18 Kb
BRAM18_WIDTH = 18                # widest single-port width used for the depth calc

# --- LUT/FF datapath constants (coarse; calibrated in MODEL.md §LUT/FF) -----
LUT_BASE = 120
LUT_PER_LANE = 6.0               # LUT per MAC lane per weight-bit
LUT_REQUANT = 90
FF_BASE = 100
FF_PER_LANE = 5.0                # FF per MAC lane per (w+a) bit
FF_REQUANT = 64
ELTWISE_LUT_PER_LANE = 8
ELTWISE_FF_PER_LANE = 8

# --- Latency constants -----------------------------------------------------
PIPELINE_DEPTH = 8               # fill/drain of the innermost pipeline
II_SEQ = 4                       # initiation interval when not pipelined


def macs_per_dsp(bits: int) -> int:
    try:
        return MACS_PER_DSP[bits]
    except KeyError:
        raise ValueError(f"no packing model for {bits}-bit; known {sorted(MACS_PER_DSP)}")


def _knobs(node, knobs):
    return knobs if knobs is not None else node.knobs


def _unroll(node, knobs) -> int:
    return max(1, _knobs(node, knobs).unroll)


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------
def dsp(node, knobs=None) -> int:
    if not getattr(node, "is_mac", False):
        return 0
    u = _unroll(node, knobs)
    return math.ceil(u / macs_per_dsp(node.w_bits))


# ---------------------------------------------------------------------------
# BRAM
# ---------------------------------------------------------------------------
def bram18(depth: int, width_bits: int) -> int:
    """Number of 18Kb BRAMs to hold `depth` words of `width_bits` each."""
    if depth <= 0 or width_bits <= 0:
        return 0
    return math.ceil(width_bits / BRAM18_WIDTH) * math.ceil(depth / 1024)


def _buffer_bram(num_elems: int, bits: int, partitions: int) -> int:
    """A logical buffer of `num_elems` words at `bits`, split `partitions` ways.

    Partitioning increases BRAM count because each partition needs >= 1 primitive
    (fragmentation) -> monotonic non-decreasing in `partitions`.
    """
    if num_elems <= 0:
        return 0
    p = max(1, min(partitions, num_elems))
    per_partition = math.ceil(num_elems / p)
    return p * bram18(per_partition, bits)


def bram(node, knobs=None) -> int:
    k = _knobs(node, knobs)
    u = max(1, k.unroll)
    if isinstance(node, Conv2d):
        w_elems = node.out_ch * node.in_ch * node.kh * node.kw
        bram_w = _buffer_bram(w_elems, node.w_bits, u)
        line_elems = node.in_ch * node.in_w * node.kh
        bram_line = _buffer_bram(line_elems, node.a_bits, 1)
        tile_elems = k.tile if k.tile > 0 else node.out_ch * node.out_h() * node.out_w()
        bram_out = _buffer_bram(tile_elems, node.out_bits, u)
        return bram_w + bram_line + bram_out
    if isinstance(node, Linear):
        w_elems = node.out_features * node.in_features
        bram_w = _buffer_bram(w_elems, node.w_bits, u)
        bram_in = _buffer_bram(node.in_features, node.a_bits, 1)
        tile_elems = k.tile if k.tile > 0 else node.out_features
        bram_out = _buffer_bram(tile_elems, node.out_bits, u)
        return bram_w + bram_in + bram_out
    if isinstance(node, MaxPool2d):
        return _buffer_bram(node.in_ch * node.in_w * node.kh, node.bits, 1)
    # ReLU / Add are streamed -> negligible BRAM
    return 0


# ---------------------------------------------------------------------------
# LUT / FF
# ---------------------------------------------------------------------------
def lut(node, knobs=None) -> int:
    u = _unroll(node, knobs)
    if getattr(node, "is_mac", False):
        return int(LUT_BASE + LUT_PER_LANE * u * node.w_bits + LUT_REQUANT)
    return int(LUT_BASE + ELTWISE_LUT_PER_LANE * u)


def ff(node, knobs=None) -> int:
    u = _unroll(node, knobs)
    if getattr(node, "is_mac", False):
        return int(FF_BASE + FF_PER_LANE * u * (node.w_bits + node.a_bits) + FF_REQUANT)
    return int(FF_BASE + ELTWISE_FF_PER_LANE * u)


# ---------------------------------------------------------------------------
# Latency (cycles)
# ---------------------------------------------------------------------------
def latency_cycles(node, knobs=None) -> int:
    k = _knobs(node, knobs)
    u = max(1, k.unroll)
    work = node.work()
    ii = 1 if k.pipeline else II_SEQ
    return math.ceil(work / u) * ii + PIPELINE_DEPTH


# ---------------------------------------------------------------------------
# Whole-graph estimate
# ---------------------------------------------------------------------------
@dataclass
class LayerEstimate:
    name: str
    kind: str
    macs: int
    dsp: int
    bram: int
    lut: int
    ff: int
    latency: int


@dataclass
class GraphEstimate:
    per_layer: list
    totals: dict
    bottleneck: str

    def __str__(self):
        rows = [f"{'layer':<14}{'kind':<10}{'MACs':>10}{'DSP':>7}{'BRAM':>7}"
                f"{'LUT':>9}{'FF':>9}{'cyc':>9}"]
        for le in self.per_layer:
            rows.append(f"{le.name:<14}{le.kind:<10}{le.macs:>10}{le.dsp:>7}{le.bram:>7}"
                        f"{le.lut:>9}{le.ff:>9}{le.latency:>9}")
        t = self.totals
        rows.append(f"{'TOTAL':<24}{t['macs']:>10}{t['dsp']:>7}{t['bram']:>7}"
                    f"{t['lut']:>9}{t['ff']:>9}{t['latency']:>9}")
        rows.append(f"bottleneck: {self.bottleneck}")
        return "\n".join(rows)


def estimate_graph(graph) -> GraphEstimate:
    per_layer = []
    for n in graph.nodes:
        per_layer.append(LayerEstimate(
            name=n.name, kind=n.kind, macs=n.macs(),
            dsp=dsp(n), bram=bram(n), lut=lut(n), ff=ff(n),
            latency=latency_cycles(n)))
    totals = {
        "macs": sum(le.macs for le in per_layer),
        "dsp": sum(le.dsp for le in per_layer),
        "bram": sum(le.bram for le in per_layer),
        "lut": sum(le.lut for le in per_layer),
        "ff": sum(le.ff for le in per_layer),
        # DATAFLOW: all layers run concurrently -> throughput limited by slowest layer.
        "latency": max((le.latency for le in per_layer), default=0),
        "latency_seq": sum(le.latency for le in per_layer),
    }
    bottleneck = max(per_layer, key=lambda le: le.latency).name if per_layer else ""
    return GraphEstimate(per_layer, totals, bottleneck)
