"""Published FPGA resource numbers for calibration.

Every number here is quoted from a peer-reviewed paper (citation on each entry) and
was captured once into this file so the calibration test is deterministic and offline.

The reference designs are reconstructed as IR graphs with the *same parallelism the
published design used*, so the comparison is apples-to-apples:
- hls4ml's "reuse factor" R means each multiplier is reused R times, so the number of
  instantiated multipliers in a layer is `multiplications / R`. We set each layer's
  `unroll` to that number.
- For dense (MLP) layers hls4ml unrolls the whole layer, so multiplications == MACs.
- For hls4ml convolutional layers the sliding window is *streamed*: the kernel is
  parallel but the spatial loop is not, so instantiated multipliers per conv layer are
  `out_ch * in_ch * kh * kw / R`, not the full MAC count.

Read MODEL.md §Calibration for the measured error and what it means. Where the model
is badly wrong (narrow bit widths, DSP-saturated designs) it says so with the number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ir import Conv2d, Graph, Knobs, Linear, MaxPool2d, ReLU, TensorSpec


@dataclass
class LiteratureEntry:
    name: str
    source: str
    device: str
    graph: Graph
    published: dict
    band: dict
    notes: str = ""


# ---------------------------------------------------------------------------
# Reference 1 & 2 — hls4ml jet substructure tagger (MLP)
#   J. Duarte et al., "Fast inference of deep neural networks in FPGAs for particle
#   physics", JINST 13 (2018) P07027, arXiv:1804.06913, Table 2.
#   Topology: 16 inputs -> 64 -> 32 -> 32 -> 5 outputs, fixed-point <16,6>,
#   fully pipelined at 200 MHz, reuse factor 1, Xilinx Kintex UltraScale.
#   Published: uncompressed 4389 params, DSP48E 3329, Logic (LUT+FF) 263,234, 75 ns.
#              compressed   1338 params, DSP48E  954, Logic (LUT+FF)  88,797, 75 ns.
#   4389 params = 4256 multiplications + 133 biases; compressed 1338 = 1205 + 133.
# ---------------------------------------------------------------------------
JET_LAYERS = [(16, 64), (64, 32), (32, 32), (32, 5)]
JET_MULTS = sum(i * o for i, o in JET_LAYERS)          # 4256
JET_PRUNED_MULTS = 1205


def _jet_tagger(retained_mults: int) -> Graph:
    """Dense jet tagger at reuse factor 1.

    `retained_mults` is the number of multiplications actually instantiated (the
    pruned model drops zero weights, which HLS removes entirely). It is spread over
    the layers in proportion to layer size; total DSP is the sum either way.
    """
    scale = retained_mults / JET_MULTS
    nodes, prev = [], "x"
    for idx, (i, o) in enumerate(JET_LAYERS):
        out = f"h{idx}"
        unroll = max(1, round(i * o * scale))
        nodes.append(Linear(
            in_features=i, out_features=o, w_bits=16, a_bits=16, out_bits=16,
            bias=True, relu=idx < len(JET_LAYERS) - 1, mult=1, shift=8,
            name=f"fc{idx}", inputs=(prev,), output=out,
            knobs=Knobs(unroll=unroll, tile=0, pipeline=True)))
        prev = out
    return Graph(nodes, TensorSpec((1, 16), bits=16, name="x"), nodes[-1].out_spec())


# ---------------------------------------------------------------------------
# Reference 3 & 4 — hls4ml SVHN convolutional classifier (a real CNN)
#   T. Aarrestad et al., "Fast convolutional neural networks on FPGAs with hls4ml",
#   Mach. Learn.: Sci. Technol. 2 (2021) 045015, arXiv:2101.05108.
#   Architecture (their Fig. 4 / Table 1), input 32x32x3, all convs 3x3 valid, no bias
#   except the output layer, each conv block = conv -> maxpool(2,2) -> BN -> ReLU:
#     Conv0 f=16 (32,32,3)   ->30x30x16 -> pool 15x15x16
#     Conv1 f=16 (15,15,16)  ->13x13x16 -> pool  6x6x16
#     Conv2 f=24 (6,6,16)    -> 4x4x24  -> pool  2x2x24 = 96
#     Dense0 96->42, Dense1 42->64, Output 64->10
#   Published Table 3 (Xilinx Virtex UltraScale+ VU9P, reuse factor 1):
#     BF 14-bit: DSP 6,377 (93.2%), LUT 228,823, FF 80,278, BRAM 66.5
#   Published Table 4 (Xilinx Zynq XC7Z020, QP 7-bit, 100 MHz):
#     DSP 213 (97%), LUT 48,259 (91%), FF 35,118 (33%), BRAM18 122 (44%),
#     latency 17,085 cc (171 us), II 16,385 cc.
# ---------------------------------------------------------------------------
def _svhn_cnn(bits: int, reuse: int, sparsity: float = 0.0) -> Graph:
    """SVHN CNN with hls4ml-style streamed convolutions.

    Instantiated multipliers per conv layer = out_ch*in_ch*kh*kw / reuse (the spatial
    loop is streamed, one pixel at a time). Dense layers are fully unrolled / reuse.
    """
    keep = 1.0 - sparsity
    nodes, prev = [], "x"

    convs = [  # (in_ch, out_ch, in_h, in_w)
        (3, 16, 32, 32),
        (16, 16, 15, 15),
        (16, 24, 6, 6),
    ]
    for idx, (ic, oc, h, w) in enumerate(convs):
        kernel_mults = oc * ic * 3 * 3
        unroll = max(1, round(kernel_mults * keep / reuse))
        conv = Conv2d(in_ch=ic, out_ch=oc, kh=3, kw=3, in_h=h, in_w=w,
                      stride=1, pad=0, w_bits=bits, a_bits=bits, out_bits=bits,
                      bias=False, relu=False, mult=1, shift=8,
                      name=f"conv{idx}", inputs=(prev,), output=f"c{idx}",
                      knobs=Knobs(unroll=unroll, tile=0, pipeline=True))
        nodes.append(conv)
        pool = MaxPool2d(in_ch=oc, in_h=conv.out_h(), in_w=conv.out_w(),
                         kh=2, kw=2, stride=2, bits=bits,
                         name=f"pool{idx}", inputs=(f"c{idx}",), output=f"p{idx}",
                         knobs=Knobs(unroll=1, pipeline=True))
        nodes.append(pool)
        relu = ReLU(numel=pool.out_spec().numel(), bits=bits, shape=pool.out_spec().shape,
                    name=f"relu{idx}", inputs=(f"p{idx}",), output=f"r{idx}",
                    knobs=Knobs(unroll=1, pipeline=True))
        nodes.append(relu)
        prev = f"r{idx}"

    denses = [(96, 42), (42, 64), (64, 10)]
    for idx, (i, o) in enumerate(denses):
        unroll = max(1, round(i * o * keep / reuse))
        fc = Linear(in_features=i, out_features=o, w_bits=bits, a_bits=bits,
                    out_bits=bits, bias=(idx == len(denses) - 1),
                    relu=idx < len(denses) - 1, mult=1, shift=8,
                    name=f"fc{idx}", inputs=(prev,), output=f"d{idx}",
                    knobs=Knobs(unroll=unroll, tile=0, pipeline=True))
        nodes.append(fc)
        prev = f"d{idx}"

    return Graph(nodes, TensorSpec((1, 3, 32, 32), bits=bits, name="x"),
                 nodes[-1].out_spec())


# ---------------------------------------------------------------------------
# Bands: measured, then widened slightly so the test is a regression guard rather
# than a restatement of the current constants. See MODEL.md §Calibration.
# ---------------------------------------------------------------------------
LITERATURE = [
    LiteratureEntry(
        name="hls4ml-jet-tagger-uncompressed",
        source=("J. Duarte et al., JINST 13 (2018) P07027, arXiv:1804.06913, Table 2 "
                "(Kintex UltraScale, <16,6>, reuse factor 1, 200 MHz)"),
        device="Xilinx Kintex UltraScale",
        graph=_jet_tagger(JET_MULTS),
        published={"dsp": 3329, "logic": 263234},
        band={"dsp": (1.10, 1.45), "logic": (0.85, 1.30)},
        notes=("Measured: DSP +28%, logic +7%. Fully parallel 16-bit MLP -- the regime "
               "the DSP model targets. We predict one DSP per multiplier; the tool "
               "folds ~22% of them away. The LUT/FF coefficients were FITTED here, so "
               "the logic ratio is not an independent validation for this entry."),
    ),
    LiteratureEntry(
        name="hls4ml-jet-tagger-compressed",
        source=("J. Duarte et al., JINST 13 (2018) P07027, arXiv:1804.06913, Table 2 "
                "(compressed model, 1338 parameters)"),
        device="Xilinx Kintex UltraScale",
        graph=_jet_tagger(JET_PRUNED_MULTS),
        published={"dsp": 954, "logic": 88797},
        band={"dsp": (1.10, 1.45), "logic": (0.75, 1.10)},
        notes=("Measured: DSP +26%, logic -9%. Pruned to 1205 multiplications -- an "
               "independent check (3.3x smaller) that the DSP overestimate factor and "
               "the fitted logic coefficients hold at a different scale."),
    ),
    LiteratureEntry(
        name="hls4ml-svhn-cnn-14bit",
        source=("T. Aarrestad et al., Mach. Learn.: Sci. Technol. 2 (2021) 045015, "
                "arXiv:2101.05108, Table 3 (BF 14-bit, VU9P, reuse factor 1)"),
        device="Xilinx Virtex UltraScale+ VU9P",
        graph=_svhn_cnn(bits=16, reuse=1),
        published={"dsp": 6377, "lut": 228823, "ff": 80278},
        band={"dsp": (1.80, 2.50), "lut": (2.10, 3.20), "ff": (3.00, 4.60)},
        notes=("Measured: DSP +113%, LUT +161%, FF +275%. Fully independent of the "
               "fit (different network, device and topology class). The design is "
               "DSP-saturated at 93.2%, so the tool spills multiplications into LUT "
               "fabric and shares datapath across the streamed spatial loop; we model "
               "neither. This is the model's worst honest case -- see MODEL.md."),
    ),
]
