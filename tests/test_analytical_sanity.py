"""Acceptance #1 — analytical sanity.

For a hand-computed conv layer, predicted DSP must equal the closed-form
MAC/parallelism calculation exactly. The expected values here are written by hand
(literal packing constants), independent of the model's internal implementation.
"""
import math
from dataclasses import replace

from hls_estimate.ir import Conv2d, Knobs
from hls_estimate.model import dsp


def _conv(w_bits=8):
    # 3x3 conv, 16->32 channels, 8x8 input, valid padding, stride 1 -> 6x6 output.
    return Conv2d(in_ch=16, out_ch=32, kh=3, kw=3, in_h=8, in_w=8,
                  stride=1, pad=0, w_bits=w_bits, a_bits=8)


def test_macs_closed_form():
    conv = _conv()
    out_h = (8 + 2 * 0 - 3) // 1 + 1  # = 6
    out_w = 6
    expected_macs = 32 * out_h * out_w * 16 * 3 * 3
    assert conv.out_h() == out_h
    assert conv.out_w() == out_w
    assert conv.macs() == expected_macs
    assert conv.macs() == 165888


def test_dsp_int8_equals_unroll():
    # 8-bit: exactly one MAC per DSP -> DSP == unroll.
    conv = _conv(w_bits=8)
    for unroll in (1, 2, 8, 64, 100):
        assert dsp(conv, Knobs(unroll=unroll)) == math.ceil(unroll / 1)


def test_dsp_int4_is_half():
    # 4-bit: two packed MACs per DSP -> DSP == ceil(unroll / 2).
    conv = _conv(w_bits=4)
    assert dsp(conv, Knobs(unroll=64)) == 32
    assert dsp(conv, Knobs(unroll=1)) == 1   # ceil(1/2)
    assert dsp(conv, Knobs(unroll=7)) == 4   # ceil(7/2)


def test_dsp_matches_hand_computation():
    conv8 = _conv(w_bits=8)
    conv4 = replace(conv8, w_bits=4)
    # Same parallelism, int4 uses exactly half the DSPs (even unroll).
    for unroll in (2, 4, 16, 64):
        assert dsp(conv4, Knobs(unroll=unroll)) == dsp(conv8, Knobs(unroll=unroll)) // 2
