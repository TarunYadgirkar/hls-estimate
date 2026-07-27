"""Acceptance #3 — bit-width scaling follows the documented packing model.

macs_per_dsp: {16:1, 8:1, 4:2, 2:4}. DSP == ceil(unroll / macs_per_dsp(w_bits)).
int4 uses half the DSP of int8 at equal parallelism.
"""
import math
from dataclasses import replace

import pytest

from hls_estimate.ir import Conv2d, Knobs
from hls_estimate.model import dsp, macs_per_dsp, MACS_PER_DSP


def _conv(w_bits):
    return Conv2d(in_ch=8, out_ch=16, kh=3, kw=3, in_h=16, in_w=16, w_bits=w_bits, a_bits=8)


def test_packing_table():
    assert MACS_PER_DSP == {2: 4, 4: 2, 8: 1, 16: 1}
    for b, m in MACS_PER_DSP.items():
        assert macs_per_dsp(b) == m


@pytest.mark.parametrize("w_bits", [2, 4, 8, 16])
@pytest.mark.parametrize("unroll", [1, 2, 3, 8, 15, 64])
def test_dsp_follows_packing(w_bits, unroll):
    conv = _conv(w_bits)
    assert dsp(conv, Knobs(unroll=unroll)) == math.ceil(unroll / MACS_PER_DSP[w_bits])


@pytest.mark.parametrize("unroll", [2, 4, 8, 64, 256])
def test_int4_is_half_of_int8(unroll):
    c8 = _conv(8)
    c4 = replace(c8, w_bits=4)
    assert dsp(c4, Knobs(unroll=unroll)) == dsp(c8, Knobs(unroll=unroll)) // 2


def test_int8_never_cheaper_than_int4():
    c8, c4 = _conv(8), _conv(4)
    for u in range(1, 65):
        assert dsp(c8, Knobs(unroll=u)) >= dsp(c4, Knobs(unroll=u))
