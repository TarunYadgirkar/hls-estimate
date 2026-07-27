"""Acceptance #2 — monotonicity invariants over randomized configs.

- doubling unroll must not decrease DSP
- doubling unroll must not decrease LUT/FF (more datapath)
- increasing tile size must not decrease BRAM
- latency must strictly decrease as parallelism increases (until compute is 1 cycle)
"""
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from hls_estimate.ir import Conv2d, Linear, Knobs
from hls_estimate.model import dsp, bram, lut, ff, latency_cycles


@st.composite
def conv_nodes(draw):
    return Conv2d(
        in_ch=draw(st.integers(1, 32)),
        out_ch=draw(st.integers(1, 64)),
        kh=draw(st.sampled_from([1, 3, 5])),
        kw=draw(st.sampled_from([1, 3, 5])),
        in_h=draw(st.integers(4, 32)),
        in_w=draw(st.integers(4, 32)),
        stride=draw(st.sampled_from([1, 2])),
        pad=draw(st.sampled_from([0, 1])),
        w_bits=draw(st.sampled_from([4, 8])),
    )


@st.composite
def linear_nodes(draw):
    return Linear(
        in_features=draw(st.integers(1, 512)),
        out_features=draw(st.integers(1, 512)),
        w_bits=draw(st.sampled_from([4, 8])),
    )


def _valid(node):
    return node.out_h() >= 1 and node.out_w() >= 1 if isinstance(node, Conv2d) else True


@settings(max_examples=250, deadline=None)
@given(node=st.one_of(conv_nodes(), linear_nodes()),
       unroll=st.integers(1, 64), tile=st.integers(1, 128))
def test_doubling_unroll_never_decreases_dsp(node, unroll, tile):
    if not _valid(node):
        return
    a = dsp(node, Knobs(unroll=unroll, tile=tile))
    b = dsp(node, Knobs(unroll=2 * unroll, tile=tile))
    assert b >= a


@settings(max_examples=250, deadline=None)
@given(node=st.one_of(conv_nodes(), linear_nodes()),
       unroll=st.integers(1, 64), tile=st.integers(1, 128))
def test_doubling_unroll_never_decreases_lut_ff(node, unroll, tile):
    if not _valid(node):
        return
    assert lut(node, Knobs(unroll=2 * unroll, tile=tile)) >= lut(node, Knobs(unroll=unroll, tile=tile))
    assert ff(node, Knobs(unroll=2 * unroll, tile=tile)) >= ff(node, Knobs(unroll=unroll, tile=tile))


@settings(max_examples=250, deadline=None)
@given(node=st.one_of(conv_nodes(), linear_nodes()),
       unroll=st.integers(1, 32), tile=st.integers(1, 64))
def test_increasing_tile_never_decreases_bram(node, unroll, tile):
    if not _valid(node):
        return
    small = bram(node, Knobs(unroll=unroll, tile=tile))
    large = bram(node, Knobs(unroll=unroll, tile=2 * tile))
    assert large >= small


@settings(max_examples=250, deadline=None)
@given(node=st.one_of(conv_nodes(), linear_nodes()), unroll=st.integers(1, 64))
def test_more_parallelism_strictly_decreases_latency(node, unroll):
    if not _valid(node):
        return
    macs = node.macs()
    slow = latency_cycles(node, Knobs(unroll=unroll))
    fast = latency_cycles(node, Knobs(unroll=2 * unroll))
    if math.ceil(macs / unroll) > 1:
        assert fast < slow
    else:
        assert fast <= slow
