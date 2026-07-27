"""Acceptance #6 — known-model calibration against FINN/hls4ml literature.

For each reference design we build the equivalent IR graph, estimate resources, and
assert the estimate lands within a stated error band. The bands live in the data
(hls_estimate.models.literature) and are set HONESTLY from measured ratios — see
MODEL.md for the discussion of where the model is accurate and where it is off.
"""
import pytest

from hls_estimate.model import estimate_graph
from hls_estimate.models.literature import LITERATURE


def test_at_least_two_reference_designs():
    assert len(LITERATURE) >= 2


@pytest.mark.parametrize("entry", LITERATURE, ids=[e.name for e in LITERATURE])
def test_estimate_within_band(entry):
    totals = estimate_graph(entry.graph).totals
    assert entry.source, f"{entry.name}: missing literature citation"
    for res, (lo, hi) in entry.band.items():
        published = entry.published[res]
        est = totals[res]
        ratio = est / published if published else float("inf")
        assert lo <= ratio <= hi, (
            f"{entry.name} {res}: estimate {est} vs published {published} "
            f"ratio {ratio:.2f} outside stated band [{lo}, {hi}] "
            f"(source: {entry.source})")
