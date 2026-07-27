"""Report: per-layer attribution, bottleneck identification, fit verdict (SPEC §8)."""
import pytest

from hls_estimate.devices import get_device
from hls_estimate.ir import Knobs
from hls_estimate.models import conv_relu_pool, tiny_linear
from hls_estimate.report import build_report


def test_report_has_one_row_per_layer():
    _, graph = conv_relu_pool()
    rep = build_report(graph, get_device("zynq-7020"))
    assert len(rep.rows) == len(graph.nodes)
    assert [r.name for r in rep.rows] == [n.name for n in graph.nodes]


def test_small_model_fits_zynq7020():
    _, graph = tiny_linear()
    rep = build_report(graph, get_device("zynq-7020"))
    assert rep.fits is True
    assert rep.binding_resource in ("lut", "ff", "dsp", "bram")


def test_oversized_model_does_not_fit_and_names_the_binding_resource():
    _, graph = conv_relu_pool()
    # Crank parallelism far past the device: DSP must become the binding resource.
    for node in graph.nodes:
        node.knobs = Knobs(unroll=4096)
    rep = build_report(graph, get_device("zynq-7020"))
    assert rep.fits is False
    assert rep.binding_resource == "dsp"
    assert rep.utilisation["dsp"] > 1.0


def test_bottleneck_is_the_slowest_layer():
    _, graph = conv_relu_pool()
    rep = build_report(graph, get_device("ultra96"))
    slowest = max(rep.rows, key=lambda r: r.latency)
    assert rep.bottleneck == slowest.name


def test_attribution_percentages_sum_to_one():
    _, graph = conv_relu_pool()
    rep = build_report(graph, get_device("ultra96"))
    for res in ("dsp", "bram", "lut", "ff"):
        total = sum(getattr(r, res) for r in rep.rows)
        if total == 0:
            continue
        share = sum(r.share[res] for r in rep.rows)
        assert share == pytest.approx(1.0, abs=1e-6)


def test_report_renders_text():
    _, graph = conv_relu_pool()
    text = build_report(graph, get_device("ultra96")).render()
    assert "bottleneck" in text.lower()
    assert "conv" in text
