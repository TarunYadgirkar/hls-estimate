"""Acceptance #5 — DSE never returns a config exceeding the device budget."""
import pytest

from hls_estimate.devices import get_device, DEVICES
from hls_estimate.dse import explore
from hls_estimate.models import conv_relu_pool, tiny_linear


@pytest.mark.parametrize("device_name", ["zynq-7020", "ultra96"])
@pytest.mark.parametrize("factory", [conv_relu_pool, tiny_linear])
def test_dse_respects_budget(device_name, factory):
    device = get_device(device_name)
    _, graph = factory()
    front = explore(graph, device)
    assert len(front) >= 1, "DSE returned nothing; at least unroll=1 should fit"
    for pt in front:
        assert pt.lut <= device.lut
        assert pt.ff <= device.ff
        assert pt.dsp <= device.dsp
        assert pt.bram <= device.bram18
        assert pt.fits is True


def test_dse_front_is_pareto():
    device = get_device("ultra96")
    _, graph = conv_relu_pool()
    front = explore(graph, device)
    # No point may dominate another (strictly better latency AND peak utilisation).
    def util(p):
        return max(p.lut / device.lut, p.ff / device.ff,
                   p.dsp / device.dsp, p.bram / device.bram18)
    for a in front:
        for b in front:
            if a is b:
                continue
            dominates = (b.latency <= a.latency and util(b) <= util(a) and
                         (b.latency < a.latency or util(b) < util(a)))
            assert not dominates, "front contains a dominated point"


def test_all_devices_have_valid_budgets():
    for name in DEVICES:
        d = get_device(name)
        assert d.lut > 0 and d.ff > 0 and d.dsp > 0 and d.bram18 > 0
