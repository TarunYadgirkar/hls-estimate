"""Generate golden vectors so the TypeScript model port can be checked against Python.

The Python model is the source of truth. Run this whenever model.py changes:

    .venv/bin/python scripts/gen_golden.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hls_estimate.devices import DEVICES
from hls_estimate.ir import Conv2d, Knobs, Linear, MaxPool2d, ReLU, Add
from hls_estimate.model import bram, dsp, estimate_graph, ff, latency_cycles, lut
from hls_estimate.models import ALL_EXAMPLES

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "web", "lib", "golden.json")

CONV_CASES = [
    dict(in_ch=16, out_ch=32, kh=3, kw=3, in_h=8, in_w=8, stride=1, pad=0, w_bits=8, a_bits=8),
    dict(in_ch=3, out_ch=16, kh=3, kw=3, in_h=32, in_w=32, stride=1, pad=1, w_bits=8, a_bits=8),
    dict(in_ch=8, out_ch=8, kh=5, kw=5, in_h=16, in_w=16, stride=2, pad=2, w_bits=4, a_bits=8),
    dict(in_ch=32, out_ch=64, kh=1, kw=1, in_h=7, in_w=7, stride=1, pad=0, w_bits=16, a_bits=16),
]
LINEAR_CASES = [
    dict(in_features=8, out_features=6, w_bits=8, a_bits=8),
    dict(in_features=512, out_features=128, w_bits=4, a_bits=8),
    dict(in_features=64, out_features=32, w_bits=16, a_bits=16),
]
KNOBS = [(1, 0), (2, 0), (4, 16), (16, 64), (64, 256), (128, 1024)]


def layer_rows():
    rows = []
    for spec in CONV_CASES:
        node = Conv2d(**spec)
        for u, t in KNOBS:
            k = Knobs(unroll=u, tile=t)
            rows.append({
                "kind": "conv2d", "spec": spec, "unroll": u, "tile": t,
                "macs": node.macs(), "out_h": node.out_h(), "out_w": node.out_w(),
                "dsp": dsp(node, k), "bram": bram(node, k), "lut": lut(node, k),
                "ff": ff(node, k), "latency": latency_cycles(node, k),
            })
    for spec in LINEAR_CASES:
        node = Linear(**spec)
        for u, t in KNOBS:
            k = Knobs(unroll=u, tile=t)
            rows.append({
                "kind": "linear", "spec": spec, "unroll": u, "tile": t,
                "macs": node.macs(),
                "dsp": dsp(node, k), "bram": bram(node, k), "lut": lut(node, k),
                "ff": ff(node, k), "latency": latency_cycles(node, k),
            })
    pool = MaxPool2d(in_ch=8, in_h=16, in_w=16, kh=2, kw=2, stride=2, bits=8)
    relu = ReLU(numel=1024, bits=8)
    add = Add(numel=512, bits=8)
    for node, kind in ((pool, "maxpool2d"), (relu, "relu"), (add, "add")):
        for u, t in KNOBS:
            k = Knobs(unroll=u, tile=t)
            rows.append({
                "kind": kind,
                "spec": ({"in_ch": 8, "in_h": 16, "in_w": 16, "kh": 2, "kw": 2,
                          "stride": 2, "bits": 8} if kind == "maxpool2d"
                         else {"numel": node.numel, "bits": 8}),
                "unroll": u, "tile": t, "macs": 0, "work": node.work(),
                "dsp": dsp(node, k), "bram": bram(node, k), "lut": lut(node, k),
                "ff": ff(node, k), "latency": latency_cycles(node, k),
            })
    return rows


def graph_rows():
    rows = []
    for factory in ALL_EXAMPLES:
        _, g = factory()
        est = estimate_graph(g)
        rows.append({
            "name": factory.__name__,
            "totals": est.totals,
            "bottleneck": est.bottleneck,
            "per_layer": [{"name": le.name, "kind": le.kind, "macs": le.macs,
                           "dsp": le.dsp, "bram": le.bram, "lut": le.lut,
                           "ff": le.ff, "latency": le.latency}
                          for le in est.per_layer],
        })
    return rows


CODEGEN_UNROLLS = [1, 2, 8, 32]


def codegen_rows():
    """Emitted HLS C++ (weights elided) for example x unroll, so the web viewer's
    TypeScript emitter can be checked against the real Python emitter."""
    from hls_estimate.codegen import emit_graph

    rows = []
    for factory in ALL_EXAMPLES:
        for u in CODEGEN_UNROLLS:
            _, g = factory()
            for node in g.nodes:
                node.knobs = Knobs(unroll=u, tile=0, pipeline=True)
            rows.append({"name": factory.__name__, "unroll": u,
                         "source": emit_graph(g, "net", elide_weights=True)})
    return rows


def dse_rows():
    from hls_estimate.devices import get_device
    from hls_estimate.dse import explore

    rows = []
    for factory in ALL_EXAMPLES:
        for dev_name in ("zynq-7020", "ultra96"):
            _, g = factory()
            front = explore(g, get_device(dev_name))
            rows.append({
                "name": factory.__name__, "device": dev_name,
                "front": [{"unrolls": [k.unroll for k in pt.config],
                           "latency": pt.latency, "dsp": pt.dsp, "bram": pt.bram,
                           "lut": pt.lut, "ff": pt.ff,
                           "utilisation": pt.utilisation} for pt in front],
            })
    return rows


def main():
    data = {
        "note": "Generated by scripts/gen_golden.py. Python is the source of truth.",
        "devices": {name: {"lut": d.lut, "ff": d.ff, "dsp": d.dsp, "bram18": d.bram18}
                    for name, d in DEVICES.items()},
        "layers": layer_rows(),
        "graphs": graph_rows(),
        "codegen": codegen_rows(),
        "dse": dse_rows(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {OUT}: {len(data['layers'])} layer cases, "
          f"{len(data['graphs'])} graph cases")


if __name__ == "__main__":
    main()
