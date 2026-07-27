"""Command line interface."""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys

from . import models
from .codegen import emit_full, emit_graph
from .devices import DEVICES, get_device
from .dse import explore
from .report import build_report

EXAMPLES = {f.__name__: f for f in models.ALL_EXAMPLES}


def load_graph(spec: str):
    """Resolve a bundled example name, or a `module:factory` entry point."""
    if spec in EXAMPLES:
        return EXAMPLES[spec]()[1]
    if ":" in spec:
        mod_name, _, attr = spec.partition(":")
        mod = importlib.import_module(mod_name)
        result = getattr(mod, attr)()
        return result[1] if isinstance(result, tuple) else result
    raise KeyError(f"unknown model {spec!r}; try one of {sorted(EXAMPLES)} "
                   f"or 'module:factory'")


def _report_json(rep) -> str:
    return json.dumps({
        "device": rep.device.name,
        "fits": rep.fits,
        "binding_resource": rep.binding_resource,
        "bottleneck": rep.bottleneck,
        "totals": {k: v for k, v in rep.totals.items()},
        "utilisation": rep.utilisation,
        "layers": [{
            "name": r.name, "kind": r.kind, "macs": r.macs, "dsp": r.dsp,
            "bram": r.bram, "lut": r.lut, "ff": r.ff, "latency": r.latency,
            "share": r.share, "device_fraction": r.device_fraction,
        } for r in rep.rows],
    }, indent=2)


def _maybe_validate() -> str:
    """Optional Vitis validation. Never required — reports its absence and moves on."""
    vitis = shutil.which("vitis_hls")
    if not vitis:
        return ("\n[validate] Vitis HLS not found on PATH; skipping real synthesis. "
                "Estimates above are analytical only.")
    return (f"\n[validate] Found Vitis HLS at {vitis}. Run "
            "`hls-estimate emit MODEL -o net.cpp` and synthesize it to compare; "
            "automated csynth comparison is not implemented (see HANDOFF.md).")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="hls-estimate",
        description="Analytical FPGA resource/latency estimation and HLS C++ codegen "
                    "for small quantized models. No synthesis required.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list bundled example models")

    p_est = sub.add_parser("estimate", help="estimate resources and fit verdict")
    p_est.add_argument("model")
    p_est.add_argument("--device", default="zynq-7020", choices=sorted(DEVICES))
    p_est.add_argument("--json", action="store_true")
    p_est.add_argument("--validate", action="store_true",
                       help="also run Vitis HLS if it happens to be installed")

    p_emit = sub.add_parser("emit", help="write synthesizable HLS C++")
    p_emit.add_argument("model")
    p_emit.add_argument("-o", "--output", required=True)
    p_emit.add_argument("--tb", action="store_true", help="include a test bench main()")

    p_dse = sub.add_parser("dse", help="explore tiling/unroll configs")
    p_dse.add_argument("model")
    p_dse.add_argument("--device", default="zynq-7020", choices=sorted(DEVICES))
    p_dse.add_argument("--limit", type=int, default=12)

    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name in sorted(EXAMPLES):
            print(name)
        return 0

    try:
        graph = load_graph(args.model)
    except (KeyError, ImportError, AttributeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "estimate":
        rep = build_report(graph, get_device(args.device))
        print(_report_json(rep) if args.json else rep.render())
        if args.validate:
            print(_maybe_validate())
        return 0

    if args.cmd == "emit":
        source = emit_full(graph, "net") if args.tb else emit_graph(graph, "net")
        with open(args.output, "w") as f:
            f.write(source)
        print(f"wrote {args.output} ({len(source.splitlines())} lines)")
        return 0

    if args.cmd == "dse":
        device = get_device(args.device)
        front = explore(graph, device)
        if not front:
            print(f"no configuration fits {device.name}")
            return 0
        print(f"Pareto front on {device.name} ({len(front)} points, "
              f"latency vs peak utilisation):")
        print(f"{'latency':>10}{'DSP':>8}{'BRAM':>8}{'LUT':>10}{'FF':>10}{'util':>8}  config")
        for pt in front[:args.limit]:
            knobs = ",".join(f"u{k.unroll}" for k in pt.config)
            print(f"{pt.latency:>10}{pt.dsp:>8}{pt.bram:>8}{pt.lut:>10}{pt.ff:>10}"
                  f"{pt.utilisation:>7.1%}  [{knobs}]")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
