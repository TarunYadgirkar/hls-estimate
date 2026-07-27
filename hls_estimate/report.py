"""Per-layer resource attribution, bottleneck identification and fit verdict."""
from __future__ import annotations

from dataclasses import dataclass, field

from .devices import DeviceBudget
from .model import estimate_graph

RESOURCES = ("lut", "ff", "dsp", "bram")


@dataclass
class Row:
    name: str
    kind: str
    macs: int
    dsp: int
    bram: int
    lut: int
    ff: int
    latency: int
    share: dict = field(default_factory=dict)   # fraction of the design's total
    device_fraction: dict = field(default_factory=dict)  # fraction of the device


@dataclass
class Report:
    rows: list
    totals: dict
    utilisation: dict
    fits: bool
    binding_resource: str
    bottleneck: str
    device: DeviceBudget

    def render(self) -> str:
        head = (f"{'layer':<12}{'kind':<10}{'MACs':>10}{'DSP':>7}{'BRAM':>7}"
                f"{'LUT':>9}{'FF':>9}{'cycles':>9}  {'%dev':>6}")
        lines = [f"target: {self.device.name}", head, "-" * len(head)]
        for r in self.rows:
            worst = max(r.device_fraction.values()) if r.device_fraction else 0.0
            lines.append(f"{r.name:<12}{r.kind:<10}{r.macs:>10}{r.dsp:>7}{r.bram:>7}"
                         f"{r.lut:>9}{r.ff:>9}{r.latency:>9}  {worst:>5.1%}")
        lines.append("-" * len(head))
        t = self.totals
        lines.append(f"{'TOTAL':<22}{t['macs']:>10}{t['dsp']:>7}{t['bram']:>7}"
                     f"{t['lut']:>9}{t['ff']:>9}{t['latency']:>9}")
        lines.append("")
        for res in RESOURCES:
            cap = getattr(self.device, "bram18" if res == "bram" else res)
            lines.append(f"  {res.upper():<5} {t[res]:>9,} / {cap:>9,}"
                         f"  {self.utilisation[res]:>6.1%}")
        lines.append("")
        lines.append(f"bottleneck (slowest stage): {self.bottleneck}")
        verdict = "FITS" if self.fits else "DOES NOT FIT"
        lines.append(f"verdict: {verdict} on {self.device.name} "
                     f"(binding resource: {self.binding_resource.upper()} at "
                     f"{self.utilisation[self.binding_resource]:.1%})")
        return "\n".join(lines)


def build_report(graph, device: DeviceBudget) -> Report:
    est = estimate_graph(graph)
    caps = {"lut": device.lut, "ff": device.ff, "dsp": device.dsp,
            "bram": device.bram18}
    totals = est.totals

    rows = []
    for le in est.per_layer:
        row = Row(name=le.name, kind=le.kind, macs=le.macs, dsp=le.dsp, bram=le.bram,
                  lut=le.lut, ff=le.ff, latency=le.latency)
        row.share = {res: (getattr(le, res) / totals[res]) if totals[res] else 0.0
                     for res in RESOURCES}
        row.device_fraction = {res: getattr(le, res) / caps[res] for res in RESOURCES}
        rows.append(row)

    utilisation = {res: totals[res] / caps[res] for res in RESOURCES}
    binding = max(utilisation, key=utilisation.get)
    fits = all(v <= 1.0 for v in utilisation.values())
    return Report(rows=rows, totals=totals, utilisation=utilisation, fits=fits,
                  binding_resource=binding, bottleneck=est.bottleneck, device=device)
