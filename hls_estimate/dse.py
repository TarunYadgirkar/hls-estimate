"""Design space exploration.

Enumerate per-layer knob configurations, drop anything that exceeds the device
budget, return the Pareto front over (latency, peak resource utilisation).

Enumerate-then-filter, deliberately: the search spaces here are small (a handful of
power-of-two unroll factors per layer), and a config either fits or it does not.
Keeping it dumb makes the budget invariant trivially checkable — see SPEC §7.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .devices import DeviceBudget
from .ir import Conv2d, Knobs, Linear
from .model import bram, dsp, ff, latency_cycles, lut

MAX_CONFIGS = 20000  # backstop; the caller is told when the space is truncated


@dataclass(frozen=True)
class DesignPoint:
    config: tuple           # per-layer Knobs, positionally aligned with graph.nodes
    latency: int
    lut: int
    ff: int
    dsp: int
    bram: int
    fits: bool
    utilisation: float = 0.0
    per_layer_latency: tuple = field(default=())

    def describe(self) -> str:
        knobs = ", ".join(f"u{k.unroll}/t{k.tile}" for k in self.config)
        return (f"[{knobs}] lat={self.latency} dsp={self.dsp} bram={self.bram} "
                f"lut={self.lut} ff={self.ff} util={self.utilisation:.1%}")


def _max_parallel(node) -> int:
    """Largest sensible unroll: the layer's MAC-parallel dimension."""
    if isinstance(node, Conv2d):
        return node.in_ch * node.kh * node.kw
    if isinstance(node, Linear):
        return node.in_features
    return max(1, node.work())


def _pow2_upto(limit: int) -> list[int]:
    vals, u = [], 1
    while u <= max(1, limit):
        vals.append(u)
        u *= 2
    if limit > vals[-1]:
        vals.append(limit)
    return vals


def default_knob_grid(graph) -> list[list[Knobs]]:
    """Per-layer candidate knob settings."""
    grid = []
    for node in graph.nodes:
        unrolls = _pow2_upto(_max_parallel(node))
        opts = [Knobs(unroll=u, tile=0, pipeline=True) for u in unrolls]
        grid.append(opts)
    return grid


def evaluate(graph, config, device: DeviceBudget | None = None) -> DesignPoint:
    """Estimate one full-graph configuration."""
    lat_layers, tot = [], {"lut": 0, "ff": 0, "dsp": 0, "bram": 0}
    for node, k in zip(graph.nodes, config):
        tot["lut"] += lut(node, k)
        tot["ff"] += ff(node, k)
        tot["dsp"] += dsp(node, k)
        tot["bram"] += bram(node, k)
        lat_layers.append(latency_cycles(node, k))
    # DATAFLOW: layers run concurrently; throughput is set by the slowest stage.
    latency = max(lat_layers, default=0)
    fits, util = True, 0.0
    if device is not None:
        fits = (tot["lut"] <= device.lut and tot["ff"] <= device.ff
                and tot["dsp"] <= device.dsp and tot["bram"] <= device.bram18)
        util = max(tot["lut"] / device.lut, tot["ff"] / device.ff,
                   tot["dsp"] / device.dsp, tot["bram"] / device.bram18)
    return DesignPoint(config=tuple(config), latency=latency, lut=tot["lut"],
                       ff=tot["ff"], dsp=tot["dsp"], bram=tot["bram"], fits=fits,
                       utilisation=util, per_layer_latency=tuple(lat_layers))


def pareto_front(points: list[DesignPoint]) -> list[DesignPoint]:
    """Non-dominated over (latency ↓, utilisation ↓)."""
    front = []
    for p in points:
        dominated = any(
            q is not p
            and q.latency <= p.latency and q.utilisation <= p.utilisation
            and (q.latency < p.latency or q.utilisation < p.utilisation)
            for q in points
        )
        if not dominated:
            front.append(p)
    # De-duplicate identical (latency, utilisation) pairs, keep the cheapest DSP.
    best: dict[tuple, DesignPoint] = {}
    for p in front:
        key = (p.latency, round(p.utilisation, 12))
        if key not in best or p.dsp < best[key].dsp:
            best[key] = p
    return sorted(best.values(), key=lambda p: (p.latency, p.utilisation))


def explore(graph, device: DeviceBudget, knob_grid=None) -> list[DesignPoint]:
    """Return the Pareto front of configurations that FIT `device`.

    Configurations exceeding any of the four resource caps are never returned.
    """
    grid = knob_grid or default_knob_grid(graph)
    feasible = []
    for i, config in enumerate(itertools.product(*grid)):
        if i >= MAX_CONFIGS:
            break
        point = evaluate(graph, config, device)
        if point.fits:
            feasible.append(point)
    return pareto_front(feasible) if feasible else []


def truncated(graph, knob_grid=None) -> bool:
    grid = knob_grid or default_knob_grid(graph)
    return math.prod(len(opts) for opts in grid) > MAX_CONFIGS
