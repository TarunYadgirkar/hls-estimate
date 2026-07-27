/**
 * Parity: the TypeScript model must agree with the Python model exactly.
 *
 * Python (hls_estimate/model.py) is the source of truth. Golden vectors are
 * regenerated with `.venv/bin/python scripts/gen_golden.py`. If this test fails,
 * the port drifted — fix the port, never the golden file.
 */
import { describe, expect, it } from "vitest";
import golden from "./golden.json";
import {
  DEVICES,
  estimateGraph,
  makeAdd,
  makeConv2d,
  makeLinear,
  makeMaxPool2d,
  makeReLU,
  resources,
  type Node,
} from "./model";
import { EXAMPLES } from "./examples";

type LayerCase = (typeof golden.layers)[number];

function nodeFor(c: LayerCase): Node {
  const s = c.spec as Record<string, number>;
  switch (c.kind) {
    case "conv2d":
      return makeConv2d(s);
    case "linear":
      return makeLinear(s);
    case "maxpool2d":
      return makeMaxPool2d(s);
    case "relu":
      return makeReLU(s);
    case "add":
      return makeAdd(s);
    default:
      throw new Error(`unknown kind ${c.kind}`);
  }
}

describe("layer model parity with Python", () => {
  for (const c of golden.layers as LayerCase[]) {
    const label = `${c.kind} ${JSON.stringify(c.spec)} u=${c.unroll} t=${c.tile}`;
    it(label, () => {
      const node = nodeFor(c);
      const knobs = { unroll: c.unroll, tile: c.tile, pipeline: true };
      const r = resources(node, knobs);
      expect(r.dsp).toBe(c.dsp);
      expect(r.bram).toBe(c.bram);
      expect(r.lut).toBe(c.lut);
      expect(r.ff).toBe(c.ff);
      expect(r.latency).toBe(c.latency);
    });
  }
});

describe("graph totals parity with Python", () => {
  for (const g of golden.graphs) {
    it(g.name, () => {
      const graph = EXAMPLES[g.name as keyof typeof EXAMPLES];
      expect(graph, `example ${g.name} missing from web examples`).toBeTruthy();
      const est = estimateGraph(graph.build());
      for (const key of ["dsp", "bram", "lut", "ff", "macs", "latency"] as const) {
        expect(est.totals[key], `${g.name}.${key}`).toBe(
          (g.totals as Record<string, number>)[key],
        );
      }
      expect(est.bottleneck).toBe(g.bottleneck);
      expect(est.perLayer.map((l) => l.name)).toEqual(
        g.per_layer.map((l) => l.name),
      );
    });
  }
});

describe("device budgets parity with Python", () => {
  for (const [name, d] of Object.entries(golden.devices)) {
    it(name, () => {
      const dev = DEVICES[name];
      expect(dev).toBeTruthy();
      expect(dev.lut).toBe(d.lut);
      expect(dev.ff).toBe(d.ff);
      expect(dev.dsp).toBe(d.dsp);
      expect(dev.bram18).toBe(d.bram18);
    });
  }
});
