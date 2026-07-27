/**
 * The TypeScript emitter shown in the web UI must produce byte-identical output to
 * the real Python emitter (with weight literals elided). If this fails the UI is
 * showing code that hls-estimate would not actually generate — fix the port.
 */
import { describe, expect, it } from "vitest";
import golden from "./golden.json";
import { emitGraph } from "./codegen";
import { explore, paretoFront } from "./dse";
import { DEVICES, type Knobs } from "./model";
import { EXAMPLES } from "./examples";

const knobs = (unroll: number): Knobs => ({ unroll, tile: 0, pipeline: true });

describe("emitted HLS C++ parity with Python", () => {
  for (const row of golden.codegen) {
    it(`${row.name} @ unroll=${row.unroll}`, () => {
      const graph = EXAMPLES[row.name as keyof typeof EXAMPLES].build();
      for (const node of graph.nodes) node.knobs = knobs(row.unroll);
      expect(emitGraph(graph, "net")).toBe(row.source);
    });
  }
});

describe("DSE parity with Python", () => {
  for (const row of golden.dse) {
    it(`${row.name} on ${row.device}`, () => {
      const graph = EXAMPLES[row.name as keyof typeof EXAMPLES].build();
      const front = explore(graph, DEVICES[row.device]);
      expect(front.length).toBe(row.front.length);
      front.forEach((pt, i) => {
        const want = row.front[i];
        expect(pt.config.map((c) => c.unroll)).toEqual(want.unrolls);
        expect(pt.latency).toBe(want.latency);
        expect(pt.dsp).toBe(want.dsp);
        expect(pt.bram).toBe(want.bram);
        expect(pt.lut).toBe(want.lut);
        expect(pt.ff).toBe(want.ff);
        expect(pt.utilisation).toBeCloseTo(want.utilisation, 12);
      });
    });
  }
});

describe("DSE never exceeds the device budget", () => {
  for (const key of Object.keys(EXAMPLES)) {
    for (const devName of ["zynq-7020", "ultra96"]) {
      it(`${key} on ${devName}`, () => {
        const device = DEVICES[devName];
        const front = explore(EXAMPLES[key as keyof typeof EXAMPLES].build(), device);
        expect(front.length).toBeGreaterThan(0);
        for (const pt of front) {
          expect(pt.lut).toBeLessThanOrEqual(device.lut);
          expect(pt.ff).toBeLessThanOrEqual(device.ff);
          expect(pt.dsp).toBeLessThanOrEqual(device.dsp);
          expect(pt.bram).toBeLessThanOrEqual(device.bram18);
          expect(pt.fits).toBe(true);
        }
      });
    }
  }
});

describe("paretoFront drops dominated points", () => {
  it("keeps only non-dominated entries", () => {
    const pts = [
      { latency: 100, utilisation: 0.5 },
      { latency: 200, utilisation: 0.9 }, // dominated by the first
      { latency: 50, utilisation: 0.8 },
    ].map((p) => ({ ...p, config: [], dsp: 0, bram: 0, lut: 0, ff: 0, fits: true }));
    const front = paretoFront(pts);
    expect(front.map((p) => p.latency).sort((a, b) => a - b)).toEqual([50, 100]);
  });
});
