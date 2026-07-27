/**
 * TypeScript port of hls_estimate/dse.py. Parity-checked in lib/codegen.test.ts.
 */
import {
  bram,
  dsp,
  ff,
  latencyCycles,
  lut,
  type Device,
  type Graph,
  type Knobs,
  type Node,
} from "./model";

const MAX_CONFIGS = 20000;

export type DesignPoint = {
  config: Knobs[];
  latency: number;
  lut: number;
  ff: number;
  dsp: number;
  bram: number;
  fits: boolean;
  utilisation: number;
};

function maxParallel(node: Node): number {
  if (node.kind === "conv2d") return node.inCh * node.kh * node.kw;
  if (node.kind === "linear") return node.inFeatures;
  if (node.kind === "maxpool2d") {
    return Math.max(1, node.inCh * node.kh * node.kw);
  }
  return Math.max(1, node.numel);
}

function pow2Upto(limit: number): number[] {
  const vals: number[] = [];
  let u = 1;
  const lim = Math.max(1, limit);
  while (u <= lim) {
    vals.push(u);
    u *= 2;
  }
  if (limit > vals[vals.length - 1]) vals.push(limit);
  return vals;
}

export function defaultKnobGrid(graph: Graph): Knobs[][] {
  return graph.nodes.map((node) =>
    pow2Upto(maxParallel(node)).map((unroll) => ({
      unroll,
      tile: 0,
      pipeline: true,
    })),
  );
}

export function evaluate(
  graph: Graph,
  config: Knobs[],
  device?: Device,
): DesignPoint {
  const tot = { lut: 0, ff: 0, dsp: 0, bram: 0 };
  const lat: number[] = [];
  graph.nodes.forEach((node, i) => {
    const k = config[i];
    tot.lut += lut(node, k);
    tot.ff += ff(node, k);
    tot.dsp += dsp(node, k);
    tot.bram += bram(node, k);
    lat.push(latencyCycles(node, k));
  });
  const latency = lat.length ? Math.max(...lat) : 0;
  let fits = true;
  let util = 0;
  if (device) {
    fits =
      tot.lut <= device.lut &&
      tot.ff <= device.ff &&
      tot.dsp <= device.dsp &&
      tot.bram <= device.bram18;
    util = Math.max(
      tot.lut / device.lut,
      tot.ff / device.ff,
      tot.dsp / device.dsp,
      tot.bram / device.bram18,
    );
  }
  return { config: [...config], latency, ...tot, fits, utilisation: util };
}

type Comparable = { latency: number; utilisation: number; dsp: number };

export function paretoFront<T extends Comparable>(points: T[]): T[] {
  const front = points.filter(
    (p) =>
      !points.some(
        (q) =>
          q !== p &&
          q.latency <= p.latency &&
          q.utilisation <= p.utilisation &&
          (q.latency < p.latency || q.utilisation < p.utilisation),
      ),
  );
  // De-duplicate identical (latency, utilisation) pairs, keeping the cheapest DSP.
  const best = new Map<string, T>();
  for (const p of front) {
    const key = `${p.latency}|${p.utilisation}`;
    const cur = best.get(key);
    if (!cur || p.dsp < cur.dsp) best.set(key, p);
  }
  return [...best.values()].sort(
    (a, b) => a.latency - b.latency || a.utilisation - b.utilisation,
  );
}

function* product(grid: Knobs[][]): Generator<Knobs[]> {
  const idx = new Array(grid.length).fill(0);
  if (grid.length === 0) return;
  for (;;) {
    yield grid.map((opts, i) => opts[idx[i]]);
    let d = grid.length - 1;
    while (d >= 0) {
      idx[d] += 1;
      if (idx[d] < grid[d].length) break;
      idx[d] = 0;
      d -= 1;
    }
    if (d < 0) return;
  }
}

export function explore(
  graph: Graph,
  device: Device,
  knobGrid?: Knobs[][],
): DesignPoint[] {
  const grid = knobGrid ?? defaultKnobGrid(graph);
  const feasible: DesignPoint[] = [];
  let i = 0;
  for (const config of product(grid)) {
    if (i++ >= MAX_CONFIGS) break;
    const point = evaluate(graph, config, device);
    if (point.fits) feasible.push(point);
  }
  return feasible.length ? paretoFront(feasible) : [];
}
