/**
 * TypeScript port of hls_estimate/model.py.
 *
 * Python is the source of truth; lib/model.test.ts asserts exact parity against
 * golden vectors generated from it. Keep the formulas here byte-for-byte equivalent
 * — every constant below is documented in MODEL.md.
 */

export const MACS_PER_DSP: Record<number, number> = { 2: 4, 4: 2, 8: 1, 16: 1 };

const BRAM18_WIDTH = 18;

const LUT_BASE = 120;
const LUT_PER_LANE = 2.75;
const LUT_REQUANT = 90;
const FF_BASE = 100;
const FF_PER_LANE = 0.69;
const FF_REQUANT = 64;
const ELTWISE_LUT_PER_LANE = 8;
const ELTWISE_FF_PER_LANE = 8;

const PIPELINE_DEPTH = 8;
const II_SEQ = 4;

export type Knobs = { unroll: number; tile: number; pipeline: boolean };

export const defaultKnobs = (): Knobs => ({ unroll: 1, tile: 0, pipeline: true });

export type Conv2dNode = {
  kind: "conv2d";
  name: string;
  isMac: true;
  inCh: number;
  outCh: number;
  kh: number;
  kw: number;
  inH: number;
  inW: number;
  stride: number;
  pad: number;
  wBits: number;
  aBits: number;
  outBits: number;
  bias: boolean;
  relu: boolean;
  mult: number;
  shift: number;
  knobs: Knobs;
  inputs: string[];
  output: string;
};

export type LinearNode = {
  kind: "linear";
  name: string;
  isMac: true;
  inFeatures: number;
  outFeatures: number;
  wBits: number;
  aBits: number;
  outBits: number;
  bias: boolean;
  relu: boolean;
  mult: number;
  shift: number;
  knobs: Knobs;
  inputs: string[];
  output: string;
};

export type ReLUNode = {
  kind: "relu";
  name: string;
  isMac: false;
  numel: number;
  bits: number;
  knobs: Knobs;
  inputs: string[];
  output: string;
};

export type MaxPoolNode = {
  kind: "maxpool2d";
  name: string;
  isMac: false;
  inCh: number;
  inH: number;
  inW: number;
  kh: number;
  kw: number;
  stride: number;
  bits: number;
  knobs: Knobs;
  inputs: string[];
  output: string;
};

export type AddNode = {
  kind: "add";
  name: string;
  isMac: false;
  numel: number;
  bits: number;
  knobs: Knobs;
  inputs: string[];
  output: string;
};

export type Node = Conv2dNode | LinearNode | ReLUNode | MaxPoolNode | AddNode;

export type Graph = { nodes: Node[]; inputShape: number[] };

// --- factories (snake_case input keys mirror the Python dataclasses) --------
type Spec = Record<string, number>;

export function makeConv2d(s: Spec, name = "conv", knobs = defaultKnobs()): Conv2dNode {
  return {
    kind: "conv2d",
    name,
    isMac: true,
    inCh: s.in_ch ?? s.inCh,
    outCh: s.out_ch ?? s.outCh,
    kh: s.kh,
    kw: s.kw,
    inH: s.in_h ?? s.inH,
    inW: s.in_w ?? s.inW,
    stride: s.stride ?? 1,
    pad: s.pad ?? 0,
    wBits: s.w_bits ?? s.wBits ?? 8,
    aBits: s.a_bits ?? s.aBits ?? 8,
    outBits: s.out_bits ?? s.outBits ?? 8,
    bias: Boolean(s.bias ?? 0),
    relu: Boolean(s.relu ?? 0),
    mult: s.mult ?? 1,
    shift: s.shift ?? 4,
    knobs,
    inputs: ["x"],
    output: name,
  };
}

export function makeLinear(s: Spec, name = "fc", knobs = defaultKnobs()): LinearNode {
  return {
    kind: "linear",
    name,
    isMac: true,
    inFeatures: s.in_features ?? s.inFeatures,
    outFeatures: s.out_features ?? s.outFeatures,
    wBits: s.w_bits ?? s.wBits ?? 8,
    aBits: s.a_bits ?? s.aBits ?? 8,
    outBits: s.out_bits ?? s.outBits ?? 8,
    bias: Boolean(s.bias ?? 0),
    relu: Boolean(s.relu ?? 0),
    mult: s.mult ?? 1,
    shift: s.shift ?? 4,
    knobs,
    inputs: ["x"],
    output: name,
  };
}

export function makeReLU(s: Spec, name = "relu", knobs = defaultKnobs()): ReLUNode {
  return {
    kind: "relu",
    name,
    isMac: false,
    numel: s.numel,
    bits: s.bits ?? 8,
    knobs,
    inputs: ["x"],
    output: name,
  };
}

export function makeMaxPool2d(s: Spec, name = "pool", knobs = defaultKnobs()): MaxPoolNode {
  return {
    kind: "maxpool2d",
    name,
    isMac: false,
    inCh: s.in_ch ?? s.inCh,
    inH: s.in_h ?? s.inH,
    inW: s.in_w ?? s.inW,
    kh: s.kh ?? 2,
    kw: s.kw ?? 2,
    stride: s.stride ?? 2,
    bits: s.bits ?? 8,
    knobs,
    inputs: ["x"],
    output: name,
  };
}

export function makeAdd(s: Spec, name = "add", knobs = defaultKnobs()): AddNode {
  return {
    kind: "add",
    name,
    isMac: false,
    numel: s.numel,
    bits: s.bits ?? 8,
    knobs,
    inputs: ["a", "b"],
    output: name,
  };
}

// --- shape / work ----------------------------------------------------------
export function convOutH(n: Conv2dNode): number {
  return Math.floor((n.inH + 2 * n.pad - n.kh) / n.stride) + 1;
}
export function convOutW(n: Conv2dNode): number {
  return Math.floor((n.inW + 2 * n.pad - n.kw) / n.stride) + 1;
}
export function poolOutH(n: MaxPoolNode): number {
  return Math.floor((n.inH - n.kh) / n.stride) + 1;
}
export function poolOutW(n: MaxPoolNode): number {
  return Math.floor((n.inW - n.kw) / n.stride) + 1;
}

export function macs(n: Node): number {
  if (n.kind === "conv2d") {
    return n.outCh * convOutH(n) * convOutW(n) * n.inCh * n.kh * n.kw;
  }
  if (n.kind === "linear") return n.outFeatures * n.inFeatures;
  return 0;
}

export function work(n: Node): number {
  switch (n.kind) {
    case "conv2d":
    case "linear":
      return macs(n);
    case "maxpool2d":
      return n.inCh * poolOutH(n) * poolOutW(n) * n.kh * n.kw;
    default:
      return n.numel;
  }
}

export function outputNumel(n: Node): number {
  switch (n.kind) {
    case "conv2d":
      return n.outCh * convOutH(n) * convOutW(n);
    case "linear":
      return n.outFeatures;
    case "maxpool2d":
      return n.inCh * poolOutH(n) * poolOutW(n);
    default:
      return n.numel;
  }
}

// --- resource model --------------------------------------------------------
export function macsPerDsp(bits: number): number {
  const m = MACS_PER_DSP[bits];
  if (!m) throw new Error(`no packing model for ${bits}-bit`);
  return m;
}

export function dsp(n: Node, k: Knobs): number {
  if (!n.isMac) return 0;
  return Math.ceil(Math.max(1, k.unroll) / macsPerDsp(n.wBits));
}

export function bram18(depth: number, widthBits: number): number {
  if (depth <= 0 || widthBits <= 0) return 0;
  return Math.ceil(widthBits / BRAM18_WIDTH) * Math.ceil(depth / 1024);
}

function bufferBram(numElems: number, bits: number, partitions: number): number {
  if (numElems <= 0) return 0;
  const p = Math.max(1, Math.min(partitions, numElems));
  return p * bram18(Math.ceil(numElems / p), bits);
}

export function bram(n: Node, k: Knobs): number {
  const u = Math.max(1, k.unroll);
  if (n.kind === "conv2d") {
    const w = bufferBram(n.outCh * n.inCh * n.kh * n.kw, n.wBits, u);
    const line = bufferBram(n.inCh * n.inW * n.kh, n.aBits, 1);
    const tile = k.tile > 0 ? k.tile : n.outCh * convOutH(n) * convOutW(n);
    return w + line + bufferBram(tile, n.outBits, u);
  }
  if (n.kind === "linear") {
    const w = bufferBram(n.outFeatures * n.inFeatures, n.wBits, u);
    const inBuf = bufferBram(n.inFeatures, n.aBits, 1);
    const tile = k.tile > 0 ? k.tile : n.outFeatures;
    return w + inBuf + bufferBram(tile, n.outBits, u);
  }
  if (n.kind === "maxpool2d") {
    return bufferBram(n.inCh * n.inW * n.kh, n.bits, 1);
  }
  return 0;
}

export function lut(n: Node, k: Knobs): number {
  const u = Math.max(1, k.unroll);
  if (n.isMac) {
    return Math.trunc(LUT_BASE + LUT_PER_LANE * u * n.wBits + LUT_REQUANT);
  }
  return Math.trunc(LUT_BASE + ELTWISE_LUT_PER_LANE * u);
}

export function ff(n: Node, k: Knobs): number {
  const u = Math.max(1, k.unroll);
  if (n.isMac) {
    return Math.trunc(FF_BASE + FF_PER_LANE * u * (n.wBits + n.aBits) + FF_REQUANT);
  }
  return Math.trunc(FF_BASE + ELTWISE_FF_PER_LANE * u);
}

export function latencyCycles(n: Node, k: Knobs): number {
  const u = Math.max(1, k.unroll);
  const ii = k.pipeline ? 1 : II_SEQ;
  return Math.ceil(work(n) / u) * ii + PIPELINE_DEPTH;
}

export type LayerEstimate = {
  name: string;
  kind: string;
  macs: number;
  dsp: number;
  bram: number;
  lut: number;
  ff: number;
  latency: number;
};

export function resources(n: Node, k: Knobs) {
  return {
    dsp: dsp(n, k),
    bram: bram(n, k),
    lut: lut(n, k),
    ff: ff(n, k),
    latency: latencyCycles(n, k),
  };
}

export type GraphEstimate = {
  perLayer: LayerEstimate[];
  totals: Record<string, number>;
  bottleneck: string;
};

export function estimateGraph(graph: Graph): GraphEstimate {
  const perLayer: LayerEstimate[] = graph.nodes.map((n) => ({
    name: n.name,
    kind: n.kind,
    macs: macs(n),
    ...resources(n, n.knobs),
  }));
  const sum = (key: keyof LayerEstimate) =>
    perLayer.reduce((acc, l) => acc + (l[key] as number), 0);
  const totals: Record<string, number> = {
    macs: sum("macs"),
    dsp: sum("dsp"),
    bram: sum("bram"),
    lut: sum("lut"),
    ff: sum("ff"),
    latency: perLayer.length ? Math.max(...perLayer.map((l) => l.latency)) : 0,
    latency_seq: sum("latency"),
  };
  totals.logic = totals.lut + totals.ff;
  const bottleneck = perLayer.length
    ? perLayer.reduce((a, b) => (b.latency > a.latency ? b : a)).name
    : "";
  return { perLayer, totals, bottleneck };
}

// --- devices ---------------------------------------------------------------
export type Device = {
  name: string;
  label: string;
  lut: number;
  ff: number;
  dsp: number;
  bram18: number;
};

export const DEVICES: Record<string, Device> = {
  "zynq-7020": {
    name: "zynq-7020",
    label: "Zynq-7020 (XC7Z020)",
    lut: 53200,
    ff: 106400,
    dsp: 220,
    bram18: 280,
  },
  ultra96: {
    name: "ultra96",
    label: "Ultra96 (ZU3EG)",
    lut: 70560,
    ff: 141120,
    dsp: 360,
    bram18: 432,
  },
  "pynq-z2": {
    name: "pynq-z2",
    label: "PYNQ-Z2 (XC7Z020)",
    lut: 53200,
    ff: 106400,
    dsp: 220,
    bram18: 280,
  },
};

export const RESOURCE_KEYS = ["lut", "ff", "dsp", "bram"] as const;
export type ResourceKey = (typeof RESOURCE_KEYS)[number];

export function capacityOf(device: Device, res: ResourceKey): number {
  return res === "bram" ? device.bram18 : device[res];
}

export function utilisation(totals: Record<string, number>, device: Device) {
  const out = {} as Record<ResourceKey, number>;
  for (const res of RESOURCE_KEYS) out[res] = totals[res] / capacityOf(device, res);
  return out;
}

export function fitsDevice(totals: Record<string, number>, device: Device): boolean {
  return RESOURCE_KEYS.every((res) => totals[res] <= capacityOf(device, res));
}

export function bindingResource(
  totals: Record<string, number>,
  device: Device,
): ResourceKey {
  const u = utilisation(totals, device);
  return RESOURCE_KEYS.reduce((a, b) => (u[b] > u[a] ? b : a));
}
