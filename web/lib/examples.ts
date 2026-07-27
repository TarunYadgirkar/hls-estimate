/**
 * Example networks, mirroring hls_estimate/models/__init__.py exactly — including
 * tensor wiring, bias/ReLU fusion and requantization shifts. Parity with the Python
 * model and emitter is enforced by lib/model.test.ts and lib/codegen.test.ts.
 */
import {
  makeAdd,
  makeConv2d,
  makeLinear,
  makeMaxPool2d,
  makeReLU,
  type Graph,
  type Knobs,
  type Node,
} from "./model";

const k = (): Knobs => ({ unroll: 1, tile: 0, pipeline: true });

export type Example = {
  key: string;
  label: string;
  blurb: string;
  inputShape: number[];
  build: () => Graph;
};

function wire<T extends Node>(node: T, inputs: string[], output: string): T {
  node.inputs = inputs;
  node.output = output;
  return node;
}

export const EXAMPLES: Record<string, Example> = {
  tiny_conv: {
    key: "tiny_conv",
    label: "tiny_conv",
    blurb:
      "One 3×3 convolution, 2→3 channels over a 6×6 input, with ReLU fused into the requantizer. The smallest thing worth estimating.",
    inputShape: [1, 2, 6, 6],
    build: () => ({
      nodes: [
        wire(
          makeConv2d(
            {
              in_ch: 2, out_ch: 3, kh: 3, kw: 3, in_h: 6, in_w: 6,
              pad: 0, stride: 1, w_bits: 8, relu: 1, shift: 4,
            },
            "conv",
            k(),
          ),
          ["x"],
          "y",
        ),
      ],
      inputShape: [1, 2, 6, 6],
    }),
  },
  conv_relu_pool: {
    key: "conv_relu_pool",
    label: "conv_relu_pool",
    blurb:
      "The classic CNN block: padded 3×3 conv, then ReLU, then 2×2 max pool. Three dataflow stages that run concurrently.",
    inputShape: [1, 3, 8, 8],
    build: () => ({
      nodes: [
        wire(
          makeConv2d(
            {
              in_ch: 3, out_ch: 4, kh: 3, kw: 3, in_h: 8, in_w: 8,
              pad: 1, stride: 1, w_bits: 8, shift: 4,
            },
            "conv",
            k(),
          ),
          ["x"],
          "c",
        ),
        wire(makeReLU({ numel: 4 * 8 * 8, bits: 8 }, "relu", k()), ["c"], "r"),
        wire(
          makeMaxPool2d(
            { in_ch: 4, in_h: 8, in_w: 8, kh: 2, kw: 2, stride: 2, bits: 8 },
            "pool",
            k(),
          ),
          ["r"],
          "y",
        ),
      ],
      inputShape: [1, 3, 8, 8],
    }),
  },
  tiny_linear: {
    key: "tiny_linear",
    label: "tiny_linear",
    blurb:
      "A single fully-connected layer, 8→6, with bias. The simplest possible MAC array.",
    inputShape: [1, 8],
    build: () => ({
      nodes: [
        wire(
          makeLinear(
            { in_features: 8, out_features: 6, w_bits: 8, bias: 1, shift: 3 },
            "fc",
            k(),
          ),
          ["x"],
          "y",
        ),
      ],
      inputShape: [1, 8],
    }),
  },
  residual: {
    key: "residual",
    label: "residual",
    blurb:
      "A skip connection: a 3×3 conv added back to its own input, ResNet-style. The add reads two buffers at once.",
    inputShape: [1, 4, 5, 5],
    build: () => ({
      nodes: [
        wire(
          makeConv2d(
            {
              in_ch: 4, out_ch: 4, kh: 3, kw: 3, in_h: 5, in_w: 5,
              pad: 1, stride: 1, w_bits: 8, shift: 4,
            },
            "conv",
            k(),
          ),
          ["x"],
          "c1",
        ),
        wire(makeAdd({ numel: 4 * 5 * 5, bits: 8 }, "add", k()), ["c1", "x"], "y"),
      ],
      inputShape: [1, 4, 5, 5],
    }),
  },
  mlp_int4: {
    key: "mlp_int4",
    label: "mlp_int4",
    blurb:
      "Two int4 dense layers. At the same parallelism this uses half the DSPs of the int8 equivalent — that is the packing model, visible.",
    inputShape: [1, 16],
    build: () => ({
      nodes: [
        wire(
          makeLinear(
            { in_features: 16, out_features: 12, w_bits: 4, relu: 1, shift: 3 },
            "fc1",
            k(),
          ),
          ["x"],
          "h",
        ),
        wire(
          makeLinear(
            { in_features: 12, out_features: 4, w_bits: 4, shift: 3 },
            "fc2",
            k(),
          ),
          ["h"],
          "y",
        ),
      ],
      inputShape: [1, 16],
    }),
  },
};

export const EXAMPLE_LIST = Object.values(EXAMPLES);
