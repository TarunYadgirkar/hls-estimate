"""Internal dataflow IR.

Nodes carry shapes, quantization bit-widths, tunable knobs, and (for codegen) the
concrete integer weights/requant params. Resource/latency modelling only needs the
shape + bits + knobs; codegen also needs the weight data and tensor connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_BITS = (2, 4, 8, 16)


@dataclass
class Knobs:
    """Per-layer tuning knobs (the DSE search axes)."""
    unroll: int = 1          # number of parallel MAC (or elementwise) lanes
    tile: int = 0            # output tile depth; 0 == whole layer
    pipeline: bool = True    # II=1 pipelined loop vs sequential

    def __post_init__(self):
        assert self.unroll >= 1, "unroll must be >= 1"
        assert self.tile >= 0, "tile must be >= 0"


@dataclass
class TensorSpec:
    shape: tuple[int, ...]
    bits: int = 8
    signed: bool = True
    name: str = "x"

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class Conv2d:
    in_ch: int
    out_ch: int
    kh: int
    kw: int
    in_h: int
    in_w: int
    stride: int = 1
    pad: int = 0
    w_bits: int = 8
    a_bits: int = 8
    out_bits: int = 8
    bias: bool = True
    relu: bool = False
    mult: int = 1
    shift: int = 0
    name: str = "conv"
    inputs: tuple[str, ...] = ("x",)
    output: str = "y"
    knobs: Knobs = field(default_factory=Knobs)
    weight: Any = None       # int array [out_ch, in_ch, kh, kw]
    bias_data: Any = None    # int array [out_ch]

    kind = "conv2d"
    is_mac = True

    def __post_init__(self):
        assert self.w_bits in SUPPORTED_BITS, f"unsupported w_bits {self.w_bits}"

    def out_h(self) -> int:
        return (self.in_h + 2 * self.pad - self.kh) // self.stride + 1

    def out_w(self) -> int:
        return (self.in_w + 2 * self.pad - self.kw) // self.stride + 1

    def macs(self) -> int:
        return self.out_ch * self.out_h() * self.out_w() * self.in_ch * self.kh * self.kw

    def work(self) -> int:
        return self.macs()

    def out_spec(self) -> TensorSpec:
        return TensorSpec((1, self.out_ch, self.out_h(), self.out_w()),
                          bits=self.out_bits, name=self.output)


@dataclass
class Linear:
    in_features: int
    out_features: int
    w_bits: int = 8
    a_bits: int = 8
    out_bits: int = 8
    bias: bool = True
    relu: bool = False
    mult: int = 1
    shift: int = 0
    name: str = "fc"
    inputs: tuple[str, ...] = ("x",)
    output: str = "y"
    knobs: Knobs = field(default_factory=Knobs)
    weight: Any = None       # int array [out_features, in_features]
    bias_data: Any = None    # int array [out_features]

    kind = "linear"
    is_mac = True

    def __post_init__(self):
        assert self.w_bits in SUPPORTED_BITS, f"unsupported w_bits {self.w_bits}"

    def macs(self) -> int:
        return self.out_features * self.in_features

    def work(self) -> int:
        return self.macs()

    def out_spec(self) -> TensorSpec:
        return TensorSpec((1, self.out_features), bits=self.out_bits, name=self.output)


@dataclass
class ReLU:
    numel: int
    bits: int = 8
    shape: tuple[int, ...] = ()
    name: str = "relu"
    inputs: tuple[str, ...] = ("x",)
    output: str = "y"
    knobs: Knobs = field(default_factory=Knobs)

    kind = "relu"
    is_mac = False
    w_bits = 0

    def macs(self) -> int:
        return 0

    def work(self) -> int:
        return self.numel

    def out_spec(self) -> TensorSpec:
        return TensorSpec(self.shape or (self.numel,), bits=self.bits, name=self.output)


@dataclass
class MaxPool2d:
    in_ch: int
    in_h: int
    in_w: int
    kh: int = 2
    kw: int = 2
    stride: int = 2
    bits: int = 8
    name: str = "pool"
    inputs: tuple[str, ...] = ("x",)
    output: str = "y"
    knobs: Knobs = field(default_factory=Knobs)

    kind = "maxpool2d"
    is_mac = False
    w_bits = 0

    def out_h(self) -> int:
        return (self.in_h - self.kh) // self.stride + 1

    def out_w(self) -> int:
        return (self.in_w - self.kw) // self.stride + 1

    def macs(self) -> int:
        return 0

    def work(self) -> int:
        return self.in_ch * self.out_h() * self.out_w() * self.kh * self.kw

    def out_spec(self) -> TensorSpec:
        return TensorSpec((1, self.in_ch, self.out_h(), self.out_w()),
                          bits=self.bits, name=self.output)


@dataclass
class Add:
    numel: int
    bits: int = 8
    shape: tuple[int, ...] = ()
    name: str = "add"
    inputs: tuple[str, ...] = ("a", "b")
    output: str = "y"
    knobs: Knobs = field(default_factory=Knobs)

    kind = "add"
    is_mac = False
    w_bits = 0

    def macs(self) -> int:
        return 0

    def work(self) -> int:
        return self.numel

    def out_spec(self) -> TensorSpec:
        return TensorSpec(self.shape or (self.numel,), bits=self.bits, name=self.output)


Node = Any  # any of the above


@dataclass
class Graph:
    nodes: list
    input_spec: TensorSpec
    output_spec: TensorSpec

    def tensors(self) -> dict[str, TensorSpec]:
        t = {self.input_spec.name: self.input_spec}
        for n in self.nodes:
            t[n.output] = n.out_spec()
        return t
