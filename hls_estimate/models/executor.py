"""Golden reference: execute an IR Graph with exact integer arithmetic in PyTorch.

This is the "PyTorch model" the emitted C++ is checked against. It is written in
idiomatic torch (int64 tensors, broadcasting, reductions) and is a completely separate
implementation from the C++ emitter, so bit-exact agreement is a real cross-check of
the op semantics (accumulation, requant rounding, clamping).
"""
from __future__ import annotations

import torch

from ..ir import Conv2d, Linear, ReLU, MaxPool2d, Add


def qrange(bits: int, relu: bool = False):
    qmin = 0 if relu else -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    return qmin, qmax


def requant(acc: torch.Tensor, mult: int, shift: int, out_bits: int, relu: bool):
    rnd = (1 << (shift - 1)) if shift > 0 else 0
    t = acc * mult + rnd
    y = torch.div(t, 1 << shift, rounding_mode="floor")  # arithmetic (floor) shift
    qmin, qmax = qrange(out_bits, relu)
    return torch.clamp(y, qmin, qmax)


def _pad2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return x
    n, c, h, w = x.shape
    out = torch.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=x.dtype)
    out[:, :, pad:pad + h, pad:pad + w] = x
    return out


def _im2col(x: torch.Tensor, kh: int, kw: int, stride: int) -> torch.Tensor:
    # x: [N,C,H,W] -> patches [N,C,OH,OW,kh,kw] (unfold is a dtype-agnostic view)
    return x.unfold(2, kh, stride).unfold(3, kw, stride)


def _run_conv(n: Conv2d, x: torch.Tensor) -> torch.Tensor:
    w = torch.as_tensor(n.weight, dtype=torch.int64)          # [O,C,kh,kw]
    xp = _pad2d(x, n.pad)
    patches = _im2col(xp, n.kh, n.kw, n.stride)               # [N,C,OH,OW,kh,kw]
    prod = patches.unsqueeze(1) * w.view(1, n.out_ch, n.in_ch, 1, 1, n.kh, n.kw)
    acc = prod.sum(dim=(2, 5, 6))                             # [N,O,OH,OW]
    if n.bias and n.bias_data is not None:
        acc = acc + torch.as_tensor(n.bias_data, dtype=torch.int64).view(1, -1, 1, 1)
    return requant(acc, n.mult, n.shift, n.out_bits, n.relu)


def _run_linear(n: Linear, x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(1, -1)
    w = torch.as_tensor(n.weight, dtype=torch.int64)          # [O,F]
    acc = (x.view(1, 1, -1) * w.view(1, n.out_features, n.in_features)).sum(dim=2)
    if n.bias and n.bias_data is not None:
        acc = acc + torch.as_tensor(n.bias_data, dtype=torch.int64).view(1, -1)
    return requant(acc, n.mult, n.shift, n.out_bits, n.relu)


def _run_node(node, ins: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(node, Conv2d):
        return _run_conv(node, ins[0])
    if isinstance(node, Linear):
        return _run_linear(node, ins[0])
    if isinstance(node, ReLU):
        qmin, qmax = qrange(node.bits, relu=True)
        return torch.clamp(ins[0], qmin, qmax)
    if isinstance(node, MaxPool2d):
        patches = _im2col(ins[0], node.kh, node.kw, node.stride)
        return patches.amax(dim=(4, 5))
    if isinstance(node, Add):
        qmin, qmax = qrange(node.bits)
        return torch.clamp(ins[0] + ins[1], qmin, qmax)
    raise TypeError(f"unsupported node {type(node).__name__}")


def run_graph(graph, x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.int64)
    env = {graph.input_spec.name: x}
    for node in graph.nodes:
        ins = [env[name] for name in node.inputs]
        env[node.output] = _run_node(node, ins)
    return env[graph.output_spec.name]


class TorchGraphModule(torch.nn.Module):
    """Thin nn.Module wrapper so tests can call `module(tensor)`."""

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def forward(self, x):
        return run_graph(self.graph, x)
