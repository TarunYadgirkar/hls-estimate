"""torch.fx -> internal IR.

Traces a plain `nn.Module`, folds batchnorm into the preceding conv/linear, quantizes
weights per-tensor symmetrically, and emits IR nodes with shapes resolved by shape
propagation over the trace.
"""
from __future__ import annotations

import operator

import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from ..ir import Add, Conv2d, Graph, Knobs, Linear, MaxPool2d, ReLU, TensorSpec
from .quant import quantize_tensor


class UnsupportedOp(Exception):
    """Raised when the traced graph contains an op outside SPEC §1."""


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Fold a BatchNorm2d into the preceding conv. Returns (weight, bias)."""
    w = conv.weight.detach()
    b = conv.bias.detach() if conv.bias is not None else torch.zeros(w.shape[0])
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    folded_w = w * scale.reshape(-1, 1, 1, 1)
    folded_b = (b - bn.running_mean.detach()) * scale + bn.bias.detach()
    return folded_w, folded_b


def _fold_linear_bn(lin: nn.Linear, bn: nn.BatchNorm1d):
    w = lin.weight.detach()
    b = lin.bias.detach() if lin.bias is not None else torch.zeros(w.shape[0])
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    return w * scale.reshape(-1, 1), (b - bn.running_mean.detach()) * scale + bn.bias.detach()


def _is_relu(mod, node) -> bool:
    if isinstance(mod, nn.ReLU):
        return True
    return node.op == "call_function" and node.target in (torch.relu, torch.nn.functional.relu)


class _Lowering:
    """Walks the traced graph and builds IR nodes."""

    def __init__(self, traced, module, input_shape, w_bits, a_bits, shift):
        self.traced = traced
        self.modules = dict(module.named_modules())
        self.input_shape = tuple(input_shape)
        self.w_bits, self.a_bits, self.shift = w_bits, a_bits, shift
        self.nodes: list = []
        self.shapes: dict[str, tuple] = {}
        self.names: dict[str, str] = {}   # fx node name -> IR tensor name
        self.counter = 0

    def _fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def _mod(self, node):
        return self.modules.get(node.target) if node.op == "call_module" else None

    def _quantize_weights(self, w, b):
        wq, _ = quantize_tensor(w, self.w_bits)
        bq = None
        if b is not None:
            # Bias lives in the accumulator domain; keep the wider activation width.
            bq, _ = quantize_tensor(b, min(16, self.a_bits + 8))
        return wq, bq

    # -- op handlers --------------------------------------------------------
    def _add_conv(self, conv: nn.Conv2d, in_shape, weight, bias, out_name, in_name):
        if conv.groups != 1:
            raise UnsupportedOp("grouped/depthwise conv (groups != 1) is not supported")
        if conv.dilation not in ((1, 1), 1):
            raise UnsupportedOp("dilated conv is not supported")
        wq, bq = self._quantize_weights(weight, bias)
        node = Conv2d(
            in_ch=conv.in_channels, out_ch=conv.out_channels,
            kh=conv.kernel_size[0], kw=conv.kernel_size[1],
            in_h=in_shape[2], in_w=in_shape[3],
            stride=conv.stride[0], pad=conv.padding[0],
            w_bits=self.w_bits, a_bits=self.a_bits, out_bits=self.a_bits,
            bias=bias is not None, relu=False, mult=1, shift=self.shift,
            name=self._fresh("conv"), inputs=(in_name,), output=out_name,
            weight=wq, bias_data=bq, knobs=Knobs())
        self.nodes.append(node)
        return node

    def _add_linear(self, lin: nn.Linear, weight, bias, out_name, in_name):
        wq, bq = self._quantize_weights(weight, bias)
        node = Linear(
            in_features=lin.in_features, out_features=lin.out_features,
            w_bits=self.w_bits, a_bits=self.a_bits, out_bits=self.a_bits,
            bias=bias is not None, relu=False, mult=1, shift=self.shift,
            name=self._fresh("fc"), inputs=(in_name,), output=out_name,
            weight=wq, bias_data=bq, knobs=Knobs())
        self.nodes.append(node)
        return node

    # -- main walk ----------------------------------------------------------
    def run(self) -> Graph:
        fx_nodes = list(self.traced.graph.nodes)
        skip: set = set()
        input_name = "x"
        last_ir_name = None

        for idx, fxn in enumerate(fx_nodes):
            if fxn in skip:
                continue

            if fxn.op == "placeholder":
                self.names[fxn.name] = input_name
                self.shapes[input_name] = self.input_shape
                last_ir_name = input_name
                continue

            if fxn.op == "output":
                continue

            in_names = [self.names[a.name] for a in fxn.all_input_nodes]
            src = in_names[0] if in_names else input_name
            in_shape = self.shapes[src]
            out_name = self._fresh("t")
            mod = self._mod(fxn)

            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                weight = mod.weight.detach()
                bias = mod.bias.detach() if mod.bias is not None else None
                # Fold an immediately-following batchnorm.
                nxt = fx_nodes[idx + 1] if idx + 1 < len(fx_nodes) else None
                nxt_mod = self._mod(nxt) if nxt is not None else None
                if isinstance(mod, nn.Conv2d) and isinstance(nxt_mod, nn.BatchNorm2d):
                    weight, bias = fold_conv_bn(mod, nxt_mod)
                    skip.add(nxt)
                    self.names[nxt.name] = out_name
                elif isinstance(mod, nn.Linear) and isinstance(nxt_mod, nn.BatchNorm1d):
                    weight, bias = _fold_linear_bn(mod, nxt_mod)
                    skip.add(nxt)
                    self.names[nxt.name] = out_name

                if isinstance(mod, nn.Conv2d):
                    node = self._add_conv(mod, in_shape, weight, bias, out_name, src)
                else:
                    node = self._add_linear(mod, weight, bias, out_name, src)

            elif isinstance(mod, nn.MaxPool2d) or (
                    fxn.op == "call_function" and fxn.target is torch.nn.functional.max_pool2d):
                kh = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
                kw = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[1]
                stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
                node = MaxPool2d(in_ch=in_shape[1], in_h=in_shape[2], in_w=in_shape[3],
                                 kh=kh, kw=kw, stride=stride, bits=self.a_bits,
                                 name=self._fresh("pool"), inputs=(src,), output=out_name,
                                 knobs=Knobs())
                self.nodes.append(node)

            elif _is_relu(mod, fxn):
                numel = 1
                for d in in_shape:
                    numel *= d
                node = ReLU(numel=numel, bits=self.a_bits, shape=in_shape,
                            name=self._fresh("relu"), inputs=(src,), output=out_name,
                            knobs=Knobs())
                self.nodes.append(node)

            elif fxn.op == "call_function" and fxn.target in (operator.add, torch.add):
                if len(in_names) != 2:
                    raise UnsupportedOp("add with a constant operand is not supported")
                numel = 1
                for d in in_shape:
                    numel *= d
                node = Add(numel=numel, bits=self.a_bits, shape=in_shape,
                           name=self._fresh("add"), inputs=tuple(in_names),
                           output=out_name, knobs=Knobs())
                self.nodes.append(node)

            elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                raise UnsupportedOp(
                    "standalone batchnorm: it must directly follow a conv/linear so it "
                    "can be folded (SPEC §1)")

            elif isinstance(mod, nn.Flatten) or (
                    fxn.op == "call_method" and fxn.target in ("view", "reshape", "flatten")):
                # Shape-only: pass the buffer through untouched.
                self.names[fxn.name] = src
                self.shapes[src] = in_shape
                last_ir_name = src
                continue

            else:
                what = mod if mod is not None else fxn.target
                raise UnsupportedOp(f"unsupported op in traced graph: {what}")

            self.names[fxn.name] = out_name
            self.shapes[out_name] = node.out_spec().shape
            last_ir_name = out_name

        if not self.nodes:
            raise UnsupportedOp("traced graph produced no supported layers")

        out_spec = self.nodes[-1].out_spec()
        return Graph(self.nodes,
                     TensorSpec(self.input_shape, bits=self.a_bits, name=input_name),
                     out_spec)


def from_torch(module: nn.Module, input_shape, w_bits: int = 8, a_bits: int = 8,
               shift: int = 8) -> Graph:
    """Trace `module` with torch.fx and lower it to the internal IR.

    Weights are quantized per-tensor symmetrically to `w_bits`. `shift` sets the
    fixed-point requantization shift applied after every MAC layer.
    """
    traced = symbolic_trace(module)
    return _Lowering(traced, module, input_shape, w_bits, a_bits, shift).run()
