"""Bundled example models for tests and demos.

Each factory returns `(module, graph)`:
- `graph`  : the internal IR (also carries the integer weights, for codegen).
- `module` : a `TorchGraphModule` executing that same IR in exact int64 arithmetic.

Weights are generated deterministically so runs are reproducible.
"""
from __future__ import annotations

import numpy as np

from ..ir import Conv2d, Linear, ReLU, MaxPool2d, Add, Graph, TensorSpec
from .executor import TorchGraphModule


def _w(rng, shape, lo=-3, hi=3):
    return rng.integers(lo, hi + 1, size=shape, dtype=np.int64)


def _wrap(graph):
    return TorchGraphModule(graph), graph


def tiny_conv():
    rng = np.random.default_rng(1)
    conv = Conv2d(in_ch=2, out_ch=3, kh=3, kw=3, in_h=6, in_w=6, pad=0, stride=1,
                  w_bits=8, a_bits=8, out_bits=8, bias=False, relu=True,
                  mult=1, shift=4, name="conv", inputs=("x",), output="y",
                  weight=_w(rng, (3, 2, 3, 3)))
    g = Graph([conv], TensorSpec((1, 2, 6, 6), name="x"), conv.out_spec())
    return _wrap(g)


def conv_relu_pool():
    rng = np.random.default_rng(2)
    conv = Conv2d(in_ch=3, out_ch=4, kh=3, kw=3, in_h=8, in_w=8, pad=1, stride=1,
                  w_bits=8, a_bits=8, out_bits=8, bias=False, relu=False,
                  mult=1, shift=4, name="conv", inputs=("x",), output="c",
                  weight=_w(rng, (4, 3, 3, 3)))
    relu = ReLU(numel=4 * 8 * 8, bits=8, shape=(1, 4, 8, 8),
                name="relu", inputs=("c",), output="r")
    pool = MaxPool2d(in_ch=4, in_h=8, in_w=8, kh=2, kw=2, stride=2, bits=8,
                     name="pool", inputs=("r",), output="y")
    g = Graph([conv, relu, pool], TensorSpec((1, 3, 8, 8), name="x"), pool.out_spec())
    return _wrap(g)


def tiny_linear():
    rng = np.random.default_rng(3)
    fc = Linear(in_features=8, out_features=6, w_bits=8, a_bits=8, out_bits=8,
                bias=True, relu=False, mult=1, shift=3, name="fc",
                inputs=("x",), output="y",
                weight=_w(rng, (6, 8)), bias_data=_w(rng, (6,), -2, 2))
    g = Graph([fc], TensorSpec((1, 8), name="x"), fc.out_spec())
    return _wrap(g)


def residual():
    rng = np.random.default_rng(4)
    conv = Conv2d(in_ch=4, out_ch=4, kh=3, kw=3, in_h=5, in_w=5, pad=1, stride=1,
                  w_bits=8, a_bits=8, out_bits=8, bias=False, relu=False,
                  mult=1, shift=4, name="conv", inputs=("x",), output="c1",
                  weight=_w(rng, (4, 4, 3, 3)))
    add = Add(numel=4 * 5 * 5, bits=8, shape=(1, 4, 5, 5),
              name="add", inputs=("c1", "x"), output="y")
    g = Graph([conv, add], TensorSpec((1, 4, 5, 5), name="x"), add.out_spec())
    return _wrap(g)


def mlp_int4():
    """A 2-layer int4 MLP (exercises int4 codegen + packing)."""
    rng = np.random.default_rng(5)
    f1 = Linear(in_features=16, out_features=12, w_bits=4, a_bits=8, out_bits=8,
                bias=False, relu=True, mult=1, shift=3, name="fc1",
                inputs=("x",), output="h", weight=_w(rng, (12, 16), -4, 3))
    f2 = Linear(in_features=12, out_features=4, w_bits=4, a_bits=8, out_bits=8,
                bias=False, relu=False, mult=1, shift=3, name="fc2",
                inputs=("h",), output="y", weight=_w(rng, (4, 12), -4, 3))
    g = Graph([f1, f2], TensorSpec((1, 16), name="x"), f2.out_spec())
    return _wrap(g)


ALL_EXAMPLES = [tiny_conv, conv_relu_pool, tiny_linear, residual, mlp_int4]
