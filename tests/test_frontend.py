"""Frontend: ingest a real torch.nn.Module via torch.fx into the internal IR.

SPEC §1: conv2d, linear, relu, maxpool, batchnorm-folded, add, int8/int4.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from hls_estimate.codegen import emit_full
from hls_estimate.frontend import UnsupportedOp, from_torch
from hls_estimate.ir import Add, Conv2d, Linear, MaxPool2d, ReLU
from hls_estimate.models.executor import run_graph
from tests.util import compile_cpp, cxx, run_binary


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class BnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x) + x


class MlpNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class UnsupportedNet(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def test_conv_relu_pool_lowers_to_expected_ir():
    graph = from_torch(ConvNet().eval(), (1, 3, 8, 8))
    kinds = [type(n) for n in graph.nodes]
    assert kinds == [Conv2d, ReLU, MaxPool2d]
    conv = graph.nodes[0]
    assert (conv.in_ch, conv.out_ch, conv.kh, conv.kw, conv.pad) == (3, 4, 3, 3, 1)
    assert conv.out_h() == 8 and conv.out_w() == 8
    assert graph.output_spec.shape == (1, 4, 4, 4)


def test_mlp_lowers_to_linear_nodes():
    graph = from_torch(MlpNet().eval(), (1, 8))
    assert [type(n) for n in graph.nodes] == [Linear, ReLU, Linear]
    assert graph.nodes[0].in_features == 8
    assert graph.nodes[-1].out_features == 3


def test_residual_add_is_lowered():
    graph = from_torch(ResidualNet().eval(), (1, 3, 5, 5))
    assert [type(n) for n in graph.nodes] == [Conv2d, Add]
    add = graph.nodes[-1]
    assert len(add.inputs) == 2
    assert "x" in add.inputs, "residual branch must read the graph input"


def test_batchnorm_is_folded_away():
    graph = from_torch(BnNet().eval(), (1, 2, 6, 6))
    kinds = [type(n) for n in graph.nodes]
    assert MaxPool2d not in kinds
    assert kinds == [Conv2d, ReLU], f"batchnorm not folded: {kinds}"


def test_batchnorm_folding_preserves_float_math():
    torch.manual_seed(0)
    net = BnNet().eval()
    # Give the BN non-trivial statistics so folding is a real transformation.
    with torch.no_grad():
        net.bn.weight.copy_(torch.tensor([1.7, 0.4, -1.2]))
        net.bn.bias.copy_(torch.tensor([0.3, -0.8, 0.5]))
        net.bn.running_mean.copy_(torch.tensor([0.1, -0.2, 0.05]))
        net.bn.running_var.copy_(torch.tensor([0.9, 1.4, 0.7]))

    from hls_estimate.frontend import fold_conv_bn

    x = torch.randn(1, 2, 6, 6)
    with torch.no_grad():
        expected = net.bn(net.conv(x))
    folded_w, folded_b = fold_conv_bn(net.conv, net.bn)
    got = torch.nn.functional.conv2d(x, folded_w, folded_b, padding=1)
    assert torch.allclose(expected, got, atol=1e-5)


@pytest.mark.parametrize("w_bits", [4, 8])
def test_bit_width_is_propagated(w_bits):
    graph = from_torch(ConvNet().eval(), (1, 3, 8, 8), w_bits=w_bits)
    conv = graph.nodes[0]
    assert conv.w_bits == w_bits
    lim = 1 << (w_bits - 1)
    assert np.all(np.abs(np.asarray(conv.weight)) <= lim), "weights outside int range"


def test_unsupported_op_raises():
    with pytest.raises(UnsupportedOp):
        from_torch(UnsupportedNet().eval(), (1, 4))


@pytest.mark.skipif(cxx() is None, reason="no C++ compiler available")
@pytest.mark.parametrize("net,shape", [
    (ConvNet, (1, 3, 8, 8)),
    (BnNet, (1, 2, 6, 6)),
    (ResidualNet, (1, 3, 5, 5)),
    (MlpNet, (1, 8)),
])
def test_frontend_graph_emits_bit_exact_cpp(net, shape, tmp_path):
    torch.manual_seed(0)
    graph = from_torch(net().eval(), shape)
    binp = compile_cpp(emit_full(graph, name="net"), workdir=str(tmp_path))
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.integers(-8, 9, size=shape, dtype=np.int64)
        expected = run_graph(graph, torch.from_numpy(x)).reshape(-1).tolist()
        got = run_binary(binp, x.reshape(-1).tolist())
        assert got == expected
