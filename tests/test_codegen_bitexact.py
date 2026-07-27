"""Acceptance #4 — the test that matters.

Emitted HLS C++ compiles as plain C++ (pragmas ignored) and produces bit-exact
output vs the golden PyTorch model on random inputs, for every supported op.
"""
import numpy as np
import pytest
import torch

from hls_estimate.codegen import emit_full
from hls_estimate.models import ALL_EXAMPLES
from tests.util import compile_cpp, run_binary, cxx

pytestmark = pytest.mark.skipif(cxx() is None, reason="no C++ compiler available")


def _random_input(spec, rng):
    lo, hi = -16, 15  # modest signed range; arithmetic is exact in int64 either way
    return rng.integers(lo, hi + 1, size=spec.shape, dtype=np.int64)


@pytest.mark.parametrize("factory", ALL_EXAMPLES, ids=[f.__name__ for f in ALL_EXAMPLES])
def test_emitted_cpp_is_bit_exact(factory, tmp_path):
    torch.manual_seed(0)
    module, graph = factory()
    source = emit_full(graph, name="net")
    binp = compile_cpp(source, workdir=str(tmp_path))

    rng = np.random.default_rng(1234)
    for _ in range(8):
        x = _random_input(graph.input_spec, rng)
        with torch.no_grad():
            y_torch = module(torch.from_numpy(x)).reshape(-1).tolist()
        y_cpp = run_binary(binp, x.reshape(-1).tolist())
        assert len(y_cpp) == len(y_torch), (
            f"{factory.__name__}: size mismatch {len(y_cpp)} vs {len(y_torch)}")
        assert y_cpp == y_torch, f"{factory.__name__}: not bit-exact"


def test_compiles_with_pragmas_present():
    # The emitted source must actually contain HLS pragmas and still compile as C++.
    _, graph = ALL_EXAMPLES[0]()
    source = emit_full(graph, name="net")
    assert "#pragma HLS" in source
    for pragma in ("PIPELINE", "UNROLL", "ARRAY_PARTITION", "DATAFLOW"):
        assert pragma in source, f"missing {pragma} pragma"
    compile_cpp(source)  # asserts on failure
