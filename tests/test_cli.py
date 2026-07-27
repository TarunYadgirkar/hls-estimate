"""CLI surface (SPEC §9)."""
import json
import os

import pytest

from hls_estimate.cli import main


def run(args, capsys):
    code = main(args)
    return code, capsys.readouterr().out


def test_list_models(capsys):
    code, out = run(["list"], capsys)
    assert code == 0
    assert "conv_relu_pool" in out


def test_estimate_prints_report(capsys):
    code, out = run(["estimate", "conv_relu_pool", "--device", "zynq-7020"], capsys)
    assert code == 0
    assert "verdict:" in out
    assert "bottleneck" in out


def test_estimate_json(capsys):
    code, out = run(["estimate", "tiny_linear", "--json"], capsys)
    assert code == 0
    data = json.loads(out)
    assert data["fits"] is True
    assert len(data["layers"]) >= 1
    assert "dsp" in data["totals"]


def test_emit_writes_cpp(tmp_path, capsys):
    dst = tmp_path / "net.cpp"
    code, _ = run(["emit", "tiny_conv", "-o", str(dst)], capsys)
    assert code == 0
    src = dst.read_text()
    assert "#pragma HLS DATAFLOW" in src
    assert "int main()" not in src


def test_emit_with_testbench(tmp_path, capsys):
    dst = tmp_path / "net_tb.cpp"
    code, _ = run(["emit", "tiny_conv", "-o", str(dst), "--tb"], capsys)
    assert code == 0
    assert "int main()" in dst.read_text()


def test_dse_prints_pareto_front(capsys):
    code, out = run(["dse", "tiny_linear", "--device", "ultra96"], capsys)
    assert code == 0
    assert "latency" in out.lower()


def test_unknown_model_is_an_error(capsys):
    code, _ = run(["estimate", "no_such_model"], capsys)
    assert code != 0


def test_validate_without_vitis_is_reported_not_crashed(capsys, monkeypatch):
    monkeypatch.setenv("PATH", "")  # guarantee vitis_hls is not found
    code, out = run(["estimate", "tiny_linear", "--validate"], capsys)
    assert code == 0
    assert "vitis" in out.lower()
