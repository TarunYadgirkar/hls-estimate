# Progress

Updated continuously. Newest status at top.

## Now
- [x] M0 Environment: venv (py3.11), numpy/pytest/hypothesis/torch 2.2.2/onnx installed.
- [x] M1 SPEC.md written.
- [x] M2 Acceptance test suite written — RED baseline captured (6 modules).
- [~] M3 IR + resource model + devices done. analytical/monotonicity/bitwidth GREEN (39 tests).
      Golden torch executor + example model factories written; DSE not yet written.
- [ ] M3 remaining: dse.py -> budget test green.
- [ ] M4 HLS C++ codegen + bit-exact test green.
- [ ] M5 report + calibration data (calibration test) + CLI.
- [ ] M6 README + HANDOFF + MODEL.md finalised.

## Blocked
- none

## Next (resume here after restart)
1. Smoke-test executor: `.venv/bin/python -c "from hls_estimate.models import ALL_EXAMPLES; import torch; m,g=ALL_EXAMPLES[0](); print(m(torch.zeros(g.input_spec.shape,dtype=torch.int64)).shape)"`
2. Write hls_estimate/dse.py (explore + Pareto) -> run tests/test_budget.py.
3. Write hls_estimate/codegen/ -> run tests/test_codegen_bitexact.py (the key test).
4. Write hls_estimate/models/literature.py + MODEL.md bands -> tests/test_calibration.py.
5. report.py + cli.py, then README/HANDOFF.

## Env note
- venv at .venv (python3.11 from /opt/miniconda3). Run tests: `.venv/bin/python -m pytest -q`.
