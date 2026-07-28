# Progress

**Status: complete.** All six acceptance criteria green. See [HANDOFF.md](HANDOFF.md)
for what works, what is stubbed, and the highest-value next steps.

Live: https://hls-estimate.vercel.app · Repo: https://github.com/TarunYadgirkar/hls-estimate

## Milestones

- [x] **M0 Environment** — venv on Python 3.11 (system was 3.8, EOL); numpy, pytest,
      hypothesis, torch 2.2.2, onnx.
- [x] **M1 SPEC.md** — scope, interfaces, out-of-scope, acceptance criteria.
- [x] **M2 Acceptance suite written first** — RED baseline captured before any
      implementation (6 modules failing on missing imports).
- [x] **M3 IR + resource model + devices** — analytical / monotonicity / bit-width
      tests green.
- [x] **M4 DSE** — budget enforcement test green.
- [x] **M5 Codegen** — bit-exactness test green, and mutation-checked (dropping the
      rounding term fails 5 of 6 cases).
- [x] **M6 Calibration** — real published numbers pulled from the hls4ml papers;
      LUT/FF coefficients recalibrated (the hand-guessed ones were 4× high); error
      bands measured and recorded honestly in MODEL.md.
- [x] **M7 Frontend** — torch.fx ingest with batchnorm folding and quantization.
- [x] **M8 Report + CLI** — per-layer attribution, bottleneck, fit verdict.
- [x] **M9 Web UI** — Next.js estimator with live device floorplan, deployed to
      Vercel with security headers. TypeScript ports pinned to Python by 118 parity
      tests.
- [x] **M10 Docs** — README, MODEL, DECISIONS, DEPENDENCIES, HANDOFF.

## Test counts

| Suite | Cases |
|---|---|
| Python (`pytest -q`) | 82 |
| Web parity (`vitest run`) | 118 |

## Blocked

Nothing. Two items are deliberately unfinished and recorded in HANDOFF: the ONNX
frontend (secondary path per DECISIONS D6) and automated Vitis validation (Vitis is
not installed on this machine, so the comparison could not be tested).

## Notable corrections made along the way

- Initial LUT/FF constants were ~4× too high versus published data; refitted and
  labelled as fitted, with the independent-check rows called out separately.
- The web parity suite caught two real port bugs: wrong tensor wiring in the web
  examples, and a `maxParallel` that disagreed with `dse.py` on pooling layers.
- DSE truncated its search at 20,000 configurations silently; it now warns in Python
  and shows a banner in the UI.
