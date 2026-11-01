# Phase 6 Perf Gate Spec

## Scope

| Backend | Gate | Reference hardware |
|---|---|---|
| CUDA | BLOCKING | NVIDIA T4 |
| ROCm | BLOCKING (hardware-blocked, see clause below) | — |
| TPU | BLOCKING | TPU v5 |
| CPU | warn-only | Kaggle CPU |
| OpenCL | warn-only | — |

## Measurement protocol

- warmup: 10 iterations (untimed)
- timed: 100 iterations
- reported: mean, median, p95 (milliseconds)
- GPU timings bracketed with device synchronize on both sides
- same hardware, same build flags, same input shapes for baseline vs candidate

## Block criterion

A candidate run BLOCKS the phase when any BLOCKING-scope benchmark regresses
more than 10% in median latency versus the recorded baseline:

    regression_pct = (candidate_median - baseline_median) / baseline_median * 100
    BLOCK if regression_pct > 10.0 for any BLOCKING benchmark

Numerics are verified separately (PyTorch is the reference standard); a perf
regression never justifies a silent numeric divergence.

## Baseline recording

- The first green Phase 6 run on each reference hardware records baselines to
  `qa/phase6/perf/baseline_<hw>.json` (e.g. `baseline_t4.json`, `baseline_tpuv5.json`)
  using `bench_kernels.py --device <dev> --record-baseline`.
- Baselines are re-recorded only after deliberate, QA-approved perf work; the
  re-record run must itself pass the gate against the old baseline first.
- Baseline files are committed with the run log (notebook output + JSON).

## ROCm honesty clause

No ROCm hardware exists locally or on Kaggle. The ROCm-blocking gate is
recorded as hardware-blocked and applies where measurable; no ROCm numbers
are fabricated. If ROCm hardware becomes available, `bench_kernels.py`
runs unchanged with `--device cuda` semantics mapped to the ROCm build.

## How to run

Kaggle (canonical):
    !python3 bench_kernels.py --device cuda --out bench_t4.json
    !python3 bench_kernels.py --device cpu  --out bench_cpu.json
TPU v5 cell:
    !python3 bench_kernels.py --device tpu  --out bench_tpuv5.json
Compare:
    !python3 check_gate.py --baseline baseline_t4.json --candidate bench_t4.json

CPU/OpenCL results are warn-only: reported in the phase summary, never blocking.

## Evidence required per benchmark

benchmark name, device, iters, median/mean/p95 ms for litetorch and torch,
litetorch/torch ratio, baseline delta %. Every BLOCKING verdict cites the
numbers; "looks fine" is not a verdict.

---

## qa-p7 worker report (2026-10-17)

Delivered (tagged qa-p7):
- `perf-gate.md` (this spec): BLOCKING scopes CUDA/ROCm/TPU, warn-only CPU/OpenCL; T4 + TPU v5 references; warmup-10/timed-100 median protocol; >10% median regression = BLOCK; baseline recording procedure; ROCm hardware-blocked honesty clause.
- `bench_kernels.py`: 9 benchmarks (matmul fp32 1024^3; ew add fp32/uint8/int64 16M; conv2d 3x64x32x32 k3; softmax 1M; reduce sum 16M; all_reduce 2-rank 4MB gloo; onnx export 3-layer MLP), each timed for litetorch (probed pybind11 module, defensive API probing with SKIP reasons) and torch; median/mean/p95 + lt/torch ratio; JSON output; `--record-baseline` writes `baseline_<device>.json`. Zero comment lines in code (verified: no `#` characters; py_compile clean).
- `check_gate.py`: compares baseline vs candidate JSON, prints BLOCK lines with exact % deltas, exit 1 on BLOCK / 0 on PASS. Verified on synthetic data: +20% -> BLOCK exit 1; +5% -> PASS exit 0.

Environment findings (local VM):
- No torch installed locally; no usable Linux pybind11 `litetorch` module (only a Windows `.pyd` artifact and `liblitetorch.so` C++ lib in integration/). CPU Python benchmarks NOT runnable locally — recorded as "Kaggle CPU cell covers it"; bench_kernels.py runs on Kaggle where torch + built bindings exist.
- Dry run locally (no torch, no litetorch): script exits 0 with clear SKIP reasons per benchmark and prints exact build instructions for the litetorch module instead of crashing.

What runs where:
- Kaggle T4 x2 (commander browser session): `bench_kernels.py --device cuda` — BLOCKING verdict via check_gate.py.
- Kaggle TPU v5: `bench_kernels.py --device tpu` — BLOCKING.
- Kaggle CPU: `bench_kernels.py --device cpu` — warn-only.
- ROCm: hardware-blocked, no numbers fabricated.
