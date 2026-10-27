# Changelog — Docs Rewrite

This is the changelog for the LiteTorch documentation rewrite itself (2026-10-26 → present): what each batch landed, how it was verified, what defects it found, and what is still open. Product changes are tracked in the framework repo; this page tracks only the docs.

- **Baseline:** the rewrite starts from the skeleton in `integration/docs-en/` + `docs-vie/` (12 sections each, ~1,600 lines EN / ~1,800 lines VI) plus `integration/docs/mobile-arm.md` + `tokenizers.md`. The rewrite lives in `workspace/framework/docs/` (`en/` + `vi/`), with patches in `workspace/framework/patches/docs/` for QA review. Nothing is written directly to `integration/`.
- **Verification standard:** every code example runs against the current tree — CPU minimum, GPU verified on Kaggle via QA. An example that does not run is a defect. Parity against PyTorch (torch 2.14.1+cpu) at FP32 1e-4, values AND grads. Full method: `plan/verification.md`.
- **Extension baseline:** all batches so far verify against the Core D3 extension `build_commit 88a4ba1` (`workspace/framework/core/litetorch-d3/python`). Tree HEAD has moved since (now `b326f29`); B5 re-verifies at head.

## Batch B0 — Harness, glossary, defect fixes (2026-10-26)

- **Delivered:** verification harness (`docs/tools/`: `extract_examples.py`, `run_examples.py`, `parity_check.py`, `requirements-check.txt`); Vietnamese glossary (`docs/glossary/vi-terms.md`, 50 terms, 11 inconsistencies logged); defect analysis D1–D8 + D9 resolution.
- **Verification:** skeleton extraction ran 122 examples (61 EN + 61 VI): PASS 68 / FAIL 54. Failure classification in `plan/skeleton-failures.md`: Group A DEPENDENT 19 IDs, Group B real API mismatches 7 IDs, Group C hang 1 ID (8.2-1, rendezvous without mp spawn). All failures symmetric EN/VI — 27 unique IDs.
- **Findings:** D9 (§8.3 FSDP example used `lt.distributed.FullyShardedDataParallel`, which exists nowhere in the tree) — resolved via Distributed dist-w16: rewrite around `fsdp_fully_shard`. Build-time measurement: local full build does not complete in one session (killed at 44.6 min, 99/167 sources) — verification builds moved to Kaggle CPU.
- **Defects filed:** D-06 + D-07 fix requests to Core (comms `20261026-212600-docs-to-core-d6f1.md`); D9 question to Distributed (comms `20261026-212800-docs-to-distributed-d9f2.md`); D9 side findings to QA (comms `20261026-213400-docs-to-qa-d9side.md`).

## Batch B1 — Front matter + tutorials (2026-10-26)

- **Delivered:** front matter, 10 pages, 1,828 lines (`en/index.md`, `en/installation.md`, `en/installation/docker.md`, `en/installation/kaggle.md`, `en/quickstart.md` + VI mirrors). Tutorials T1–T4, 8 files, 2,178 lines, 50 fenced blocks.
- **Verification:** front matter 16/16 PASS (8 blocks × EN/VI) vs D3 ext 88a4ba1; every documented expected output matches byte-for-byte. Tutorials 48/48 PASS, 0 FAIL, 2 SKIPPED (T2.4 GPU variant, REQUIRES-GPU/Kaggle-B5). Parity FP32 1e-4: CE loss+grads identical, flash/paged attention values identical to torch, w8a8 exact vs int8 formula, JIT exact, fused GELU exact vs `F.gelu(approximate='tanh')` (D-09).
- **Findings:** `DataLoader(shuffle=True)` is NOT reproducible despite `lt.manual_seed(42)` (D-46) — examples switched to `shuffle=False`. Docker binary absent locally (Dockerfiles unexecuted); Kaggle GPU checks not run (B5); front-matter torch parity not run (torch lives in `~/torch-venv`, not system pip) — B5 item.

## Batch B2 — How-to guides + core reference (2026-10-26)

- **Delivered:** how-to H1–H8, 16 pages, 2,854 lines. Reference R1–R4, 8 files, 3,008 lines (1,504 EN / 1,504 VI), 127 entries.
- **Verification:** how-to PASS 50 / FAIL 0 / SKIPPED 6 (56 blocks; H1.3 GPU skipped, 2 skip-run fragments). Parity: H3.3 4.8e-07, H4.1 8.9e-08, H5.4 7.5e-09 (fwd+grad), H8.1 gradcheck True, H8.2 0.0/0.0. Reference R1–R4: 112/112 PASS, parity 70/70 PASS FP32 1e-4 incl. autograd grads.
- **Findings:** skeleton failures fixed (Group C hang via mp spawn; Group B paged_attention/scheduler/prefix-cache/w8a8; D9 §8.3 rewrite; D3 `lt.DataType.INT64`; D4 standalone H7.1; D6). New defect H7.2 filed to Core (comms `20261026-233500-docs-to-core-h7d2.md`): Python-backed `Dataset` aborts at shutdown (SIGABRT) unless `del loader/dataset` explicitly; trampoline dispatches to `size`/`get`, not `__len__`/`__getitem__` (D-37).
- **Fixes landed during the batch:** D-07 merged agent-v1 @7b8981e; D-06 merged agent-v1 @89afd67 (qa-r62). Tree HEAD moved 114f2e1 → 89afd67 during the batch.

## Batch B3 — Full API reference R5–R16 (2026-10-26/27)

- **Delivered:** 24 files, 15,720 lines total.
  - R5–R6 (ref-a): 4 files, 1,680 lines, 50 entries. Examples 40/40 PASS, parity 39/39 PASS (worst 1.9e-06).
  - R7–R8 (ref-b): 4 files, 6,234 lines, 109 nn classes + `nn.generate` + 10 functional + 11 init + 5 utils. Examples 136/136 PASS, parity 100 checks PASS.
  - R9–R11 (ref-c): 6 files, 3,928 lines. Examples 196/196 PASS, parity 86/88.
  - R12–R16 (ref-d): 10 files, 3,878 lines, ~129 entries. Examples 68/74 PASS (6 skipped fragments), parity 7/9.
  - **Batch totals:** 440 examples PASS, 0 FAIL; parity 232 checks PASS.
- **Defects filed:**
  - Core, comms `20261026-234500-docs-to-core-ceint64.md`: CE/NLL/fused-LCE reject INT64 targets (D-23).
  - Distributed, comms `20261026-234600-docs-to-distributed-ring.md`: `ring_attention` uncallable from Python (D-38).
  - Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md`: NAdam bug (D-30), OneCycleLR shift (D-31), NegativeBinomial swap (D-32), AsyncCheckpointSaver hang (D-36, side note).
  - Core, comms `20261027-000800-docs-to-core-ddp-grad.md`: DDP grad doubling (D-33).
  - Core, comms `20261027-001600-docs-to-core-r7-divergences.md`: D-22–D-29 (multi-arg `__call__`, INT32 loss targets, Module registration, Parameter construction, HingeEmbeddingLoss, MultiMarginLoss, CTCLoss, TransformerDecoderLayer) + VocabParallelEmbedding forward broken (D-39).
- **Environment incident:** the VM was replaced mid-batch; `~/torch-venv` torch was wiped (27 packages left, corrupt sympy dist-info). Torch 2.14.1+cpu reinstalled with `--ignore-installed`; all verification after the reinstall used the restored venv. Tree HEAD moved 88a4ba1 → b326f29 (sync to remote agent-v1 @cdbc716c, sigmoid merge). Binding drift 88a4ba1→b326f29 across all cited files: exactly two changes — the D-06 fix (`bind_tpu.cpp:155`) and a `NoGradGuard` in `load_parameters` (no API impact).

## Batch B4 — Explanations + front matter (2026-10-27, in progress)

- **Delivered (this worker):** `migration.md` (PyTorch → LiteTorch mapping tables + worked port, parity 5.96e-08), `behavioral-differences.md` (canonical D-01–D-46 with fix status + live demonstrations), `troubleshooting.md` (symptom → cause → fix, from verified failures only), `changelog.md` (this page). EN + VI full parity.
- **Delivered (parallel worker, docs-b4-expl):** explanations E1–E8 (tensor, autograd, memory eviction, backends, distributed, JIT fusion, +2).
- **New in this batch:** D-30–D-46 numbered and folded into the Behavioral Differences page; D-01 corrected (data_ptr is C++-only — unbound in Python, verified live); the obsolete D-06 `.cast(INT64)` workaround and D-07 "cast the result back" line are removed from the docs.
- **Verification:** see `results/b4-front2-<timestamp>/summary.md` (per-example logs alongside).

## Known gaps (B5 — GPU/TPU verification via QA)

- Front-matter torch parity (B1-front's 8 examples, `torch-parity=n/a`) — torch is in `~/torch-venv`.
- Dockerfiles (docker binary absent locally — unexecuted, stated on the page).
- Kaggle GPU checks: T2.4 GPU variant, H1.3 (2-GPU training), `flash_attention_qkv`, ROCm `UNVERIFIED-ROCm` callouts, TPU package (separate `kaggle-run-inputs` push, only after GPU green).
- Re-verification at head: all batches verify against ext 88a4ba1; tree HEAD is b326f29. B5 re-runs the harness against the head extension.
- Regression checks: L-02 (D-07 linalg parity), L-05 (D-06 `.long()` INT64 as indices), `r1-cast-long` example re-run (its printed output changes with the D-06 fix; header reset to `unverified`).
- Live checks L-01–L-04 (Kaggle GPU/TPU): D-02 fallback with a real GPU present-but-busy, D-10 transfer direction, D-21 int64 FP32 error magnitudes, D-19 quantized promotion vs torch, D-20 `flash_attention_half` on TPU.
- Open defects awaiting owner-team triage: D-22–D-29, D-30–D-33, D-37–D-39 (filed, see Behavioral Differences page for the comms threads); open issues D-14 (`.numpy()` INT64), D-01, D-02, D-09, D-20, D-35, D-40–D-45; post-training OOM (9 occurrences) still open. MNIST OOM [BLOCKER] CLOSED 2026-10-27 as not-reproduced (Path A PASS on b476a5e, exact config; no named cause, no candidate fix found).
