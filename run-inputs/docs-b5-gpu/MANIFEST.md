# B5 GPU verification package — manifest

- Assembled: 2026-10-27 02:26+07, agent docs-b5-package
- Local HEAD: `b326f29` (sync integration/ to remote agent-v1 @cdbc716c, sigmoid merge)
- Remote tip: `cdbc716c` — QA correction: the B5 build pins to the remote tip current at launch, re-verified then (standing PIN RULE). `b326f29` is the local sync commit, not the remote tip.
- Extension reference: Core D3 build, `build_commit()` = `88a4ba1` (reference only — Kaggle rebuilds from the pinned tip with `python3 setup.py build_ext --inplace`)
- Torch: 2.14.1+cpu (Kaggle env: use torch if importable, else scripts print `TORCH-ABSENT` and still run the litetorch side)
- QA thread: `docs-b5-gpu-20261027` (request comms 20261027-022500-docs-to-qa-b5-gpu.md)
- Correction 2026-10-27 (B5 e1-9 triage, QA APPROVE patch 07): scripts b5-e1-9.py and b5-e3-3.py corrected the str(t.device) assert from cuda:0 to gpu:0 (product prints gpu:0 by design); their SHA-256 rows above updated. Nothing else in these scripts changed.


## Count correction — 7 runnable scripts, not 9

The QA request lists 9 items. Two of them have NO code block to ship:

- E7.8 (`E7-quantization.md` §7.8, "When not to quantize, and what needs a GPU") — prose-only: serving throughput, FP8 matmul at scale, INT8 kernel benchmarks are listed as GPU measurements with no code.
- E8.8 (`E8-serving-llm.md` §8.8, "What is NOT verified here") — prose-only: tokens/s, time-to-first-token, paged attention at scale, prefix-cache hit rates, 2-GPU serving, with no code.

No benchmark scripts were invented for them — that would ship unverified new claims. If QA wants throughput measurements for E7.8/E8.8, they need a separate measurement task, not this package.

## Run order

1. `b5-e4-4.py` — pre-flight gate FIRST; exits non-zero if the gate fails, nothing else runs
2. `b5-e1-9.py`
3. `b5-e3-3.py`
4. `b5-r5-flash-attention-qkv.py`
5. `b5-e5-7.py`
6. `b5-t2.4-1.py`
7. `b5-h1.3-1.py` — LAST, needs both T4 GPUs

## Setup on Kaggle

- Set `LITETORCH_TREE` to the checked-out tree (or pass it as argv[1]); every script inserts `<tree>/python` on `sys.path`.
- Build first: `python3 setup.py build_ext --inplace` from the pinned tip; check/install the CUDA toolkit explicitly before the run.
- Each script prints `B5 <id> PASS` on success (exit 0) or `B5 <id> FAIL: <reason>` on failure (exit 1).
- Pre-flight gate (e4-4): `lt.cuda.is_available()` True, `device_count() >= 1`, `get_backend_name()` in ("CUDA", "ROCm"), `liblitetorch_gpu.so` next to the extension, and the `Native GPU (CUDA/HIP) successfully initialized.` line present in a fresh import's stdout. Note: if the init line is missing from the subprocess probe but present in the session log, that is a gate-logic bug — report it, do not waive the gate silently.
- e4-5 (ROCm) is out of scope: explicit UNVERIFIED-ROCm callout, no ROCm hardware in this pipeline.

## Files

| Script | Source page + header id | SHA-256 |
|---|---|---|
| scripts/b5-e4-4.py | en/explanations/E4-backends.md, id=e4-4 | 82d3b49689e994f1f28aaf1f69c5c65f5204fd079e94b1ea80719a1980ce653f |
| scripts/b5-e1-9.py | en/explanations/E1-tensor-memory-model.md, id=E1-9 | 230dfb9780e21929a4fb815aab9bfca5b116b88da75be4d3f500d3cd25df3e16 |
| scripts/b5-e3-3.py | en/explanations/E3-memory-eviction.md, id=e3-3 | bac2ed5d772810b4bfeea70213b28f6c0955d6b75b9757ea6ba960f2e23824a3 |
| scripts/b5-r5-flash-attention-qkv.py | en/reference/ops-conv-attention.md, `Ops.flash_attention_qkv` skip-run block (no metadata header) | b0acd63c81021ed9fa004b47015383056773eae984e8f5396b4d2068b79b67a3 |
| scripts/b5-e5-7.py | en/explanations/E5-distributed.md, id=E5-7 | 82fd6fcda06f6424d5b8714284163852ecf4f318120f57deb77228a766cf06cc |
| scripts/b5-t2.4-1.py | en/tutorials/T2-train-a-classifier.md, id=T2.4-1 | e571df06d0bf40d0a58d92a98ac634adb66d652fca4de3a2c6f807fae6f3589b |
| scripts/b5-h1.3-1.py | en/howto/H1-train-on-2-gpus.md, id=H1.3-1 | 89a9a842b9c2800383050cbfbe069b87760442dd16c30d7836d9845523efeb31 |

Verify after fetch: `sha256sum scripts/*.py` against this table before running.
