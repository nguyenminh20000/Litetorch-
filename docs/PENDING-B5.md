# Pending GPU verification (Batch B5)

The 7 GPU examples below are NOT verified on CPU. They stay
`REQUIRES-GPU` / `status=unverified` in this merge. Do NOT flip any
header in this package. The B5 GPU run is in flight on Kaggle (QA
owns the session; run order: pre-flight gate `e4-4` first,
`H1.3-1` last — it needs both T4 GPUs). B5 scripts and SHAs:
`results/b5-package-20261027-0226/MANIFEST.md` in the local
workspace.

| # | B5 script | Page | Header / block | Status in this merge |
|---|---|---|---|---|
| 1 | `b5-e4-4.py` | `en/explanations/E4-backends.md` (+ `vi/` mirror) | `<!-- ex: id=e4-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->` | unverified — pre-flight gate as code; runs FIRST |
| 2 | `b5-e1-9.py` | `en/explanations/E1-tensor-memory-model.md` (+ `vi/`) | `<!-- ex: id=E1-9 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->` | unverified |
| 3 | `b5-e3-3.py` | `en/explanations/E3-memory-eviction.md` (+ `vi/`) | `<!-- ex: id=e3-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->` | unverified |
| 4 | `b5-r5-flash-attention-qkv.py` | `en/reference/ops-conv-attention.md` §`Ops.flash_attention_qkv` (+ `vi/`) | No `ex:` header exists. Bare `python skip-run` block + `> REQUIRES-GPU (B5)` callout: the call throws without a GPU native backend; the block is the exact form to run on Kaggle GPU. | unverified — keyed by the B5 script name, not a page header |
| 5 | `b5-e5-7.py` | `en/explanations/E5-distributed.md` (+ `vi/`) | `<!-- ex: id=E5-7 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->` | unverified |
| 6 | `b5-t2.4-1.py` | `en/tutorials/T2-train-a-classifier.md` §T2.4 (+ `vi/`) | `<!-- ex: id=T2.4-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->` | unverified |
| 7 | `b5-h1.3-1.py` | `en/howto/H1-train-on-2-gpus.md` §H1.3 (+ `vi/`) | `<!-- ex: id=H1.3-1 device=gpu backend=cuda runtime=~60s torch-parity=n/a status=unverified -->` | unverified — runs LAST, needs both GPUs |

Out of scope: `e4-5` (ROCm callout in `E4-backends.md`,
`backend=rocm`, `status=unverified`) — no ROCm hardware in this
pipeline; it stays `UNVERIFIED-ROCm` and is not part of B5.

## Prose-only GPU sections — no measurement invented

- E7.8 (`en/explanations/E7-quantization.md`, "When not to
  quantize, and what needs a GPU"): serving throughput with
  quantized weights (tokens/s, time-to-first-token), FP8 matmul at
  scale, INT8 kernel benchmarks. Prose only — 0 code blocks. No
  throughput number is claimed.
- E8.8 (`en/explanations/E8-serving-llm.md`, "What is NOT verified
  here"): tokens/s and time-to-first-token for the scheduler,
  speculative decoding, Medusa; paged attention at scale;
  prefix-cache hit rates under real traffic; 2-GPU serving. Prose
  only — 0 code blocks. No throughput number is claimed.

No benchmark script was invented for E7.8/E8.8. Throughput
measurements for these sections need a separate measurement task.

## Post-B5 follow-up (only after B5 PASS)

One small patch, and nothing else:

- Flip the 6 existing headers above from `status=unverified` to
  `status=verified-<date>@<commit>` in both `en/` and `vi/`
  (12 header edits).
- For `r5-flash-attention-qkv` there is no header to flip: record
  the B5 result in the existing `> REQUIRES-GPU (B5)` callout prose
  on the page (both languages). Do not invent a header id into the
  page.
- Leave E7.8/E8.8 exactly as-is. Do not add measurements.
