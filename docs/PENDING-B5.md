# Pending GPU verification (Batch B5)

State as of 2026-10-31: the B5 GPU re-run (track 2, QA-owned
Kaggle session) completed 13:13-14:31+07 on PIN agent-v1
@cbba5af3, inputs kaggle-run-inputs @612b437b, plain product
tree, build rc 0 (731.7s), Native GPU pre-flight passed.
Record: `qa/workers/qa-b5-gpu-run2-result.md`. Verdict 4/7
PASS, 3/7 FAIL. Patch 09 (landed @350b1b39) flipped the 4
passing headers below to `status=verified-2026-10-31@cbba5af3`
in both `en/` and `vi/`. The 3 remaining headers stay
`status=unverified`. Do NOT flip them until their scripts pass
on a later run.

| # | B5 script | Page | Header / block | Status after the 2026-10-31 re-run |
|---|---|---|---|---|
| 1 | `b5-e4-4.py` | `en/explanations/E4-backends.md` (+ `vi/` mirror) | `<!-- ex: id=e4-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->` | VERIFIED — pre-flight gate, runs FIRST, exit 0 |
| 2 | `b5-e1-9.py` | `en/explanations/E1-tensor-memory-model.md` (+ `vi/`) | `<!-- ex: id=E1-9 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->` | VERIFIED — device string `gpu:0`, confirms patch 07 |
| 3 | `b5-e3-3.py` | `en/explanations/E3-memory-eviction.md` (+ `vi/`) | `<!-- ex: id=e3-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->` | VERIFIED — with a caveat: its script check is `device_count >= 1`, not `>= 2`, so this PASS does NOT mean the device-count defect is fixed (see row 7) |
| 4 | `b5-r5-flash-attention-qkv.py` | `en/reference/ops-conv-attention.md` §`Ops.flash_attention_qkv` (+ `vi/`) | No `ex:` header exists. Bare `python skip-run` block + `> REQUIRES-GPU (B5)` callout: the call throws without a GPU native backend; the block is the exact form to run on Kaggle GPU. | UNVERIFIED — parity max_abs_diff 0.525, identical to Oct 27 (factor-of-8 defect unchanged) |
| 5 | `b5-e5-7.py` | `en/explanations/E5-distributed.md` (+ `vi/`) | `<!-- ex: id=E5-7 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->` | UNVERIFIED — `a rank failed`; failure mode CHANGED vs Oct 27: both ranks `NCCL unavailable` in `FlatParameter::reduce_scatter_flat`, NO double-free message this time (open questions for track 4: pip NCCL presence, w15 auto-selection engagement, explicit-lib effect) |
| 6 | `b5-t2.4-1.py` | `en/tutorials/T2-train-a-classifier.md` §T2.4 (+ `vi/`) | `<!-- ex: id=T2.4-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=verified-2026-10-31@cbba5af3 -->` | VERIFIED — `training on: gpu:0`, exit 0 |
| 7 | `b5-h1.3-1.py` | `en/howto/H1-train-on-2-gpus.md` §H1.3 (+ `vi/`) | `<!-- ex: id=H1.3-1 device=gpu backend=cuda runtime=~60s torch-parity=n/a status=unverified -->` | UNVERIFIED — `expected 2 GPUs`; API reported device count 1 while nvidia-smi listed 2x T4 (device-count defect persists in the product until the fix merges; track-3 gates passed 3/3, fix cleared to merge, merge pending) |

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

## Remaining follow-up (only as each script passes)

- Flip the 2 remaining headers above (E5-7, H1.3-1) from status=unverified to status=verified-<date>@<commit> in both en/ and vi/ (4 header edits), each only after its script passes on a later GPU run: E5-7 needs the track-4 diagnosis; H1.3-1 needs the device-count fix merged. r5 needs the factor-of-8 fix, recorded in callout prose (no header to flip).
- For `r5-flash-attention-qkv` there is no header to flip: record
  the B5 result in the existing `> REQUIRES-GPU (B5)` callout prose
  on the page (both languages). Do not invent a header id into the
  page.
- Leave E7.8/E8.8 exactly as-is. Do not add measurements.
- Retire this file once all 7 scripts are verified.
