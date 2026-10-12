# Wave 3 — MFU Push: Tiled FlashAttention for T4 + Profiling Plan

Target workload: `tests/bench_gptblock_mfu.py`, GPTBlock d_model=2560, n_q_heads=20,
n_kv_heads=5, d_ff=6912, B=1, T=512, 69.47M params.
Baseline (Kaggle 2xT4): 1126.7 ms/iter fwd+bwd = 0.19 TFLOPS FP32 = 2.3% of T4 FP32 peak.
PyTorch on the same workload: ~13.2% MFU. Goal: beat PyTorch, not an absolute number.

Two new profiling scripts (public Python API only, run on Kaggle by the coordinator):

- `tests/profile_gptblock_breakdown.py` — per-stage fwd/bwd ms (20 iters, 3 warmup,
  `lt.cuda_synchronize()` around every region), full block + attn_norm + q/k/v/o projs +
  `Ops.flash_attention` direct + rope + ffn gate/up/down + swiglu elementwise + residual add,
  with a % of full-block table. Accepts `--fp16`.
- `tests/profile_allocator.py` — `lt.cuda.memory_allocated()` / `lt.get_gpu_memory_used()`
  trace (graceful fallback if absent): per-forward-stage deltas on a manual block replica,
  3-iter fwd/bwd/zero_grad growth check on the real block, FP32 then FP16, with OOM
  capture reporting bytes allocated at failure.

## 1. Static FLOP budget (ground truth the scripts must explain)

6N accounting for linears (2 fwd + 4 bwd per param), exact GEMM counts for attention.
Per-iter FLOPs, B=1, T=512:

| stage | params | GFLOP fwd+bwd | share |
|---|---|---|---|
| ffn_gate | 17.69M | 54.36 | 24.5% |
| ffn_up | 17.69M | 54.36 | 24.5% |
| ffn_down | 17.69M | 54.36 | 24.5% |
| proj_q | 6.55M | 20.13 | 9.1% |
| proj_o | 6.55M | 20.13 | 9.1% |
| proj_k | 1.64M | 5.03 | 2.3% |
| proj_v | 1.64M | 5.03 | 2.3% |
| attention (QK^T + AV, bwd ~2x fwd) | — | 8.06 | 3.6% |
| total | 69.47M | ~221.6 | 100% |

221.6 GFLOP / 1.1267 s = 0.197 TFLOPS. PyTorch at 13.2% MFU = 1.07 TFLOPS = ~207 ms/iter.
Gap to close: ~920 ms.

Sanity check on where time *should* go: linears are 96.4% of FLOPs. If cuBLAS GEMMs ran
at even 1 TFLOPS, linears = 213 ms. Attention at 0.5 TFLOPS effective = 16 ms.
Elementwise (4 RMSNorms, 2 ropes, silu muls, adds) is memory-bound on ~200 MB traffic,
a few ms at 320 GB/s. Python per-op dispatch for ~50 ops is tens of ms. Expected total
is therefore on the order of 250–300 ms — the measured 1126.7 ms leaves ~800 ms
unexplained by FLOPs. Either GEMM throughput is far below 1 TFLOPS on these shapes, or a
pathological overhead (allocator, syncs, per-node autograd cost) dominates. The per-stage
table from `profile_gptblock_breakdown.py` decides which.

## 2. The "fused-QKV chunk backward ~300MB/iter" finding

Grep result: there is NO `Ops::chunk` / `.chunk(` call anywhere in the attention path
(`src/ops/attention.cpp`, `src/nn/gpt.cpp`, `src/nn/gpt2b.cpp`). Wave 2's fused QKV
(`ea7f6d0`, in `nn::MultiHeadAttention`) splits with the custom `qkv_split_transpose`
CUDA kernel (`qkv_do_split`), not `Ops::chunk`. Its backward (`FlashAttentionQKVNode::
backward`, `qkv_do_merge`) allocates exactly ONE `(B,T,3*C)` grad tensor per backward —
no 3x blowup from that node itself.

The pathological pattern the report describes DOES exist in the chunk/split backward
implementation itself, `NarrowNode::backward` (`src/ops/elementwise.cpp`):

- `chunk` delegates to `split`, which creates one `NarrowNode` per piece.
- Each `NarrowNode::backward` allocates `Tensor::zeros(input_shape)` — the FULL input
  shape, in FP32 by default — then `region->copy_(grad_output)`, then casts if the dtype
  mismatches. So a 3-way chunk backward transiently allocates 3x the input tensor, and
  6x for FP16 inputs (FP32 zeros + FP16 cast), before any of it is freed.

In the GPTBlock bench specifically, "~300MB/iter" matches the *aggregate* backward
transient, not one op: 7 Linear backward `grad_weight` allocations = 278 MB FP32
(= param count, freed after accumulation into `.grad`) plus activation grads. The two
attention-specific hogs inside that number:

- `ensure_attn_scratch` (`src/backend/gpu_native/nn/flash_attention.cu`): 4 x
  B*H*Tq*Tk **float** buffers = 4 x 20 x 512 x 512 x 4 B = 80 MiB, allocated with raw
  `cudaMalloc` (bypasses both the CachingAllocator and `MemoryManager`, hence invisible
  to `lt.cuda.memory_allocated()`), persistent per device, **dtype-blind** (full 80 MiB
  even in FP16 mode), and every `cudaMalloc` device-synchronizes.
- Saved `attn_p` (B,H,T,T): 20.0 MiB FP32 / 10.0 MiB FP16, plus a 20 MB D2D copy in
  `forward_save_p`.

Recommendation (no C++ changed in this wave): replace `NarrowNode::backward`'s
`Tensor::zeros` with a dtype-aware allocation, and scatter via index kernel instead of
full-size zeros + region copy. The real fix for attention memory is sections 3–5.

## 3. Tiled FlashAttention forward spec (Tesla T4, sm_75)

Key fact: the current `Ops::flash_attention` is flash in name only —
`gpu_flash_attention` runs S = QK^T (cuBLAS) → softmax S→P (kernel) → O = PV (cuBLAS),
materializing full O(T^2) S and P. The tiled kernel below removes all O(T^2) traffic.

Hardware (verified): T4 = TU104, sm_75, 40 SMs, 64 KiB shared memory per block,
65536 registers/SM, 320 GB/s GDDR6, FP32 8.14 TFLOPS, FP16 tensor 65.1 TFLOPS.
Turing tensor cores: `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`.

Algorithm: online softmax (one pass over K/V, no S/P materialization).

- Grid: one CTA per (b, h_q, query-tile i). H_q=20, T=512, Br=64 → 8 tiles → 160 CTAs.
- GQA: kv head h_kv = h_q / g, g = H_q / H_kv = 4. Each CTA loads K/V of its kv head.
- Per CTA, registers hold: m_i (row max, FP32, Br values), l_i (row sum, FP32),
  O_i accumulator (Br x D, FP32), Q_i tile (Br x D, FP16) in shared.
- Loop over KV tiles j = 0 .. Tk/Bc-1:
  - Load K_j, V_j (Bc x D, FP16) to shared (double-buffered if budget allows).
  - S_ij = Q_i @ K_j^T via mma, FP32 accumulate.
  - Causal mask: rows r in [i*Br,(i+1)*Br), cols c in [j*Bc,(j+1)*Bc). If
    (j+1)*Bc <= i*Br: no mask. If j*Bc >= (i+1)*Br: skip whole block. Else per-element
    c <= r, implemented as -inf fill before rowmax.
  - Online update: m_new = max(m_old, rowmax(S_ij)); l_new = l_old * exp(m_old-m_new)
    + rowsum(exp(S_ij - m_new)); O = O * exp(m_old - m_new) + exp(S_ij - m_new) @ V_j.
  - P_ij = exp(S_ij - m_new) kept only in registers for the GEMM; never stored.
- Epilogue: O_i *= 1/l_i; write O (FP16 or FP32); write LSE_i = m_i + log(l_i), FP32,
  shape (B, H_q, Tq) = 1 x 20 x 512 x 4 B = 40 KiB.

Shared-memory budget, recommended config Br=64, Bc=64, D=128, FP16 I/O:

| tile | shape | bytes |
|---|---|---|
| Q_i | 64 x 128 x 2 B | 16,384 |
| K_j | 64 x 128 x 2 B | 16,384 |
| V_j | 64 x 128 x 2 B | 16,384 |
| S_ij | registers (mma fragments) | 0 shared |
| total | | 49,152 < 65,536 |

1 CTA/SM at 48 KiB; alternative Br=32, Bc=32 (24 KiB) allows 2 CTAs/SM if occupancy
bites. 4 warps/CTA (128 threads), each warp owns 16 rows; k-loop over D in 8-wide MMA
steps (16 steps for D=128).

FP16 variant: Q/K/V/O in FP16, all MMA accumulation FP32, LSE FP32. No dtype-blind
FP32 scratch anywhere. Complexity O(B*H*T^2*D) FLOPs, O(B*H*T*D) memory.

## 4. FA2-style recomputation backward spec

No stored S/P/dS/dP. Inputs: Q, K, V, O, dO (B,H,T,D), LSE (B,H,Tq) FP32.
Outputs: dQ (B,Hq,Tq,D), dK/dV (B,Hkv,Tk,D).

- Grid over query tiles (b, h_q, i), same as forward. Each CTA fully accumulates its
  dQ_i tile in SRAM — no atomics on dQ.
- Per CTA: load Q_i, O_i, dO_i, LSE_i to shared/registers. Precompute
  D_i = rowsum(dO_i * O_i), FP32, Br values in registers (one pass, reused for all j).
- dQ accumulators: Br x D FP32 in registers/shared, zeroed.
- Loop over KV tiles j: load K_j, V_j; recompute S_ij = Q_i @ K_j^T (mma, FP32);
  P_ij = exp(S_ij - LSE_i) in registers; dV_j += P_ij^T @ dO_i (mma);
  dP_ij = dO_i @ V_j^T (mma); dS_ij = P_ij * (dP_ij - D_i) (elementwise, registers);
  dQ_i += dS_ij @ K_j (mma); dK_j += dS_ij^T @ Q_i via **global atomicAdd**;
  dV_j accumulation likewise via atomicAdd into the shared dK/dV buffers.
- Causal mask recomputed identically to forward (same skip logic).
- dK/dV accumulation buffers are FP32; cast to FP16 at the end if needed. sm_75
  supports FP32 and FP16 global atomicAdd (sm_70+), but FP32 accumulate + cast avoids
  any throughput doubt on FP16 atomics. Contention: Tq/Br = 8 query tiles per (b,h)
  atomically add into the same (Bc x D) dK/dV tiles — 8-way, acceptable.
- GQA: dK/dV naturally accumulate over the g=4 query heads sharing one kv head via the
  same atomic path; no separate reduction pass.

Traffic per backward: ~2.5x forward (recompute), all O(T*D). No O(T^2) allocation.

## 5. Expected memory change (B=1, H=20, T=512, D=128)

| buffer | now (FP32) | now (FP16) | tiled (FP32) | tiled (FP16) |
|---|---|---|---|---|
| saved attn_p (B,H,T,T) | 20.0 MiB | 10.0 MiB | 0 | 0 |
| raw scratch 4x S/P/dP/dS | 80.0 MiB (FP32!) | 80.0 MiB (FP32!) | 0 | 0 |
| LSE (B,H,T) FP32 | 0 | 0 | 40 KiB | 40 KiB |
| O (B,H,T,D) | 5.0 MiB | 2.5 MiB | 5.0 MiB | 2.5 MiB |
| bwd grad_q/k/v (+out/gout copies) | ~17.5 MiB | ~9 MiB | ~17.5 MiB | ~9 MiB |
| attention total | ~122 MiB | ~101 MiB | ~22.5 MiB | ~11.5 MiB |

Eliminated: 100 MiB (FP32) / 90 MiB (FP16) per block, of which 80 MiB is raw
cudaMalloc invisible to the Python memory API. Scaling: at T=2048 the current design
needs P = 20 x 2048^2 x 4 B = 335 MiB + 1.34 GiB scratch — instant OOM; tiled needs
LSE = 160 KiB. This unblocks the 2B-model context lengths.

Expected speed: attention is 8 GFLOP/iter (3.6% of FLOPs) but currently pays 4 cuBLAS
launches on skinny (20x512x512x128) shapes + 2 softmax kernels over 5.2M rows + 100 MB
extra traffic. Tiled FP16 on tensor cores should run attention at several TFLOPS
effective (few ms) vs tens of ms now. The big lever remains the GEMMs (96.4% of FLOPs):
with tensor-op math the three 17.69M-param FFN GEMMs should approach 5–10 TFLOPS
effective FP16, i.e. ~15–30 ms for all of fwd+bwd.

## 6. Attack priority (to confirm/reshuffle with measured numbers)

1. **GEMM throughput on the 7 linears** — 96.4% of FLOPs. If the per-stage table shows
   proj/ffn stages dominating, the win is cuBLAS config (tensor-op math already toggled
   by `set_fp16_math_enabled`; verify plan-cache hits, try cuBLASLt bias epilogues,
   check for accidental FP32 compute on FP16 tensors), not more kernels.
2. **Tiled attention (sections 3–4)** — biggest structural win: removes 100 MB/iter,
   removes the dtype-blind 80 MiB raw scratch, unblocks long context. Even at only 3.6%
   of FLOPs, the current implementation's memory traffic and launch overhead are
   disproportionate; measure first.
3. **FP16 upcast temporaries** — `sigmoid` and `rope` both do FP16→FP32→FP16
   (see section 7); native FP16 kernels (or a fused silu) remove 2x transients.
4. **Python/autograd per-op overhead** — the micro_sum vs full_block row in the script
   output quantifies it; if large, fuse elementwise chains at the Python level
   (swiglu is already 2 muls + sigmoid = 4 nodes).
5. **Allocator** — `CachingAllocator::free_gpu_blocks_` is keyed by exact
   (device, size) with no splitting: freed blocks only satisfy identical-size requests.
   Mixed FP16/FP32 traffic doubles distinct sizes and fragments the cache into
   unusable exact-size buckets. Consider size-class bucketing or best-fit-with-split.

## 7. FP16 OOM hypothesis (for the allocator trace to confirm)

`--fp16` OOMs in `loss.backward()` on a ~14 MB alloc. Prime suspect, by exact size
match: **(B, T, d_ff) = (1, 512, 6912) in FP32 = 14,155,776 B = 13.5 MiB**.

- `Ops::sigmoid` (`src/ops/activation.cpp`): non-FP32 input is cast to FP32, computed,
  cast back. In the SwiGLU path `mul(gate, sigmoid(gate))`, every forward transiently
  allocates a_fp32 + out_fp32 = 2 x 14.2 MiB, and the backward (SigmoidNode) allocates
  grad tensors from `out_c->dtype` — fine after `5abd508`, but the forward 2x transient
  remains. Same pattern in `Ops::rope` (FP16→FP32→FP16, 2 x 5.2 MiB per call, x2 calls).
- `ensure_attn_scratch` allocates its 80 MiB in **FP32 regardless of dtype**, via raw
  cudaMalloc outside the caching allocator — in FP16 mode this is 8x the FP16 P size
  and the single largest hidden consumer.
- Persistent FP16 state is *smaller* than FP32 (params 139 vs 278 MB, grads likewise),
  so the OOM cannot be capacity on a 16 GB T4 (limit = 3/4 device = 12 GB) — it must be
  transient 2x upcasts colliding with fragmentation from exact-size cache buckets.
- `profile_allocator.py` is built to catch it: per-stage FP16 forward deltas will show
  the sigmoid/rope 2x spikes, and the wrapped backward reports the exact byte count at
  failure. If the failing alloc is ~14.2 MB, the sigmoid-upcast hypothesis is confirmed
  and the fix is a native FP16 sigmoid (or fused SwiGLU) kernel — no framework change.

Caveat for reading the trace: `lt.cuda.memory_allocated()` tracks `MemoryManager`
registrations only; the 80 MiB raw attn scratch never appears in it. Compare
`memory_allocated` deltas against the section-5 table accordingly.

## 8. Implementation notes for the build team

- Public Python API is unchanged: `lt.Ops.flash_attention(q, k, v, causal)` keeps its
  signature; the tiled kernel slots in behind the existing `flash_attention` /
  `flash_attention_half` entry points (the `g_fa3_fwd_fn`/`g_fa3_bwd_fn` hooks in
  `flash_attention.cu` already anticipate a replacement implementation).
- Backward needs one new saved tensor: LSE (B,H,Tq) FP32 replaces saved P (B,H,T,T).
  `FlashAttentionNode::backward` signature is unchanged (recompute from saved Q/K/V/O).
- Keep the current cuBLAS path as fallback for odd D (non-multiple-of-8 head dims).
- Validate against the existing `test_flash_causal_fp16.py` and the new breakdown
  script: fwd diff 0.0 vs current impl, grad diffs < 1e-4 FP16 / < 1e-6 FP32.
- Do not touch `NarrowNode` semantics in this wave beyond the dtype-aware allocation;
  numerics must stay bit-identical for the parity suite.
