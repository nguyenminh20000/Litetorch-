# Softmax FWD warp-kernel C++ vehicle

Covers the FORWARD warp softmax kernels (`softmax_fwd_warp_kernel`,
`softmax_fwd_warp_half_kernel`), which no public Python entry point reaches
on native CUDA. The Python gate script
(`qa-softmax-gate-script/softmax-warp-gpu-gate.py`, SHA
`855f007bef8eb57254c72650a6c7a0f293957134b10b67056d548ce810cebd3f`)
covers the 24 BWD cases and reports FWD NOT_COVERED. This vehicle closes
that gap by calling the non-LSE backward entry point directly, on whose
path `launch_softmax_fwd` recomputes P.

## Verified entry point (tip agent-v1 f9ce6609)

`src/backend/gpu_native/nn/flash_attention.cu`:

- `gpu_flash_attention_backward`, :1225. Signature (confirmed by the
  matching typedef in `src/backend/backend.cpp:45`):
  `extern "C" void gpu_flash_attention_backward(void *dQ, int64_t dq_off,
  void *dK, int64_t dk_off, void *dV, int64_t dv_off, void *O, int64_t o_off,
  void *dO, int64_t do_off, void *Q, int64_t q_off, void *K, int64_t k_off,
  void *V, int64_t v_off, int64_t B, int64_t H, int64_t H_kv, int64_t Tq,
  int64_t Tk, int64_t D, float scale, int causal)`
- `gpu_flash_attention_backward_half`, :1370. Identical signature and
  argument order, `__half` buffers.
- CUDA path, FP32 :1236-1277: `gqa_gemm_s_f32` (S = scale * K^T Q into
  scratch) -> `launch_softmax_fwd(scratch_s, scratch_p, B*H*Tq, Tk, Tq,
  causal)` at :1259 -> `gqa_gemm_dv_f32` -> `gqa_gemm_ds_f32` ->
  `launch_softmax_bwd` at :1267 -> `gqa_gemm_dq_f32` -> `gqa_gemm_dk_f32`.
  Half path :1400-1415 calls `launch_softmax_fwd_half` at :1407.
  `launch_softmax_fwd` (:479) dispatches Tk <= 32 to the warp kernel,
  Tk > 32 to the fused kernel. `O` is unused on the native path (no `o_ptr`
  variable; `O` appears only in the FA3 branch) — the vehicle still passes
  a valid zero-filled O buffer.
- FA3 bypass: if `g_fa3_bwd_fn` is set, the native path is skipped entirely
  (:1231, :1376). The pointers are set only by `gpu_init()` /
  `gpu_init_device()` (`src/backend/gpu_native/entrypoint.cu:252-253,
  277-304`) when `libflash_attn.so` is found at one of `libflash_attn.so`,
  `/usr/local/cuda/lib64/libflash_attn.so`, `./libflash_attn.so`. The
  vehicle pre-checks the same three paths and exits 4 NOT_COVERED if any
  exists, before loading anything.
- `gpu_init()` (`entrypoint.cu:266`) must be called first: it sets device 0
  and creates the compute/comm/h2d/d2h streams. This mirrors
  `src/backend/backend.cpp:221,291-293`, which loads `gpu_init` and calls
  it during backend init. Streams (`dev_stream`, :37-57) and the cuBLAS
  handle (`get_cublas_handle`, :180) are otherwise lazily created inside
  the .so; no other init is required.
- The .so is built by `setup.py:259-290` with
  `nvcc -O3 --shared --extended-lambda -Xcompiler -fPIC -arch=native
  src/backend/gpu_native/kernels.cu -o <build_lib>/liblitetorch_gpu.so
  -lcublas -lcublasLt`; `kernels.cu` includes `entrypoint.cu`, which
  includes `nn/flash_attention.cu` at :138. No `-fvisibility=hidden`,
  so the `extern "C"` symbols are exported. On Kaggle the .so lands at
  `<tree>/python/liblitetorch_gpu.so` (copy from
  `build/lib.linux-x86_64-cpython-313/` next to the extension if needed).
- Why Python cannot reach this path: `src/ops/attention.cpp:69-94`
  takes the `with_lse` branch whenever `saved_tensors[3]` (LSE) exists,
  and the forward with `requires_grad` always saves LSE on native CUDA
  (`gpu_flash_attention_forward_save_lse` returns 1 without FA3,
  `flash_attention.cu:1172-1188`). The `else` branch at :94 is the only
  caller of the non-LSE symbol from the autograd node, and it is
  unreachable from Python on native CUDA. The vehicle calls the symbol
  directly, bypassing the autograd node entirely.

## Files

- `softmax-fwd-vehicle.cpp` (316 lines, SHA256
  `4a38e4245d3e901ad26badb6203f753a97505ef2b15a3464c513cfff6a1c2122`).
  Standalone program, no comments. argv: `Tk causal half outdir
  [liblitetorch_gpu.so]`. Fixed: B=1, H=2, H_kv=2, D=8, Tq=Tk,
  scale=1/sqrt(8). Deterministic xorshift64star inputs in [-1,1), seeds
  1000+Tk (q/k/v) and 2000+Tk (dO); half inputs rounded with
  `__float2half`. Writes `q/k/v/do/dq/dk/dv.bin` (raw float32 or float16
  bits) plus `meta.txt` into the case dir. Exit codes: 0 ok, 1 usage/IO,
  2 CUDA error, 3 dlopen/dlsym/`gpu_init` failure, 4 FA3 present
  (NOT_COVERED, never PASS).
- `softmax-fwd-vehicle-compare.py` (170 lines, SHA256
  `5e684086b6dafec5b73186b0955cc7e8ea69316b2ee64283a3964ec0b5534e98`).
  No comments. Usage: `compare.py <casedir>` or
  `compare.py --all <rootdir>` (expects case dirs named
  `Tk{Tk}-causal{c}-half{h}`). Reads the vehicle output files, computes
  the torch SDPA-autograd reference on CPU, prints per-case dq/dk/dv
  max_abs_diff + PASS/FAIL, and a final summary line. Tolerances exactly
  per the verdict: FP32 <= 1e-4 vs torch FP32; FP16 <= 1e-4 vs torch in
  FP16 OR <= 1e-3 vs torch FP32, with the reference used printed. Exit 0
  only if every case passes; torch absent prints TORCH-ABSENT, exit 1.

## Verification done on this box (no GPU, no nvcc)

- `py_compile` on the compare script: OK. 0 `#` lines in the .py, 0
  `//` / `/*` in the .cpp, no TODO/FIXME.
- `.cpp` host syntax checked with `g++ -fsyntax-only` against minimal
  stub CUDA headers (real `cuda_runtime.h` / `cuda_fp16.h` absent here):
  OK. This checks host-side syntax only, not CUDA API behavior; the
  Kaggle build + run is the real proof.
- Compare script tested end to end on CPU (torch 2.14.1+cpu) with
  synthetic vehicle-format case dirs: FP32 Tk=4 causal=0 diffs 0.0 PASS
  exit 0; FP16 Tk=16 causal=1 diffs 0.0 vs FP16 ref (ref=FP16) PASS exit 0
  (vs FP32 ref the same case reads 6.4e-05/8.4e-05/3.4e-04, inside the
  1e-3 FP32 allowance — the dual-reference design matters); corrupted dq
  (+0.5) diff 0.5 FAIL exit 1; `--all` on a 2-of-24 root reports the 22
  missing dirs FAIL and exits 1.

## Kaggle run sequence (GPU T4 x2, same session as the rest of the queue)

1. Push to `kaggle-run-inputs` first (QA direct push, diagnostics branch):
   the softmax patch `P-BACKEND-softmax-warp-broadcast.patch` (SHA
   `cc3abc22123a274400015098c38824186664ef081ff0e19f594fbfc4c0ebce7f`),
   the Python gate script, and both files in this directory. Re-fetch
   the branch in-session and re-verify HEAD + every SHA before use.
2. Clone `agent-v1` @ `f9ce6609`, apply the softmax patch, verify
   `src/backend/gpu_native/nn/flash_attention.cu` matches the patched
   file (byte-identical check), then `python3 setup.py build_ext
   --inplace` (check/install the CUDA toolkit first; gate on the .so
   existing + zero `error:` lines, not on the printed BUILD OK alone;
   copy `liblitetorch_gpu.so` next to the extension under `python/`).
   Pre-flight: assert the `Native GPU (CUDA/HIP)` init line, not OpenCL.
3. Compile the vehicle (nvcc is present on the Kaggle image):
   `nvcc -O3 -std=c++14 -o softmax-fwd-vehicle softmax-fwd-vehicle.cpp -ldl`
   (the vehicle needs only the CUDA runtime + dlopen; it does not link
   cuBLAS — the .so does that internally).
4. Run the matrix (B=1, H=2, D=8, Tq=Tk), one case dir per combination:
   `for Tk in 1 4 16 32 33 64; do for c in 0 1; do for h in 0 1; do
   ./softmax-fwd-vehicle $Tk $c $h /kaggle/working/fwdveh/Tk$Tk-causal$c-half$h
   $LITETORCH_TREE/python/liblitetorch_gpu.so; done; done; done`
   Any vehicle exit != 0 stops that case: record it FAIL, do not continue
   past a CUDA/load error, do not waive. Exit 4 (FA3 present) means the
   whole FWD vehicle is NOT_COVERED for that session, not a PASS.
5. Compare: `python3 softmax-fwd-vehicle-compare.py --all
   /kaggle/working/fwdveh`. PASS = all 24 cases PASS. Any case above its
   tolerance = NO LAND for the softmax patch.
6. Negative control (required, proves the vehicle actually exercises the
   patched kernels): repeat steps 2-5 on the UNPATCHED tip tree.
   Expected: warp cases Tk in {4, 16, 32} FAIL on the unpatched tree,
   Tk=1 may PASS on either tree (degenerate: single valid column, buggy
   == fixed), Tk in {33, 64} PASS on either tree (fused control). If the
   unpatched tree PASSES a Tk in {4, 16, 32} case, the vehicle is not
   discriminating — stop and investigate, do not claim a gate PASS.

## Caveats

- The vehicle exercises the fwd warp kernels only as the P-recompute step
  inside the non-LSE backward, judged through dq/dk/dv parity vs torch.
  It does not test a standalone forward softmax op, because no such
  Python entry point exists and the .so exposes no standalone forward
  softmax symbol. P itself is internal scratch (`g_attn_scratch_p`) and
  is not exported; grad parity is the observable.
- The 24 BWD cases from the Python gate script still run separately; this
  vehicle is the FWD half of gate 3, not a replacement for the BWD half.
- The vehicle runs on device 0 only (single-GPU path; the kernels are
  per-device and the gate matrix needs no multi-GPU path).
- No nvcc exists on the authoring box, so the .cpp has never been
  compiled with real CUDA headers; a Kaggle compile error is possible
  and must be fixed in the vehicle, not by editing the product tree.
