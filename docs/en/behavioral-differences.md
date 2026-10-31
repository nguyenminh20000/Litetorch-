# Behavioral Differences vs PyTorch

LiteTorch aims for PyTorch-compatible numerics, but it is not PyTorch. This page is the canonical record of every verified behavioral difference between LiteTorch and PyTorch: what LiteTorch does (with the source file and line), what PyTorch does (live-probed against torch 2.14.1+cpu unless marked otherwise), the impact, and the migration or workaround. A difference listed here as OPEN is real and present in the tree; a difference listed as FIXED is resolved and kept here as history so old code and old docs can be reconciled.

- **Baseline:** tree HEAD `b326f29`; examples in this page run against the Core D3 extension `build_commit 88a4ba1` (`workspace/framework/core/litetorch-d3/python`); PyTorch reference torch 2.14.1+cpu in `~/torch-venv`.
- **Status key:** `OPEN` — present in the tree, no fix merged. `FILED` — defect filed to the owning team via the comms board, fix not yet merged. `FIXED` — fixed and merged; the old behavior is recorded as history only. `DOCUMENTED` — intentional divergence, no fix planned; the doc is the contract.
- **Numbering:** D-01–D-21 were verified in the B0 plan pass (`plan/differences.md`). D-22–D-29 were found in the B3 nn reference pass. D-30–D-46 were found in the B3 optim/linalg/data/distributed passes and are assigned in this batch (B4). D-47–D-49 (autograd) were found in the B4 explanations pass and added by the leader after the front2 worker finished. D-50 (device string) was found in the B5 GPU verification pass (e1-9 failure) and added as a follow-up.
- **Rule for porters:** never assume a PyTorch idiom works in LiteTorch until this page or the API reference says so. When in doubt, run the parity check in `plan/verification.md` §3.

## Live demonstrations

Three behaviors below are demonstrated live on the verification extension, so the reader can see the difference instead of taking it on trust.

<!-- ex: id=bd-d01-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

cpu = lt.Device("cpu")
t_int = lt.Tensor.from_vector([1, 2, 3], [3], cpu, False).cast(lt.DataType.INT32)
print("hasattr data_ptr:", hasattr(t_int, "data_ptr"))
print("hasattr raw_data_ptr:", hasattr(t_int, "raw_data_ptr"))
```

<!-- ex: id=bd-d02-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0], [2], cpu, False)
t2 = t.to(lt.Device("cuda"))
print("requested cuda, got device:", t2.device)
print("cuda available:", lt.cuda.is_available())
```

<!-- ex: id=bd-d09-1 device=cpu backend=cpu runtime=~51s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([-3.0, -1.0, 0.0, 1.0, 3.0], [5], cpu, False)
g_lt = lt.Ops.gelu(x).to_vector()
xt = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
g_exact = torch.nn.functional.gelu(xt).tolist()
g_tanh = torch.nn.functional.gelu(xt, approximate="tanh").tolist()
print("max |lt - torch exact|:", max(abs(a - b) for a, b in zip(g_lt, g_exact)))
print("max |lt - torch tanh|:", max(abs(a - b) for a, b in zip(g_lt, g_tanh)))
```

## D-01 — `data_ptr()` throws on non-FP32 tensors (C++); unbound in Python — OPEN

- LiteTorch (VERIFIED): `Tensor::data_ptr()` throws `runtime_error("[litetorch Error] data_ptr(): tensor dtype is not FP32; use raw_data_ptr() for byte-level access")` for any dtype != FP32 (`src/tensor/tensor_core.cpp:1292-1297`). `raw_data_ptr()` returns `void*` for any dtype (`tensor_core.cpp:1300-1303`). Both are declared in `include/litetorch/tensor.h:185-186`.
- **Correction (B4, verified live on the 88a4ba1 extension): neither method is bound to Python.** `hasattr(t, "data_ptr")` and `hasattr(t, "raw_data_ptr")` are both `False`; there is no `.def("data_ptr")` in `src/bindings/`. The throw behavior above applies to C++ callers only. Python code calling `t.data_ptr()` gets `AttributeError`, not the documented runtime error.
- PyTorch: `Tensor.data_ptr()` works for every dtype and returns the raw address.
- Impact: HIGH. Any ported code doing `t.data_ptr()` on int/bool/FP16/FP64 tensors crashes; on the Python side the method does not exist at all. Also note `data_ptr()` returns a CPU pointer (`storage->get_cpu_ptr() + offset`) even for GPU tensors — it does not return a device pointer.
- Migration: in C++, use `raw_data_ptr()` for byte-level access and cast to the element type yourself. In Python, there is no equivalent — use `.numpy()` (see D-14) or `.to_vector()` to read data out.
- See the live example `bd-d01-1` above.

## D-02 — `to(device)` silently falls back to CPU when the accelerator is unavailable — OPEN

- LiteTorch (VERIFIED): `Tensor::to(Device)` builds a `StorageImpl` on the target device; if the GPU backend is absent the storage silently lands on CPU and `final_device` is rewritten to CPU with no error, no warning, no log (`src/tensor/tensor_core.cpp:997-1006`; root cause `src/tensor/storage.cpp:31-33`). Same for TPU (`storage.cpp` TPU branch), and a TPU allocation failure ALSO silently falls back to CPU, while a GPU OOM throws. `Tensor::cuda(idx)` goes through the same path (`tensor_core.cpp:2125-2128`), so `.cuda()` on a GPU-less box silently returns a CPU tensor. `Module.to()` inherits the behavior per parameter (`src/nn/container.cpp:363`).
- PyTorch: `.to("cuda")` / `.cuda()` on a box without CUDA raises (probe: `AssertionError: Torch not compiled with CUDA enabled`).
- Impact: HIGH. Silent wrong-device execution: code believes it runs on GPU, actually runs on CPU; benchmarks and training silently slow; device asserts downstream fail confusingly.
- Migration: always check `t.device` after `.to()` / `.cuda()` and fail loudly if it is not the requested device.
- See the live example `bd-d02-1` above.

## D-03 — `.to()` / `Module.to()` accept no string form; malformed device strings silently become CPU — OPEN

- LiteTorch (VERIFIED): `Tensor.to` is bound only for `Device`, `DataType`, `(Device, DataType)` (`src/bindings/bind_tpu.cpp:73-75`). `Module.to` takes only `Device` (`src/bindings/bind_nn_base.cpp:10`; `src/nn/container.cpp:363`). Passing a string raises a pybind TypeError. `Device("...")` parses "cuda"/"gpu"/"tpu"/"meta" (case-insensitive, optional `:index`) and returns CPU for ANYTHING else — including typos such as `Device("cude")` — with no error (`src/bindings/bind_core.cpp:23-43`).
- PyTorch: `.to("cuda")`, `.to("cpu")`, `.to(torch.float32)` all work; an invalid device string raises.
- Impact: MEDIUM. Ported `model.to("cuda")` / `t.to("cuda:0")` code fails at the call site; a typo'd device string silently runs on CPU (compounds D-02).
- Migration: construct `litetorch.Device("cuda")` explicitly and verify `t.device` afterwards.

## D-04 — Factory functions take a single shape sequence; no variadic form — OPEN

- LiteTorch (VERIFIED): `litetorch.randn(shape, ...)`, `zeros`, `ones` take `const std::vector<int64_t>& shape` only (`src/bindings/bind_data.cpp:175`; `src/bindings/bind_tpu.cpp:37`). `randn(2, 3)` raises.
- PyTorch: `torch.randn(2, 3)` and `torch.randn([2, 3])` both work (probe).
- Impact: LOW. Mechanical porting friction.
- Migration: always pass a list/tuple: `litetorch.randn([2, 3])`.

## D-05 — No `Tensor.pow()` method; `**` and `Ops.pow` take a scalar exponent only — OPEN

- LiteTorch (VERIFIED): no `.def("pow")` on Tensor. `__pow__` exists but takes only a `double` (`src/bindings/bind_tpu.cpp:134`). `Ops.pow(a, exponent)` takes a `float` exponent (`src/bindings/bind_ops.cpp:19`; `include/litetorch/ops.h:35`). A two-tensor `pow(a, b)` is forward-declared (`src/ops/elementwise.cpp:124`) but never defined or bound.
- PyTorch: `t.pow(exp)`, `t ** exp`, and `t ** u` (tensor exponent) all work (probe: `tensor([2.]) ** tensor([3.]) == [8.]`).
- Impact: LOW-MEDIUM. `t ** u` and `t.pow(u)` fail; `t.pow(2.0)` must be rewritten.
- Migration: use `litetorch.Ops.pow(t, float_exp)`.

## D-06 — `.long()` returned INT32, not INT64 — FIXED

- Pre-fix behavior (VERIFIED, now historical): the binding mapped `long` to `Tensor::to_int` (INT32). The correct C++ method `Tensor::to_long()` (INT64) existed (`src/tensor/tensor_core.cpp:2141`) but was not bound under any name. `.int()` -> INT32 is correct.
- Post-fix behavior (VERIFIED): `.def("long", &Tensor::to_long)` (`src/bindings/bind_tpu.cpp:155`) — `.long()` now returns INT64, matching torch. Fix: P-CORE-D06-long-binding REVISED, QA APPROVE qa-r62 2026-10-26 ~23:10, MERGED agent-v1 @89afd67. Test evidence: new test 12/12 PASS + whole `tests/test_parity_tensor.py` 24 passed against the patched build; negative control on the unpatched extension: exactly 4 FAILs, 2**31 wrapping to -2147483648.0.
- PyTorch: `.long()` is INT64 (probe: `torch.tensor([1.5, -2.5, 3.0]).long()` -> `torch.int64`).
- Impact: RESOLVED. The old migration line ("use `.cast(...)` to INT64 until the binding is fixed") is OBSOLETE and must not appear anywhere in these docs.
- Migration (current): none — `.long()` works as in torch. NOTE the interplay with D-14 (still OPEN, NOT fixed by this patch): `.numpy()` rejects INT64, so `.long().numpy()` throws "unsupported dtype for copy conversion" — use `.to_vector()` or cast first.
- Regression check: L-05 (`.long()` output now INT64 used as indices in `gather` / `index_select`).

## D-07 — Linalg ops promoted everything to FP32 — FIXED

- Pre-fix behavior (VERIFIED, now historical): `promote_dtype(a, b)` in `src/ops/linalg.cpp:41-44` returned FP64 if either input was FP64, else FP32 ALWAYS — for integer inputs too. Affected ops: `solve`, `lstsq`, `cross`, `vecdot`, `outer`, `inner`, `kron`, `vdot`.
- Post-fix behavior (VERIFIED): `promote_dtype` now delegates to the central `promote_types` (torch-aligned). `solve`/`lstsq`/`vecdot` on int/bool/FP16/mixed inputs now THROW instead of silently returning FP32; `cross`/`inner`/`vdot` preserve same-dtype; `outer`/`kron` use full `result_type` promotion; `lstsq` rank is now INT64 (was FP32); integer paths are exact in int64/uint64 with defined wraparound and no double round-trip. Fix: P-CORE-D07-linalg-promotion, QA APPROVE 2026-10-26 ~21:48, MERGED agent-v1 @7b8981e. Test evidence: 62 CHECKs, 38 promotion-sensitive failures pre-fix, ALL PASS post-fix; 16-binary regression sweep exit 0.
- PyTorch: `outer`, `vdot`, `kron` on int64 inputs return int64 (probe). The fixed behavior matches torch.
- Impact: RESOLVED for these 8 ops. The old migration line ("cast the result back to the expected integer dtype; do not rely on linalg output dtype") is OBSOLETE and must not appear anywhere in these docs.
- Migration (current): cast int/bool/FP16/mixed inputs to a float dtype before calling `solve`/`lstsq`/`vecdot`, or expect a throw; `outer`/`kron`/`inner`/`vdot`/`cross` now follow torch promotion — no cast needed.
- Correction to a prior report: the "uint8+int64 promotes to FP32" claim was LOCALIZED, not global. The elementwise promotion table `promote_types` (`src/tensor/tensor_core.cpp:555`) is CORRECT: uint8+int64 -> INT64, matching torch. The FP32 promotion happened only in the linalg family.
- Regression check: L-02 (numeric max-diff vs torch for these 8 ops, expecting parity).

## D-08 — Integer tensor + Python int scalar promotes to FP32 — OPEN

- LiteTorch (VERIFIED): `__add__` / `__sub__` / `__mul__` scalar overloads materialize the scalar as an FP32 1-element tensor (`src/bindings/bind_tpu.cpp`: `Tensor::from_vector({(float)b}, {1}, ...)`), so promotion yields FP32.
- PyTorch: `int64_tensor + 2` stays int64; `int64_tensor + 2.5` becomes float32 (probe).
- Impact: MEDIUM. Silent dtype change on the most common scalar-arithmetic idiom; int64 values above 2^24 are rounded through FP32.
- Migration: build an explicit int64 scalar tensor for integer scalar arithmetic; or cast the result back to the input dtype.

## D-09 — GELU is tanh-approximation only; no `approximate` argument — DOCUMENTED

- LiteTorch (VERIFIED): `Ops::gelu(a)` has no `approximate` parameter (`include/litetorch/ops.h:79`). Forward and backward both use the tanh approximation `0.5x(1+tanh(0.79788456(x+0.044715x^3)))` (`src/ops/activation.cpp:753-790`; backward at `activation.cpp:419-420`). No `erf` appears in the GELU path.
- PyTorch: default GELU is the exact erf form; `approximate='tanh'` is opt-in (probe: max |exact - tanh| = 4.4e-4 on linspace(-3, 3, 13); live re-measurement in this batch: 4.13e-04, and |lt - torch tanh| = 0.0 exactly).
- Impact: LOW-MEDIUM. Numerics differ from torch default by up to ~4.4e-4 absolute; parity tests must use a tolerance of at least ~5e-4 for GELU, or compare against `F.gelu(x, approximate='tanh')`.
- Migration: none — document the tolerance; do not claim exact-erf parity.
- See the live example `bd-d09-1` above.

## D-10 — Binary ops across devices silently transfer instead of raising — OPEN

- LiteTorch (VERIFIED): `add`/`sub`/`mul`/`div` move the mismatched operand to the other operand's device (`src/ops/elementwise.cpp:583, 723, 874, 1015`); same pattern in comparisons (`elementwise.cpp:2928, 2953`), matmul/linear (`src/ops/linear.cpp:142, 396, 474`), and shape ops (`src/ops/shape_ops.cpp:75`).
- PyTorch: `cpu_tensor + cuda_tensor` raises "Expected all tensors to be on the same device".
- Impact: MEDIUM. Hides device-placement bugs; a stray CPU tensor silently triggers a device transfer (and, combined with D-02, can silently pin a whole computation to CPU).
- Migration: move tensors explicitly before mixing devices; treat any implicit transfer as a bug to be found, not a feature.
- Live check still needed: L-01 (which operand moves in each op family, and the behavior when the GPU side is the fallback CPU tensor — Kaggle GPU).

## D-11 — Indexing: only int / slice on dim 0; no boolean, tensor, or multi-dim indexing — OPEN

- LiteTorch (VERIFIED): `Tensor.__getitem__` handles `int` and `slice` on dimension 0 only and raises `TypeError("unsupported index type")` otherwise (`src/bindings/bind_tpu.cpp`: `__getitem__` lambda).
- PyTorch: boolean masks, integer tensor indices, and multi-dim tuples all work (probe: `t[torch.tensor([True, False, True])] == [1, 3]`).
- Impact: MEDIUM. Common ported idioms (`x[mask]`, `x[:, 1]`, `x[idx]`) fail.
- Migration: use `Ops.gather`, `Tensor.index_select`, `Tensor.slice` (see API reference).

## D-12 — `Tensor.sum/mean/max` take a single int dim; no tuple of dims — OPEN

- LiteTorch (VERIFIED): `Tensor.sum(dim, keepdim)` casts `dim` to a single `int64_t` (`src/bindings/bind_tpu.cpp`). `Ops.sum/mean/max` DO accept a `vector<int64_t>` of dims (`src/bindings/bind_ops.cpp:121-125`).
- PyTorch: `t.sum(dim=(0, 1))` works (probe).
- Impact: LOW.
- Migration: use `litetorch.Ops.sum(t, [0, 1])`.

## D-13 — No `.bool()` / `.byte()` / `.char()` / `.short()` / `.to_long()` methods — OPEN

- LiteTorch (VERIFIED by absence): none of these are bound on Tensor (`src/bindings/bind_tpu.cpp` has only `half`, `float`, `double`, `long`->INT64 (fixed, D-06), `int`, `type_as`, `cast`). `.cast(dtype)` exists and is the general path.
- PyTorch: all exist (probe).
- Impact: LOW. Mechanical.
- Migration: use `.cast(litetorch.DataType.<X>)`.

## D-14 — `.numpy()` rejects UINT8 / INT64 / BOOL; BF16 is silently cast to FP32 — OPEN

- LiteTorch (VERIFIED): the `numpy()` conversion switch handles FP64/FP32/FP16/INT32/INT16/INT8 only; UINT8, INT64, BOOL hit the default and throw "unsupported dtype for copy conversion" (`src/bindings/bind_tpu.cpp`: `numpy` lambda). BF16 is cast to FP32 before conversion. Non-CPU tensors throw (must `.cpu()` first).
- PyTorch: `.numpy()` supports uint8 and int64 directly (probe).
- Impact: LOW-MEDIUM. Data-export code paths fail on the most common integer dtypes.
- Migration: `.cast(FP32)` first (precision caveat for int64 above 2^24); or `.tolist()`.

## D-15 — `arange` is integer-only; values are materialized as FP32 — OPEN

- LiteTorch (VERIFIED): `Tensor::arange(int64_t start, int64_t end, int64_t step, ...)` builds a `vector<float>` then casts to the requested dtype (`src/tensor/tensor_core.cpp:696-708`). No float start/end/step.
- PyTorch: `torch.arange(0, 5, 0.5)` returns float32 (probe).
- Impact: LOW.
- Migration: build float ranges via `from_vector` / `linspace` equivalents (see API reference).

## D-16 — `clamp` takes scalar bounds only — OPEN

- LiteTorch (VERIFIED): `clamp(a, float min_val, float max_val)`, `clamp_min/max(a, float)` (`include/litetorch/ops.h:75, 201-202`).
- PyTorch: `torch.clamp` accepts tensor min/max (probe).
- Impact: LOW.
- Migration: use `Ops.where` / `Ops.minimum` equivalents where tensor bounds are needed.

## D-17 — Unary / shape ops exist only as `Ops.*` functions, not Tensor methods — OPEN

- LiteTorch (VERIFIED by absence): Tensor has no `.exp()`, `.log()`, `.sqrt()`, `.relu()`, `.sigmoid()`, `.tanh()`, `.abs()`, `.squeeze()`, `.unsqueeze()` methods (only `log_softmax` and `silu` are bound as methods). All exist as `litetorch.Ops.*` (`src/bindings/bind_ops.cpp:20-32, 46-52`).
- PyTorch: all exist as Tensor methods.
- Impact: LOW. Mechanical porting friction.
- Migration: `litetorch.Ops.relu(t)` etc.

## D-18 — `.item()` and `.tolist()` / `.to_vector()` return float; int64 precision loss — OPEN

- LiteTorch (VERIFIED): `Tensor::item()` returns `float`, casting non-FP32 tensors to FP32 first (`src/tensor/tensor_core.cpp:1598-1608`). `to_vector()` likewise casts to FP32; `tolist()` builds from those doubles and only treats INT32/INT16/INT8 as integers (`src/bindings/bind_tpu.cpp`).
- PyTorch: `.item()` on an int64 tensor returns a Python `int` with full precision (probe: `2**60` round-trips).
- Impact: LOW-MEDIUM. int64 values above 2^53 silently lose precision through `.item()` / `.tolist()`.
- Migration: avoid `.item()` for large int64 values; read via `.numpy()` (after casting, see D-14) or keep values in tensor form.
- Live check: L-02 (int64 `.item()` boundary at 2^53).

## D-19 — Quantized dtypes promote to FP32 — OPEN (TREE-ONLY)

- LiteTorch (TREE-ONLY): `promote_types` returns FP32 if either operand is a quantized dtype (`src/tensor/tensor_core.cpp:555-557`).
- PyTorch: quantized promotion follows torch's quantized rules; the exact correspondence is UNVERIFIED — needs a live side-by-side probe.
- Impact: UNKNOWN until probed. Flagged so the quantization chapter does not silently assert torch parity.
- Migration: none yet.
- Live check: L-03 (D-19 quantized promotion vs torch semantics).

## D-20 — No `F.scaled_dot_product_attention`; only `Ops.flash_attention` — OPEN

- LiteTorch (VERIFIED): no SDPA anywhere in bindings. `Ops.flash_attention(q, k, v, causal=False)` and `flash_attention_qkv(qkv, num_heads, causal=False)` only (`src/bindings/bind_ops.cpp:143-144`). No `scale` or `dropout_p` arguments; head dim must satisfy 1 <= D <= 256 and H % H_kv == 0 (`src/backend/gpu_native/nn/flash_attention.cu:167-168`). TPU `flash_attention_half` is not implemented and throws (`src/backend/tpu/tpu_backend.cpp:343`).
- PyTorch: `F.scaled_dot_product_attention` exists with `is_causal`, `dropout_p`, `scale` (probe).
- Impact: MEDIUM. Ported attention code must be rewritten; scale must be applied to Q manually; FP16 attention on TPU is unavailable.
- Migration: `litetorch.Ops.flash_attention(q * scale, k, v)`; check head-dim constraints before porting.
- Live check: L-04 (`flash_attention_half` throw confirmation on TPU).

## D-21 — GPU integer elementwise math runs through FP32 — OPEN (TREE-ONLY, numeric)

- LiteTorch (TREE-ONLY): non-FP32 elementwise inputs on GPU are cast to FP32, computed, and cast back to the promoted dtype (`src/ops/elementwise.cpp` `add` path, ~lines 571-590; same pattern in sub/mul/div). The result dtype is correct, but int64 arithmetic is done in FP32.
- PyTorch: int64 elementwise math on CUDA is exact integer math.
- Impact: MEDIUM on GPU. int64 values above 2^24 lose precision; integer overflow wraps in torch but saturates/rounds in the FP32 path. CPU integer path (`ew_add_int`) is exact.
- Live check: L-01 (measured max error for int64 add/mul on Kaggle GPU vs torch).

## D-22 — Multi-arg `__call__` routes only the single-tensor forward — FILED

- LiteTorch (VERIFIED): `__call__()` only routes the single-tensor forward. `m(q, k, v)`, `m(x1, x2)` and every 2+-arg loss raise TypeError; must use `.forward(...)`. Affects MultiHeadAttention, Bilinear, TripletMarginWithDistanceLoss, GaussianNLLLoss, CTCLoss, DPOLoss, all 2-arg losses.
- PyTorch: modules are callable with their full forward signature.
- Impact: MEDIUM. Ported code that calls a module directly with multiple arguments fails at the call site.
- Migration: call `.forward(...)` explicitly for multi-argument modules.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-23 — Loss targets must be INT32; INT64 rejected — FILED

- LiteTorch (VERIFIED): CrossEntropyLoss, NLLLoss, `F.nll_loss` reject INT64 targets ("cross_entropy_loss: target must have integer or float dtype") and require INT32. Root cause: `ce_target_indices` in `src/ops/loss.cpp:24-40` has no INT64 case; same in `fused_loss.cpp:25`. Also affects `Ops.cross_entropy_loss` / `Ops.nll_loss` / `Ops.fused_linear_cross_entropy` and `nn::CrossEntropyLoss`.
- PyTorch: loss targets are int64 (probe).
- Impact: HIGH. After the D-06 fix `.long()` returns INT64 — which these losses reject. The two fixes interact: the correct dtype for a loss target in LiteTorch today is INT32, the opposite of torch.
- Migration: use INT32 targets explicitly (`.cast(litetorch.DataType.INT32)`); do not pass `.long()` results as loss targets.
- Fix status: FILED to Core, comms `20261026-234500-docs-to-core-ceint64.md` (thread `docs-fix-requests-20261026`).

## D-24 — Assigning a submodule as a Python attribute does NOT register it — FILED

- LiteTorch (VERIFIED): assigning a submodule as a Python attribute (`self.linear = lt.nn.Linear(4, 4)`) does NOT register it — `parameters()` returns 0 and `state_dict()` is empty. `add_module('linear', module)` DOES register (`parameters() == 2`).
- PyTorch: attribute assignment registers submodules automatically.
- Impact: HIGH. Every ported `nn.Module` subclass that assigns layers as attributes silently has no parameters; the optimizer then steps over nothing and training does not learn.
- Migration: register children explicitly with `add_module(name, module)`; or override `parameters()` to collect submodules manually (the T1 tutorial's `MultiLayerPerceptron` does this). `ModuleDict` entries go via `m[key] = module` / `put()`, NOT `add_module()` (which registers a child but not a dict entry — `getitem` then raises KeyError); `ModuleDict` has no `values()`/`items()` methods bound.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-25 — `Parameter(tensor)` construction is BROKEN — FILED

- LiteTorch (VERIFIED on 88a4ba1): `Parameter(tensor)` always throws "copy_: a leaf Tensor that requires grad is being used in an in-place operation", even for requires_grad=False inputs. The binding `copy_()`s into the Parameter itself.
- PyTorch: `nn.Parameter(tensor)` works.
- Impact: HIGH. No workaround exists; obtain parameters from modules (for example `Linear.weight`) instead of constructing `Parameter` directly.
- Migration: none — do not construct `Parameter` directly; use module parameters.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-26 — HingeEmbeddingLoss formula differs — FILED

- LiteTorch (VERIFIED): computes mean(max(0, margin - target*input)) for all elements (1.1750 on the reference example).
- PyTorch: computes input for target==1, max(0, margin-input) for target==-1 (0.7000 on the same example).
- Impact: MEDIUM. Same inputs, different loss values; training trajectories diverge.
- Migration: none — do not compare HingeEmbeddingLoss values across frameworks; if exact torch semantics are needed, implement the loss with `Ops` primitives.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-27 — MultiMarginLoss normalization differs; `p` is an INT — FILED

- LiteTorch (VERIFIED): single-sample all-zeros case LiteTorch 1.0000 vs torch 0.7500 (torch divides by #classes, LiteTorch by #classes-1); two-sample example 0.9667 vs 0.6000 (p=1); p=2: 1.1700 vs 0.7525. Also: `p` is an INT in this binding (`p=2.0` raises TypeError).
- PyTorch: `p` is a float; normalization divides by #classes.
- Impact: MEDIUM. Loss values differ; `p=2.0` (the torch default spelling) raises.
- Migration: pass `p` as an int (`p=2`); do not compare MultiMarginLoss values across frameworks.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-28 — CTCLoss mean reduction differs — FILED

- LiteTorch (VERIFIED): default mean = mean over batch of per-sample losses (4.0802 on the reference example).
- PyTorch: 'mean' also divides each by its target length (1.3601 on the same example). LiteTorch reduction='sum' matches torch 'sum' exactly (8.1604).
- Impact: MEDIUM. Mean-reduced CTC losses differ by the target-length factor.
- Migration: use reduction='sum' for exact parity, and divide by total target length manually if torch-mean semantics are needed.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-29 — TransformerDecoderLayer output differs from torch — FILED

- LiteTorch (VERIFIED): with identical weights copied from LiteTorch, the two-argument forward differs from torch (max diff 0.12–0.14, eval mode). Cause not isolated (possibly norm placement / cross-attention detail). TransformerEncoderLayer parity holds in eval mode only (4.2e-07); in train mode dropout 0.1 is active, so parity claims for the encoder layer are eval-mode only.
- PyTorch: reference behavior.
- Impact: HIGH for anyone porting a transformer decoder. Decoder outputs do not match torch even with identical weights.
- Migration: none — do not expect decoder parity; verify decoder outputs against torch before trusting them.
- Fix status: FILED to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-30 — NAdam update is ~6.5x too large on the first step — FILED

- LiteTorch (VERIFIED): first-step update ~6.5x too large. Root cause in `src/optim/nadam.cpp:49`: `mu_product = mu_next / mu` (a ratio) + `step_size = lr / bias_correction1`; torch keeps `mu_product` as the running product from 1 and uses two update terms with `(1 - mu_product)` denominators. Measured 3-step: lt `[0.9350, 0.8961, 0.8658]` vs torch `[0.9894, 0.9816, 0.9743]`.
- PyTorch: `torch.optim.NAdam` — the reference values above.
- Impact: HIGH. NAdam training in LiteTorch diverges from torch immediately; do not use NAdam for ported training runs.
- Migration: use Adam or AdamW instead; or implement the NAdam update manually.
- Fix status: FILED to Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-31 — OneCycleLR has a one-step phase shift — FILED

- LiteTorch (VERIFIED): from call 3 onward lt == torch's previous call; ramp takes 3 calls to max vs torch 2; final call lt lands on the floor (4e-07) while torch re-emits the previous value. Scheduler source: `src/optim/scheduler.cpp` (binding `src/bindings/bind_optim.cpp`).
- PyTorch: `torch.optim.lr_scheduler.OneCycleLR` — the reference schedule above.
- Impact: MEDIUM. Learning-rate schedules are shifted by one step; short runs see a visibly different schedule.
- Migration: none — account for the one-step shift when comparing schedules, or use CosineAnnealingLR / StepLR (both verified to match torch exactly).
- Fix status: FILED to Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-32 — NegativeBinomial `probs` is the failure probability — FILED

- LiteTorch (VERIFIED): treats `probs` as failure probability; `NB(r, p)` in lt == `NB(r, 1-p)` in torch exactly. Mean 11.667 vs torch 2.143 (r=5, p=0.3). Binomial/Geometric are NOT affected.
- PyTorch: `probs` is the success probability (probe).
- Impact: MEDIUM. Distribution samples and means are swapped relative to torch.
- Migration: pass `1 - p` to get torch semantics.
- Fix status: FILED to Core, comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-33 — DDP backward doubles one slice of the weight grad — FIXED

- Pre-fix behavior (VERIFIED, now historical): `DistributedDataParallel.backward` doubles one slice of the flat weight grad: indices `[numel/4, numel/4 + out_features)` (equivalently `[numel/4, numel/4 + bias_numel)`; settled 2026-10-27 by dist-w21's decisive re-run — a same-day correction claiming in_features was withdrawn: the B4 re-check had only re-run symmetric Linear(4,4), and the B3 shape table's asymmetric rows were transposed). Verified on 4 shapes — Linear(4,4): indices 4–7; Linear(4,8): 8–15; Linear(8,4): 8–11; Linear(2,2): 1–2 — in both `AVG` (doubled row 1.0) and `SUM` (doubled row 2.0) reduce modes. Bias grad is correct. Single-process `Linear.backward` is correct (all 0.5). Parity vs torch: max_abs_diff 0.5 on both ranks.
- PyTorch: DDP gradients match single-process gradients.
- Impact: RESOLVED. The pre-fix impact was HIGH: weights received a doubled gradient slice every step.
- Migration (current): none — DDP training verified by the post-merge gate below; the single-process/FSDP workaround applies to pre-fix trees only.
- Fix status: FIXED. Fix: dist-w20 (2-line memcpy fix), MERGED agent-v1 @39e53ee0. Post-merge gate: Track 5 PASS 2026-10-31 (PIN c195f43c, Kaggle 2x T4, cell-20 happy + mismatch-throw, both ranks, NCCL pip 2.31.2). Filed history: comms `20261027-000800-docs-to-core-ddp-grad.md` (thread `docs-fix-requests-20261026`).

## D-34 — `gradcheck` default `eps=1e-6` fails on FP32 — DOCUMENTED

- LiteTorch (VERIFIED): `lt.autograd.gradcheck` with default `eps=1e-6` fails even for `t*t` (analytical 6 vs numerical 5.72 at x=3): 1e-6 is below FP32 resolution. `eps=1e-3` passes gradcheck + gradgradcheck. Default binding: `src/bindings/bind_autograd.cpp:79` (`eps` = 1e-6).
- PyTorch: `torch.autograd.gradcheck` defaults to `eps=1e-6` as well, but torch's default input dtype for gradcheck is float64 — the same absolute eps is meaningful there.
- Impact: LOW. Affects verification code, not training.
- Migration: always pass `eps=1e-3` explicitly for FP32 gradcheck in LiteTorch.
- See `troubleshooting.md` ("gradcheck fails on a trivial function") for the runnable demonstration.

## D-35 — RendezvousStore aborts at shutdown unless the `run()` thread is joined — OPEN

- LiteTorch (VERIFIED): without joining the `run()` thread after `stop()`, the process aborts at interpreter exit (`terminate called without an active exception`, SIGABRT; flaky — the EN run failed, the VI run passed). Joining the thread fixes it (3/3 clean). Source: `src/distributed/rendezvous.cpp:113-143` (`RendezvousStore::stop`, destructor).
- PyTorch: no equivalent component (`torch.distributed` rendezvous is a different mechanism).
- Impact: MEDIUM. Flaky SIGABRT at exit in distributed launcher code.
- Migration: always `join()` the `run()` thread after `stop()` before exiting.

## D-36 — AsyncCheckpointSaver intermittently hangs at shutdown — FILED

- LiteTorch (VERIFIED): 2 hangs in ~46 runs (>60 s after `wait()` returned, killed; 0/26 on re-runs, no stack captured). Suspected background-thread teardown race. Source: `src/serialization/async_checkpoint.cpp`.
- PyTorch: no equivalent component.
- Impact: MEDIUM. Intermittent hang at exit in checkpointing code.
- Migration: `del saver` explicitly before exit; do not rely on interpreter teardown to stop the background thread.
- Fix status: flagged to Core as a side note in comms `20261027-000400-docs-to-core-nadam-onecycle-negbin.md` (thread `docs-fix-requests-20261026`).

## D-37 — Python `Dataset` must define `size`/`get`, not `__len__`/`__getitem__`; and the loader aborts at shutdown — FILED

- LiteTorch (VERIFIED): a `Dataset` subclass defining `__len__`/`__getitem__` is never called. The `PyDataset` trampoline (`src/bindings/bindings_common.h:173`) dispatches `size()`/`get()` via `PYBIND11_OVERRIDE_PURE` to Python methods named `size` and `get` — not `__len__`/`__getitem__`. `DataLoader` then raises `RuntimeError: Tried to call pure virtual function "data::Dataset::size"`. Once `size`/`get` exist, `len(dataset)` / `dataset[i]` work (they are bound to the same C++ methods). After fixing that, the example produces correct output but the process dies with `FATAL: exception not rethrown` (SIGABRT, exit -6) at shutdown — the C++ `DataLoader` holds the Python-backed dataset past teardown. Workaround: explicit `del loader` / `del dataset` before exit (verified clean exit 0). A Python `collate_fn` triggers the same shutdown abort.
- PyTorch: `Dataset` subclasses define `__len__`/`__getitem__`; no shutdown abort.
- Impact: HIGH. Every ported custom dataset fails twice: first at construction, then at exit.
- Migration: define `size()` and `get(idx)` exactly as spelled; `del loader` / `del dataset` explicitly before exit.
- Fix status: FILED to Core, comms `20261026-233500-docs-to-core-h7d2.md` (thread `docs-fix-requests-20261026`).

## D-38 — `ring_attention` is uncallable from Python — FILED

- LiteTorch (VERIFIED): `ring_attention` cannot be called from Python (`ProcessGroup.get()` reference holder vs `shared_ptr` parameter; `Unable to load a custom holder type`). Binding: `src/bindings/bind_distributed.cpp`.
- PyTorch: `torch.distributed` has no `ring_attention`; the equivalent is `torch.nn.attention.flex_attention` / manual sequence parallelism.
- Impact: MEDIUM. The documented ring-attention path does not exist in practice; use manual sequence sharding via `all_gather`/`reduce_scatter` (see H2.4).
- Migration: use the H2.4 manual-sharding fallback.
- Fix status: FILED to Distributed, comms `20261026-234600-docs-to-distributed-ring.md` (thread `docs-d9-fsdp-20261026`).

## D-39 — VocabParallelEmbedding forward() throws `data_ptr()` non-FP32 — FILED (consequence of D-01)

- LiteTorch (VERIFIED on 88a4ba1): `VocabParallelEmbedding` construction works, but `forward()` throws `data_ptr()` non-FP32 for both INT64 and FP32 index tensors — the embedding lookup hits the C++ `data_ptr()` FP32-only path (D-01).
- PyTorch: `nn.Embedding` forward works with int64 indices.
- Impact: HIGH for tensor-parallel embedding use. The module is unusable for forward passes on this build.
- Migration: none — use `nn.Embedding` (single-device) instead; the reference example is construction-only and documents the failure.
- Fix status: filed with D-22–D-29 to Core, comms `20261027-001600-docs-to-core-r7-divergences.md` (thread `docs-fix-requests-20261026`).

## D-40 — `linalg.cholesky_inverse` takes the matrix, not its Cholesky factor — OPEN

- LiteTorch (VERIFIED): `cholesky_inverse(A)` takes the matrix itself — `cholesky_inverse(A) == inv(A)`; passing the factor gives wrong results. Source: `src/ops/linalg.cpp:642`.
- PyTorch: `torch.cholesky_inverse` takes the Cholesky factor.
- Impact: MEDIUM. Passing a factor (the torch idiom) silently returns wrong results.
- Migration: pass the original matrix, not its factor.

## D-41 — `linalg.lstsq` returns 0-D tensors for residuals and rank — OPEN

- LiteTorch (VERIFIED): `lstsq` returns 0-D tensors for residuals (`0.0`) and rank (`2.0`); torch returns an empty tensor and an int. Source: `src/ops/linalg.cpp:1025`.
- PyTorch: `torch.linalg.lstsq` returns `residuals` (empty tensor) and `rank` (int).
- Impact: LOW. Shape/type mismatch on unpacking; values are correct.
- Migration: read rank via `.item()` / `.to_vector()[0]`; do not index `residuals[0]`.

## D-42 — `linalg.matrix_norm` `ord` must be a string — OPEN

- LiteTorch (VERIFIED): `matrix_norm(A, ord=1)` throws TypeError; `ord` must be a string (`ord="1"`). All 8 string ords verified: fro/nuc/1/2/inf/-inf/-1/-2. Binding: `src/bindings/bind_linalg.cpp:45` (`ord` = "fro").
- PyTorch: `torch.linalg.matrix_norm(A, ord=1)` accepts an int.
- Impact: LOW. Mechanical.
- Migration: pass `ord` as a string.

## D-43 — `fft.ifft` / `ifftn` return real tensors, not complex layout — OPEN

- LiteTorch (VERIFIED): `ifft`/`ifftn` return real tensors, not the `[..., 2]` complex layout that `fft` produces. Binding: `src/bindings/bind_fft.cpp:8, 16`.
- PyTorch: `torch.fft.ifft` returns complex tensors.
- Impact: LOW-MEDIUM. Code expecting complex output from `ifft` gets real output.
- Migration: treat `ifft` output as real; do not index a complex last dimension.

## D-44 — `Trainer.train_step()` runs no forward/backward — OPEN

- LiteTorch (VERIFIED): `Trainer.train_step()` does NOT run forward/backward (optimizer step + zero_grad only). `Trainer` checkpoints are async by default (files appear only after `wait_pending_saves()`). Source: `src/trainer/trainer.cpp`.
- PyTorch: `Trainer` is not a torch builtin (this is a `transformers`-style API); the comparison is to the documented LiteTorch contract, which the method violates.
- Impact: HIGH for anyone using `Trainer` as a training loop. Calling `train_step()` without a prior manual forward/backward steps the optimizer over stale (or zero) gradients.
- Migration: run forward + backward manually before `train_step()`; call `wait_pending_saves()` before reading checkpoint files.

## D-45 — `save` / `load` take a Module/list/Optimizer, not a state dict — OPEN

- LiteTorch (VERIFIED): `lt.save`/`lt.load` are in-place and type-dispatched (a bare tensor throws); `save_parameters` takes a param list, not a module; `save_safetensors`/`save_async` take a dict, not a module; `save_optimizer_state` supports Adam/AdamW/SGD/RMSprop only and throws for every other optimizer.
- PyTorch: `torch.save(state_dict, path)` takes arbitrary objects including state dicts.
- Impact: MEDIUM. Ported checkpoint code fails at the call site.
- Migration: `save_parameters(model.parameters())` for weights; `save_safetensors(dict)` for dicts; use `state_dict()`/`load_state_dict(sd)` separately for live-model transfers.

## D-46 — `DataLoader(shuffle=True)` is NOT reproducible under `lt.manual_seed` — DOCUMENTED

- LiteTorch (VERIFIED in B1): `DataLoader(shuffle=True)` is not reproducible despite `lt.manual_seed(42)` — epoch-1 loss varied 1.0306 / 0.9762 / 1.0724 across runs. DataLoader shuffle uses its own RNG. Data classes live under `lt.data.*` (not top level); samplers are not iterable (`indices()`/`batches()` only); `rank()`/`world_size()`/`is_distributed()` are methods.
- PyTorch: `torch.manual_seed` seeds the DataLoader shuffle (with a generator) deterministically.
- Impact: MEDIUM for reproducible doc numbers and experiments.
- Migration: use `shuffle=False` for reproducible runs (all doc examples do); seed the sampler explicitly if shuffling is required.

## D-47 — Non-leaf `.grad` is populated after `backward()` without `retain_grad()` — FILED

- LiteTorch (VERIFIED in B4, E2): after `backward()`, a non-leaf tensor's `.grad` holds the seed gradient ([1.0]). `retain_grad()` exists but changes nothing observable.
- PyTorch: non-leaf `.grad` stays `None` (with a warning) unless `retain_grad()` was called.
- Impact: MEDIUM. Code that reads non-leaf grads (e.g. debugging hooks, gradient surgery) silently gets values torch would not provide; code that checks `grad is None` to detect non-retained tensors breaks.
- Migration: none available — treat non-leaf `.grad` as populated in LiteTorch; do not rely on `None` as a signal. Filed to Core (comms 20261027-012500-docs-to-core-autograd-div.md).

## D-48 — `Tensor.zero_grad()` sets `.grad` to `None`, not zeros — FILED

- LiteTorch (VERIFIED in B4, E2): `zero_grad()` sets `.grad` to `None`. The next `backward()` re-creates it, so training loops work; direct reads of a freshly zeroed grad break.
- PyTorch: `zero_grad()` leaves a zeros tensor in `.grad`.
- Impact: LOW. Breaks only code that reads `.grad` immediately after zeroing (e.g. assertions, logging).
- Migration: guard reads with `if t.grad is not None`, or call `backward()` before reading. Filed to Core (comms 20261027-012500-docs-to-core-autograd-div.md).

## D-49 — `backward()` has no `retain_graph` parameter, though its error message names one — FILED

- LiteTorch (VERIFIED in B4, E2): `backward()` signature is `(gradient=None, create_graph=False)` — no `retain_graph`. A second backward on the same graph throws with an error message that says "pass retain_graph=true to run backward multiple times" — a parameter that does not exist.
- PyTorch: `backward(retain_graph=True)` allows repeated backward passes.
- Impact: MEDIUM. Any ported code that re-runs backward (e.g. multi-loss, gradient accumulation through the same graph) fails; the error message misdirects the user toward a nonexistent fix.
- Migration: rebuild the graph (re-run forward) for each backward, or use `create_graph=True` where applicable. Filed to Core (comms 20261027-012500-docs-to-core-autograd-div.md).

## D-50 — `str(t.device)` is `gpu:N`, not `cuda:N` — FILED

- LiteTorch (VERIFIED from source + B5 GPU run): `DeviceType` has four members — CPU, GPU, META, TPU — and no CUDA member (`include/litetorch/device.h:11-14`). `to_string()` maps GPU to `"gpu:" + index` (`device.h:35`), so a GPU tensor prints `gpu:0`. The string parser accepts `"cuda"` and `"cuda:N"` as INPUT and normalizes both to `DeviceType::GPU` (`src/bindings/bind_core.cpp:34`), so `lt.Device("cuda")` and `lt.Device("cuda:0")` construct fine — but nothing they print ever reads `cuda:0`. The B5 e1-9 check asserted `str(t.device) == "cuda:0"` and failed on the real GPU (the device printed `gpu:0`); b5-t2.4-1 passed printing `training on: gpu:0` in the same run.
- PyTorch: `str(torch.device("cuda:0"))` is `"cuda:0"`.
- Impact: MEDIUM. Ported code that asserts or branches on the `cuda:0` string (device placement checks, logging, checkpoint metadata keyed by device string) fails or misroutes on LiteTorch; the failure looks like a missing GPU when the GPU is actually in use.
- Migration: compare against `"gpu:0"` (or `t.device.type == lt.DeviceType.GPU`), never the `cuda:` string; keep using `"cuda"` / `"cuda:0"` as constructor input — the parser accepts it. Whether the product should align with PyTorch (`cuda:0`) in future is a design decision, not settled; this triage does not recommend changing the product. Filed to Backend as a docs expectation error (defect record `backend/defects/B5-E1-9-DEVICE-STRING.md`, comms thread `docs-b5-gpu-20261027`).

## Verified edge cases (not numbered — API facts, not divergences)

These were verified live during the B3 reference pass and are documented in the API reference; they are collected here so porters find them in one place.

- `fused_bias_relu` / `fused_bias_gelu` require identical shapes for x and bias (no broadcasting).
- `Tracer.trace` returns `JITFunction`, not `JITVar`; `Tracer.trace` with plain `+` intermediates raises the expired-intermediate error (use `lt.Ops.*`).
- `PreparedModel.calibrate` takes a zero-arg callable `data_fn`, not a list.
- `ltrec.negative_sampling` takes an edge-index tensor + num_nodes, not item lists.
- `FlatParameter.param_views()` requires `all_gather_flat()` first.
- `fuse_conv_bn` requires the BN in eval mode (same as `fold_conv_bn`).
- `DataType` is not iterable from Python; `FP4` aliases `FP4_E2M1`.
- `ServingScheduler.get_block_table_tensor()` returns `None` once all requests finish (blocks freed) — capture the table before the final step.
- `lt.autograd.gradcheck` clears `x.grad` — compute the analytic gradient with an explicit backward after gradcheck.
- `SyncBatchNorm` is a channel BatchNorm: rank-2 (N, C) input raises "channel BatchNorm: bad input rank or channels"; rank-3 (N, C, L) works.
- `TransformerDecoderLayer` ctor takes `(embed_dim, num_heads, dim_feedforward=2048)` — NOT `(d_model, nhead)` like `TransformerEncoderLayer`.
- `F.dpo_loss` requires ref logps unless `reference_free=True` (else ValueError).
- `lt.Device("cuda", rank)` is not a valid constructor (string ctor takes one arg); use `lt.Device("cuda:%d" % rank)`.
- Multi-process examples need the `if __name__ == "__main__"` guard under spawn.
- `w8a8_matmul` formula is `(round(x / x_scale) @ round(w / w_scale).T) * x_scale * w_scale` with INT8 clamping to [-128, 127] — not `(x @ w.T) * x_scale * w_scale`.
- `moe_gate` renormalizes the selected top-k probs to sum to 1 (`/ (sum_top_k + 1e-9f)`).
- NF4/FP4 bare `.cast()` is not usable quantization: FP4 round-trip of `[0.1, -0.5, 1.2, -2.4]` is `[0.0, -0.5, 1.0, -2.0]`; NF4 is `[0.0596, -0.5251, 1.0, -1.0]` — NF4 clamps to [-1, 1] with no per-tensor scale.
- `FSDPUnit` methods are unreachable from Python (no `py::init`, `active_units()` unbound) — do not document or call them; use `fsdp_fully_shard`. `destroy_process_group()` is NOT bound — use `shutdown()`. FSDP shards params + grads, NOT optimizer states.
- `linalg.matrix_norm` string ords: fro/nuc/1/2/inf/-inf/-1/-2 (see D-42).

## Checked and NOT divergent (do not re-investigate)

- `no_grad` and `inference_mode` context managers exist (`src/bindings/bind_core_ops.cpp:114, 124`).
- `cross_entropy_loss` has `weight`, `ignore_index`, `reduction`, `label_smoothing` (`src/bindings/bind_ops.cpp:70-75`).
- `Tensor.__pow__` exists for scalar exponents; `Ops` surface is broad (127 `def`s in bind_ops.cpp).
- `Tensor.cuda()` / `.cpu()` / `.half()` / `.float()` / `.double()` / `.int()` / `.long()` exist and map to the expected dtypes (`.long()` is INT64 since the D-06 fix).
- Linalg: 27/27 ops match torch (inv/solve/cholesky/det/slogdet/eigvalsh/trace/matrix_power/cond/diagonal/rank/pinv/norms/lu/cross/vecdot/multi_dot/outer/inner/kron/vdot; qr/svd via reconstruction due to sign ambiguity; lstsq solution matches). FFT: 9/9 match (real/imag split of the `[..., 2]` layout). Distributions: 19/20 match (NegativeBinomial is the D-32 swap).
- Schedulers StepLR, CosineAnnealingLR, ExponentialLR, MultiStepLR, LinearLR, PolynomialLR, LambdaLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts all match torch exactly. OneCycleLR is the D-31 shift.
- GradScaler overflow-skip verified: inf grad → step skipped, scale 4.0 → 2.0.
- AdamW8bit/FP8 vs torch AdamW: identical to 6 decimals on the 1-step probe.
- `linalg.lstsq` solution matches torch; `Ops.cross_entropy_loss` / NLL / fused LCE match torch exactly for INT32 targets (CE 5.96e-08, NLL 0, fused LCE 0).
- `flash_attention` matches torch manual attention (2.38e-07 causal, 1.19e-07); overlapped `all_reduce` matches exactly on both ranks.

## Live checks still needed

- L-01 (Kaggle GPU): D-02 fallback with a real GPU present-but-busy; D-10 transfer direction per op family; D-21 int64 FP32-rounding error magnitudes.
- L-02 (CPU harness, now regression checks): numeric max-diff vs torch for D-07 linalg ops (expecting parity); D-08 scalar promotion across all integer dtypes; D-09 GELU tolerance confirmation (re-measured in this batch: 4.13e-04 vs exact, 0.0 vs tanh); D-18 int64 `.item()` boundary (2^53).
- L-03 (CPU harness): D-19 quantized promotion vs torch semantics.
- L-04 (Kaggle TPU): D-02 TPU fallback; D-20 `flash_attention_half` throw confirmation on TPU.
- L-05 (CPU harness, regression check post-fix): D-06 — `.long()` output now INT64 used as indices in `gather` / `index_select` (INT64 acceptance).

## Fix-status summary

| Status | Entries |
|---|---|
| FIXED | D-06 (merged @89afd67), D-07 (merged @7b8981e) |
| FILED | D-22–D-29 (comms 20261027-001600), D-23 also (comms 20261026-234500), D-30–D-32 + D-36 side note (comms 20261027-000400), D-33 (comms 20261027-000800), D-37 (comms 20261026-233500), D-38 (comms 20261026-234600), D-39 (with D-22–D-29), D-47–D-49 (comms 20261027-012500), D-50 (comms thread docs-b5-gpu-20261027; Backend triage record backend/defects/B5-E1-9-DEVICE-STRING.md) |
| OPEN | D-02, D-03, D-04, D-05, D-08, D-10, D-11, D-12, D-13, D-14, D-15, D-16, D-17, D-18, D-19, D-21, D-35, D-40, D-41, D-42, D-43, D-44, D-45 |
| DOCUMENTED (no fix planned) | D-01 (C++ throw; Python unbound), D-09 (tanh GELU), D-34 (gradcheck eps), D-46 (DataLoader shuffle RNG) |
