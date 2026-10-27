# Troubleshooting and FAQ

Every entry below is drawn from a failure actually observed during the docs rewrite (batches B0–B4) or a verified defect in the tree. Symptom, cause, fix — no invented failure modes. Each entry links to the Behavioral Differences page (D-number) or the API reference where the behavior is documented with numbers.

## Import and installation

### `ModuleNotFoundError: No module named 'litetorch'`

**Cause:** the built extension lives under `<tree>/python/`, not the tree root. `setup.py` sets `package_dir={"": "python"}`, so `build_ext --inplace` puts the importable `.so` under `<tree>/python/`. This is the Phase 6 import-blocker root cause (2026-10-17).

**Fix:** add `<tree>/python` to `sys.path` before importing — never import from the tree root:

<!-- ex: id=ts-import-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

print("build_commit:", lt.build_commit())
print("backend:", lt.get_backend_name())
```

Also check the real gate, not the build log: `setup.py` prints `BUILD OK` even when nvcc per-file compiles fail. The real gate is the `.so` existing and `import litetorch` succeeding. On Kaggle, `liblitetorch_gpu.so` must sit next to the extension under `python/` — the copy-to-`/tmp` path is wiped on session restart.

### `ImportError` / missing symbols after a build

**Cause:** a stale extension from an older commit, or a build that silently failed per-file (see above).

**Fix:** rebuild with `python3 setup.py build_ext --inplace`, check `lt.build_commit()` matches the tree HEAD, and check the extension file exists under `<tree>/python/`. If the build was interrupted, a resume can trigger a full setuptools rebuild — budget for it (local full builds do not complete in one session on the 2-CPU VM; verification builds run on Kaggle CPU).

## Devices

### "My GPU job runs, but it is slow" / "device is cpu after `.to('cuda')`"

**Cause:** the silent CPU fallback (D-02). `.to(Device("cuda"))` on a box without a GPU backend silently returns a CPU tensor — no error, no warning. A typo'd device string does the same (D-03): `Device("cude")` silently becomes CPU.

**Fix:** check `t.device` after every `.to()` / `.cuda()` and fail loudly on mismatch:

```python skip-run
t = t.to(lt.Device("cuda"))
assert str(t.device).startswith("cuda"), f"expected cuda, got {t.device} (D-02)"
```

Also check `lt.cuda.is_available()` before assuming a GPU exists, and spell device strings exactly (`lt.Device("cuda:0")`, not `lt.Device("cuda", 0)` — the two-arg string ctor does not exist).

### Binary op silently moves tensors between devices

**Cause:** `add`/`sub`/`mul`/`div` (and comparisons, matmul, shape ops) move the mismatched operand to the other operand's device instead of raising (D-10).

**Fix:** move tensors explicitly before mixing devices; treat any implicit transfer as a bug to be found, not a feature.

## Training does not learn

### Loss decreases in torch but stays flat / goes NaN in LiteTorch

**Cause (most likely):** the gradients are wrong while the forward values look right. This is exactly the no-learn defect of 2026-10-25/26 (a `cached_native_kernel` collision ran `ConvolutionBackwardData` for every "filter" call) — and the same shape of failure recurs with DDP (D-33: backward doubles one grad slice) and NAdam (D-30: first-step update ~6.5x too large).

**Fix:** compare grads, not just forward values. Run the gradcheck / parity procedure in `plan/verification.md` §3: copy weights from LiteTorch into torch, run the same step on both, compare loss AND every parameter grad at FP32 1e-4. If forward matches and grads do not, the defect is in the backward path — file it, do not tune the learning rate around it.

### Optimizer steps but parameters never change

**Cause:** the module has no registered parameters. Assigning a submodule as a Python attribute (`self.linear = lt.nn.Linear(4, 4)`) does NOT register it — `parameters()` returns 0 and `state_dict()` is empty (D-24).

**Fix:** register children explicitly with `add_module(name, module)`, or override `parameters()` to collect submodules manually (the T1 tutorial's `MultiLayerPerceptron` does this). Verify with `len(model.parameters())` before creating the optimizer.

### Loss is NaN on the first step

**Cause:** check the usual suspects first (learning rate, data scale), then check whether the loss itself is the problem: CE/NLL/fused-LCE reject INT64 targets (D-23) — but that raises, it does not NaN. A genuine NaN on step 0 with correct grads points at the optimizer: NAdam's first step is ~6.5x too large (D-30).

**Fix:** switch to Adam/AdamW and re-run; if the NaN disappears, it was D-30.

### `Parameter(tensor)` throws on construction

**Cause:** `Parameter(tensor)` is broken on 88a4ba1 — it always throws "copy_: a leaf Tensor that requires grad is being used in an in-place operation", even for requires_grad=False inputs (D-25).

**Fix:** none — do not construct `Parameter` directly; obtain parameters from modules (for example `Linear.weight`).

### `m(q, k, v)` raises TypeError

**Cause:** `__call__()` only routes the single-tensor forward (D-22). Multi-argument modules — MultiHeadAttention, Bilinear, all 2-arg losses — must be called via `.forward(...)`.

**Fix:** `m.forward(q, k, v)`.

## Data

### `DataLoader` raises `RuntimeError: Tried to call pure virtual function "data::Dataset::size"`

**Cause:** the `Dataset` subclass defines `__len__`/`__getitem__`, which the `PyDataset` trampoline never calls — it dispatches to Python methods named `size` and `get` (D-37).

**Fix:** define `size()` and `get(idx)` exactly as spelled. Once they exist, `len(dataset)` / `dataset[i]` work (they are bound to the same C++ methods).

### Process aborts at exit with `FATAL: exception not rethrown` (SIGABRT)

**Cause:** the C++ `DataLoader` holds a Python-backed dataset past teardown (D-37). A Python `collate_fn` triggers the same abort. The same shape of failure appears with `RendezvousStore` (D-35: join the `run()` thread after `stop()`) and intermittently with `AsyncCheckpointSaver` (D-36).

**Fix:** `del loader` / `del dataset` explicitly before exit (verified clean exit 0). For `RendezvousStore`, `join()` the `run()` thread after `stop()`. For `AsyncCheckpointSaver`, `del saver` before exit.

### Epoch losses are not reproducible despite `lt.manual_seed(42)`

**Cause:** `DataLoader(shuffle=True)` uses its own RNG, not the `manual_seed` stream (D-46). Epoch-1 loss varied 1.0306 / 0.9762 / 1.0724 across runs in B1.

**Fix:** use `shuffle=False` for reproducible runs (all doc examples do).

### `torch.save(state_dict, path)` equivalent fails

**Cause:** `lt.save`/`lt.load` are in-place and type-dispatched — a bare tensor throws; `save_parameters` takes a param list, not a module; `save_safetensors`/`save_async` take a dict, not a module (D-45).

**Fix:** `save_parameters(model.parameters())` for weights; `save_safetensors(dict)` for dicts; `state_dict()`/`load_state_dict(sd)` for live-model transfers. `save_optimizer_state` supports Adam/AdamW/SGD/RMSprop only.

## Losses and numerics

### `cross_entropy_loss: target must have integer or float dtype`

**Cause:** CE/NLL/fused-LCE reject INT64 targets and require INT32 (D-23). After the D-06 fix `.long()` returns INT64 — which these losses reject.

**Fix:** cast targets to INT32 explicitly (`.cast(litetorch.DataType.INT32)`); do not pass `.long()` results as loss targets.

### GELU values differ from torch by ~4e-4

**Cause:** LiteTorch GELU is the tanh approximation only; there is no `approximate` argument and no erf path (D-09). This is intentional, not a bug.

**Fix:** use a tolerance of at least ~5e-4 for GELU, or compare against `F.gelu(x, approximate='tanh')` (exact match, 0.0).

### `gradcheck` fails on a trivial function

**Cause:** the default `eps=1e-6` is below FP32 resolution (D-34). Even `t*t` fails at x=3 (analytical 6 vs numerical 5.72).

**Fix:** always pass `eps=1e-3` explicitly for FP32 gradcheck:

<!-- ex: id=ts-gradcheck-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([3.0], [1], cpu, True)

def sq(v):
    return lt.Ops.mul(v, v)

print("gradcheck eps=1e-6:", lt.autograd.gradcheck(sq, [x], 1e-6, 1e-5, 1e-4, False))
print("gradcheck eps=1e-3:", lt.autograd.gradcheck(sq, [x], 1e-3, 1e-5, 1e-4, False))
```

Note: `gradcheck` clears `x.grad` — compute the analytic gradient with an explicit backward after gradcheck.

### `.numpy()` throws "unsupported dtype for copy conversion"

**Cause:** `.numpy()` handles FP64/FP32/FP16/INT32/INT16/INT8 only; UINT8, INT64, BOOL hit the default and throw (D-14). Non-CPU tensors throw too.

**Fix:** `.cpu()` first, then `.cast(FP32)` (precision caveat for int64 above 2^24), or use `.tolist()` / `.to_vector()`.

### `.item()` returns a wrong value for a large int64

**Cause:** `.item()` returns float, casting non-FP32 tensors to FP32 first; int64 values above 2^53 lose precision (D-18).

**Fix:** avoid `.item()` for large int64 values; read via `.numpy()` (after casting, see D-14) or keep values in tensor form.

### `linalg.matrix_norm(A, ord=1)` raises TypeError

**Cause:** `ord` must be a string in this binding (D-42). All 8 string ords work: fro/nuc/1/2/inf/-inf/-1/-2.

**Fix:** pass `ord="1"`.

## Distributed

### DDP training diverges from single-process training

**Cause:** `DistributedDataParallel.backward` doubles one slice of the flat weight grad (D-33). Bias grad is correct; single-process backward is correct.

**Fix:** do not train with DDP until fixed — use single-process training or FSDP (`fsdp_fully_shard`).

### `ring_attention` raises a pybind holder error

**Cause:** `ring_attention` is uncallable from Python (`ProcessGroup.get()` reference holder vs `shared_ptr` parameter) (D-38).

**Fix:** use the H2.4 manual sequence-sharding fallback (`all_gather`/`reduce_scatter`).

### `destroy_process_group()` raises AttributeError

**Cause:** it is not bound — use `shutdown()` (see D9 resolution in `plan/d9-resolution.md`).

**Fix:** `lt.distributed.shutdown()`.

### `FSDPUnit` cannot be constructed

**Cause:** `FSDPUnit` methods are unreachable from Python (no `py::init`, `active_units()` unbound). FSDP shards params + grads, NOT optimizer states.

**Fix:** use `fsdp_fully_shard(module)` as the entry point (see H2.1).

## Out of memory

### Post-training OOM / MNIST OOM

**Status:** the post-training OOM is a known open issue, not solved (9 identical occurrences). The MNIST OOM [BLOCKER] is **CLOSED 2026-10-27 as not-reproduced**: Path A (qa-r58a) ran the exact failing config (MLP 784→128→10, BATCH=256, EPOCHS=3) on tip b476a5e and passed — rc 0, losses 0.5112→0.2304→0.1747, accuracy 95.47%, torch parity 1.192e-07, no OOM. The earlier qa-r58 diagnostic also did not reproduce it on cdbc716c (both runs exit 0, peak lt GPU use ~231 MiB). Closure is "not reproduced on current tip", NOT proof of fix: no named cause exists, and Backend's delta review found NO candidate fix in 536f3b5→cdbc716c. If the OOM reappears, file a repro with the exact config. Do not treat these as doc errors — the docs record them as known issues.

**What to check first:** whether the OOM happens after training completes (post-training OOM — the training itself succeeded) or during data loading (MNIST OOM). Reduce batch size / enable gradient checkpointing as a workaround; file a repro if the failure is new.

### `flash_attention_half` throws on TPU

**Cause:** TPU `flash_attention_half` is not implemented and throws (D-20).

**Fix:** use FP32 attention on TPU, or run attention on GPU.

## FAQ

**Q: Is LiteTorch a drop-in replacement for PyTorch?**
A: No. The API surface is deliberately similar, but the Behavioral Differences page documents 46 verified deviations (D-01–D-46), several of them silent. Port with the migration guide, verify with parity.

**Q: Which PyTorch version is the reference?**
A: torch 2.14.1+cpu, live-probed for every PyTorch-side claim. FP32 parity tolerance is 1e-4 (standing QA criterion); FP16/BF16 is 1e-2.

**Q: Why does my ported model train in torch but not in LiteTorch?**
A: In order: (1) check `parameters()` is non-empty (D-24); (2) compare grads, not just forward values (the no-learn lesson); (3) check for DDP (D-33) or NAdam (D-30); (4) check loss targets are INT32 (D-23).

**Q: Where do I run GPU examples?**
A: On Kaggle, via QA — Docs never drives a Kaggle session directly. GPU examples are marked `REQUIRES-GPU` and batched into one `kaggle-run-inputs` package per batch. The pre-flight gate asserts the `Native GPU (CUDA/HIP)` init line, `liblitetorch_gpu.so` placement, and `device_count() == 2`.

**Q: Can I use ROCm examples?**
A: They ship as explicit `UNVERIFIED-ROCm` callouts — no ROCm hardware has verified them.

**Q: What is the fastest way to check a single claim in these docs?**
A: Every code example carries a metadata header (`id/device/backend/runtime/torch-parity/status`) and runs standalone. Extract it with `docs/tools/extract_examples.py` and run it with `docs/tools/run_examples.py` — the same harness that verified the batch.

**Q: I found a behavior that contradicts these docs. What do I do?**
A: File it on the comms board (thread `docs-fix-requests-20261026`) with the example, the extension commit (`lt.build_commit()`), and the torch mirror. A doc claim contradicted by its own example output is a FAIL on the doc, not on the code.
