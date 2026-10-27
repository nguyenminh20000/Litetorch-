# Installation on Kaggle

Kaggle is the supported way to run LiteTorch on GPU (2x T4) and TPU (v5) without local hardware. This page covers notebook setup, fetching the tree, building, the GPU pre-flight gate, and keeping the session alive. GPU work comes first: finish all GPU work before touching TPU.

> Scope note: the Kaggle GPU steps below (GPU pre-flight checks, `liblitetorch_gpu.so` placement, the T4 device count) are written from the project's Kaggle runbook and verified run history. They were not executed on this docs VM, which has no GPU; the GPU portion is a Batch B5 verification item (one GPU package per docs batch, launched by QA). The Python smoke test on this page runs on CPU locally and runs unchanged on Kaggle GPU.

---

## 1. Notebook setup

1. Create a Kaggle notebook with a **GPU** accelerator (2x T4) or, for the TPU track, a **TPU v5** accelerator. Only one interactive GPU session is allowed per account — before launching, confirm no teammate agent is already running on the account; never stop a teammate's session or wipe their `/kaggle/working` tree.
2. Use the bundled datasets, never external downloads: public dataset mirrors die without warning. Prefer Kaggle's bundled datasets and write scripts to skip download when the files already exist.
3. Sign-in uses the saved Google account (cookies persist). If Google 2-step ("Verify it's you") appears on a new session: stop at that screen and surface it to the commander immediately — the user approves on their own device; never automate it.

---

## 2. Fetch the tree (GitHub-fetch-only)

Browser uploads to Kaggle are retired. Run inputs are pushed to GitHub first; the Kaggle session only fetches and verifies a sha256 before applying. Never upload code through the upload dialog.

```bash
cd /kaggle/working
git clone https://github.com/nguyenminh20000/Litetorch-.git
cd Litetorch-
git checkout agent-v1
```

For docs-example run packages (one per batch), fetch the `kaggle-run-inputs` branch instead and verify the sha256 listed in the batch manifest before applying:

```bash
cd /kaggle/working
git clone --branch kaggle-run-inputs https://github.com/nguyenminh20000/Litetorch-.git run-inputs
cd run-inputs
sha256sum <package-file>
```

`/kaggle/input` is read-only: copy trees to `/kaggle/working` before building, and read datasets from `/kaggle/input/datasets/<owner>/<slug>/`.

---

## 3. Check the CUDA toolkit, then build

Check (and install if missing) the CUDA toolkit explicitly before any GPU CUDA run. It happened to be present on past GPU images, but that is not guaranteed.

```bash
nvcc --version
```

Build the extension in place:

```bash
cd /kaggle/working/Litetorch-
pip install pybind11
python3 setup.py build_ext --inplace
```

Two facts from the runbook:

1. **`setup.py` sets `package_dir={"": "python"}`, so the built `.so` lands under `<tree>/python/`.** Every notebook cell and script must put `<tree>/python` on `sys.path` before `import litetorch`.
2. **On the current Kaggle image, `build_ext --inplace` writes `liblitetorch_gpu.so` into `build/lib.linux-x86_64-cpython-313/`** — it must be copied next to the extension under `python/` before import works (the `python/` copy is what `dlopen` finds; `/tmp` copies are wiped on session restart).

```bash
cp build/lib.linux-x86_64-cpython-313/liblitetorch_gpu.so python/
ls python/
```

---

## 4. GPU pre-flight gate

Before running any GPU example, assert all three:

```bash
python3 -c "import sys; sys.path.insert(0, 'python'); import litetorch as lt; print(lt.get_backend_name()); print(lt.cuda.device_count())"
```

1. The `Native GPU (CUDA/HIP)` init line is present (NOT OpenCL) — this catches the missing-native-library failure.
2. `liblitetorch_gpu.so` sits next to the extension under `python/`.
3. `lt.cuda.device_count()` returns the expected count (2 on the Kaggle GPU image).

If any assertion fails, stop and report it as a harness / blocker issue, not as example failures.

---

## 5. Smoke test

The snippet below is self-contained and runs on CPU locally and on Kaggle GPU unchanged. On GPU it exercises the same import path and the GPU matmul; on CPU it verifies the install.

<!-- ex: id=2.8-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->

```python
import litetorch as lt

device = lt.auto_device()
print("Backend:", lt.get_backend_name())
print("Device:", device)
print("GPU available:", lt.is_gpu_available())

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
print("Result:", y.to_vector())
print("Sum:", lt.Ops.sum(y).item())
```

Verified on CPU, LiteTorch 88a4ba1 (torch parity verified 2026-10-27 vs torch 2.14.1+cpu: matmul result and sum match exactly, max diff 0.0). GPU run of the same snippet is a Batch B5 item.

Expected output on CPU:

```
Backend: CPU
Device: cpu:0
GPU available: False
Result: [7.0, 10.0, 15.0, 22.0]
Sum: 54.0
```

On a Kaggle GPU session the first three lines read `CUDA` / `cuda:0` / `True`; the result and sum are identical.

---

## 6. Session hygiene

- **Idle timeout (~20–30 min) wipes `/kaggle/working`.** Keep the session alive ("Continue editing" / real UI interaction every 5–6 minutes) until the task is done and reported, and re-verify the `.so` and data after any restart.
- **Editor draft-sync can silently wipe `/kaggle/working` across session restarts** (`ConcurrencyViolation`). After any restart, reload the editor until it reports "Draft saved", then re-run the whole sequence. Keep clone → patch → build → repro re-runnable in one go.
- **Never close a session on unfinished work.** Session death mid-task has already cost this project multiple runs.
- **Sign out of Kaggle + Google when the work ends.** Never skip this step.
- **TPU only after GPU green.** The TPU verification package is a separate `kaggle-run-inputs` push (one per batch, same as GPU).

---

## 7. Where to go next

- First run in the notebook: [Quickstart](../quickstart.md).
- Full install reference: [Installation](../installation.md).
- Training workloads and perf benches on Kaggle: Tutorials T2–T3 and the how-to guides (in progress).
