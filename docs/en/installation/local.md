# Installation on a Personal Machine

This page covers running LiteTorch on your own hardware: a Linux workstation or a Windows server, CPU or a single NVIDIA GPU. It does not repeat the install commands — those live in [Installation](../installation.md) and are linked from §1. This page tells you which route to pick, how to prove the GPU is really in use before you train, and what a first local training run looks like.

> Status note: the CPU steps on this page were run 2026-10-27 on the docs verification VM (2 CPUs, 7.7 GB RAM, Ubuntu 24.04, Python 3.12.3) against the Core D3 extension build, `litetorch.build_commit() == 88a4ba1`. GPU steps are marked `unverified` — there is no GPU on this VM. Windows steps are marked `unverified` — there is no Windows host on this VM. Do not treat an `unverified` block as run-verified until a machine with that hardware runs it.

---

## 1. Choose pip or source

| Situation | Route |
| :--- | :--- |
| You want the released version and will train on CPU or one GPU | pip — [Installation §2.2 (Linux pip)](../installation.md#22-linux-and-google-colab) or [§2.3 (Windows pip)](../installation.md#23-windows) |
| You need the current tree, you are patching the framework, or you need the native GPU library built from your own checkout | Source build — [Installation §2.4 (build from source)](../installation.md#24-build-from-source) |
| You want a reproducible environment that survives a host reinstall | Container — [Installation on Docker](docker.md) |

Two facts decide the choice on a small machine:

1. **PyPI ships a source distribution, not a wheel.** `pip install litetorch` downloads the sdist and compiles the C++ extension on your machine. Observed 2026-10-27: in a clean venv on this 2-CPU VM, pip built and installed `litetorch` 0.3.40 in ~28 minutes (wheel `litetorch-0.3.40-cp312-cp312-linux_x86_64.whl`, 23 MB, sha256 `c4b84b0062e8ea082009451a97d9066c5b7a13d581c0b71f9b9ef8776b6247a0`), and the installed package imports and runs the §2.6 smoke test on CPU. Note the PyPI release is older than the current tree: it has no `lt.build_commit()` attribute, so its commit cannot be pinned — the `verified-2026-10-27@88a4ba1` headers on this page rest on the Core D3 extension build, not on the PyPI package. A full local build of the current tree does not complete in one session here — see `plan/build-time.md`: killed at 44.6 min with 99/167 sources compiled and no `.so` produced. Budget accordingly, or use the prebuilt extension if one is available for your platform.
2. **A source build is only worth it if you can finish it.** On a small machine, limit build parallelism with `LT_BUILD_JOBS=1` (also forces `-O1` on the binding translation units, which lowers peak memory) — see [Installation §2.4](../installation.md#24-build-from-source). Verification builds for these docs run on Kaggle CPU instead; the configure steps (`cmake ..`, `cmake --build`) were verified here only as far as tool availability: `cmake` 3.28.3, `g++` 13.3, `CMakeLists.txt` and `setup.py` present in the tree.

---

## 2. GPU driver and CUDA toolkit check

Standing rule, not a suggestion: check (and install if missing) the CUDA toolkit explicitly before any GPU run. A missing `nvcc` means the native GPU library never builds, and the framework then falls back to CPU silently.

On Linux:

<!-- ex: id=local-2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```bash
nvidia-smi
nvcc --version
```

Verified 2026-10-27 on the docs verification VM: both commands are absent (`command not found`) — this VM has no GPU and no CUDA toolkit, which is the expected result on a CPU-only machine. On a GPU workstation both must print output: `nvidia-smi` lists the GPU and driver, `nvcc --version` prints the CUDA toolkit version (11.8 or 12.x per [Installation §2.1](../installation.md#21-system-requirements)).

If `nvcc` is not found on a machine that has an NVIDIA GPU, install the CUDA Toolkit for your distribution (for example, the NVIDIA CUDA 12.x network repository for Ubuntu) and re-run `nvcc --version` until it prints a version. cuDNN is a separate library, not included in the CUDA Toolkit — install it independently for accelerated convolutions; LiteTorch auto-detects it via `LITETORCH_CUDNN_ROOT` or standard paths. Only then build or run LiteTorch on GPU.

On Windows (unverified — no Windows host on this VM):

<!-- ex: id=local-2-2 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=unverified -->

```cmd
nvidia-smi
nvcc --version
```

Same rule: both must print output before a GPU run. For an AMD GPU, check that `hipcc` is present for native ROCm acceleration (ROCm examples in these docs are marked `UNVERIFIED-ROCm` until a ROCm run verifies them).

---

## 3. Pre-flight gate

Before any GPU run, assert all three of the following. This gate catches the most common local failure: the GPU backend silently not loading while training proceeds on CPU.

1. The `Native GPU (CUDA/HIP)` init line is present (NOT OpenCL) — this catches the missing-native-library failure.
2. `liblitetorch_gpu.so` sits next to the extension under `python/` (the `build_ext --inplace` output directory — see [Installation §2.4](../installation.md#24-build-from-source)). On the current build layout it lands in `build/lib.linux-x86_64-cpython-*/` and must be copied to `python/` before import works.
3. `lt.cuda.device_count()` returns the expected count (1 on a single-GPU workstation).

Run this pre-flight script. On a CPU-only machine it reports the CPU state; on a GPU machine the same script must report the GPU state — if it does not, stop.

<!-- ex: id=local-3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

print("Backend:", lt.get_backend_name())
print("Device:", lt.auto_device())
print("GPU available:", lt.is_gpu_available())
print("CUDA device count:", lt.cuda.device_count())
```

Verified 2026-10-27 on CPU, LiteTorch 88a4ba1 (D3 extension build; the `sys.path` preamble points at the extension directory — adjust it to your tree). Expected output on a CPU-only machine:

```
[litetorch Warning] GPU acceleration (OpenCL) is not available. Falling back to CPU.
Backend: CPU
Device: cpu:0
GPU available: False
CUDA device count: 0
```

The warning line is the CPU machine telling you there is no accelerator to fall back from — it is expected here, and it is the absence of the `Native GPU (CUDA/HIP)` init line that confirms no GPU backend loaded. On a GPU machine the first lines must instead show the `Native GPU (CUDA/HIP)` init line, `Backend: CUDA` (exact backend name depends on the build), `Device: gpu:0`, `GPU available: True`, and `CUDA device count: 1`.

### When the gate fails

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| No `Native GPU (CUDA/HIP)` line; `Backend: CPU` on a GPU machine | Native GPU library missing or `nvcc` absent at build time | Check `nvcc --version`, rebuild from source, confirm the init line appears |
| `ModuleNotFoundError: No module named 'litetorch'` | Extension not on `sys.path` | Add `<tree>/python` to `sys.path` — the extension lives under `python/`, not the tree root |
| `liblitetorch_gpu.so` not found at import | Library in the build directory, not next to the extension | Copy it from `build/lib.linux-x86_64-cpython-*/` to `python/` |
| `CUDA device count: 0` on a GPU machine | Driver or device not visible to the process | Check `nvidia-smi`; check the device is not held by another process or container |
| `CUDA device count: 1` when more GPUs are installed | Device-count detection defect (B5-DEVICE-COUNT, fix in review) | See [Behavioral differences](../behavioral-differences.md); do not assume the count is right until the fix lands |
| Training runs but is slow, `Device: cpu:0` | Silent CPU fallback — `.to(device)` falls back to CPU when the accelerator is unavailable | Stop and run the gate above; a slow run is not proof the GPU is in use |

If any assertion fails, stop and fix the install — do not treat the failure as a model or data problem. The full install verification script (matmul + backward) is [Installation §2.6](../installation.md#26-verification-script).

---

## 4. First local training run

### CPU run (verified)

This script trains the tiny model from the quickstart on four synthetic samples for 5 epochs. It is self-contained apart from the `sys.path` preamble, which points at the extension directory — adjust it to your tree. The loss must decrease; if it does not, something is wrong with the setup, not the model.

<!-- ex: id=local-4-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

lt.manual_seed(42)
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x_train = lt.Tensor.from_vector([
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    5.0, 6.0, 7.0, 8.0,
    8.0, 7.0, 6.0, 5.0
], [4, 4], device, False)

y_train = lt.Tensor.from_vector([
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0
], [4, 2], device, False)

dataset = lt.data.TensorDataset(x_train, y_train)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)

for epoch in range(5):
    loader.reset()
    total_loss = 0.0
    steps = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch
        optimizer.zero_grad()
        preds = model.forward(bx)
        loss = loss_fn.forward(preds, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    print(f"Epoch {epoch + 1}, Loss: {total_loss / max(steps, 1):.4f}")
```

Verified 2026-10-27 on CPU, LiteTorch 88a4ba1. Expected output (exact with `lt.manual_seed(42)` and `shuffle=False`):

```
Epoch 1, Loss: 1.0306
Epoch 2, Loss: 0.7767
Epoch 3, Loss: 0.5915
Epoch 4, Loss: 0.4568
Epoch 5, Loss: 0.3630
```

The first five epochs match the 10-epoch run in [Quickstart §3.4](../quickstart.md) exactly; that run's torch parity was verified 2026-10-27 vs torch 2.14.1+cpu (10-epoch AdamW trajectory, max per-epoch diff 1.38e-07). The three calls that must happen in this order every step are `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` — forgetting `zero_grad()` accumulates gradients across steps and the loss will not behave.

### GPU variant (unverified — REQUIRES-GPU)

> **REQUIRES-GPU.** The variant below was not run: there is no GPU on the docs verification VM. Run the pre-flight gate in §3 first; if the gate reports `gpu:0`, the only changes from the CPU run are the device selection and the data placement — the model, optimizer, loss, and loop are identical.

<!-- ex: id=local-4-2 device=gpu backend=cuda runtime=~1s torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

lt.manual_seed(42)
device = lt.Device("gpu:0")

model = lt.nn.Sequential([
    lt.nn.Linear(4, 8, True),
    lt.nn.ReLU(),
    lt.nn.Linear(8, 2, True)
])
model.to(device)

optimizer = lt.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = lt.nn.MSELoss()

x_train = lt.Tensor.from_vector([
    1.0, 2.0, 3.0, 4.0,
    2.0, 3.0, 4.0, 5.0,
    5.0, 6.0, 7.0, 8.0,
    8.0, 7.0, 6.0, 5.0
], [4, 4], device, False)

y_train = lt.Tensor.from_vector([
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.0, 1.0
], [4, 2], device, False)

dataset = lt.data.TensorDataset(x_train, y_train)
loader = lt.data.DataLoader(dataset, batch_size=2, shuffle=False, device=device)

for epoch in range(5):
    loader.reset()
    total_loss = 0.0
    steps = 0
    while True:
        batch = loader.next()
        if batch is None:
            break
        bx, by = batch
        optimizer.zero_grad()
        preds = model.forward(bx)
        loss = loss_fn.forward(preds, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    print(f"Epoch {epoch + 1}, Loss: {total_loss / max(steps, 1):.4f}")
```

Expected output on a single GPU is the same five loss values as the CPU run (`1.0306` down to `0.3630`) — same seed, same data, same math. If the GPU losses differ beyond 1e-4, that is a parity defect: report it, do not tune around it.

---

## 5. When to move to cloud

A single personal machine stops being the right place when any of these triggers fires:

- **More than one GPU is needed.** Multi-GPU training (NCCL collectives, 2-GPU allreduce, mismatch-throw) needs a host with two or more GPUs visible to one process — see [Installation on Kaggle](kaggle.md) for the 2-GPU reference setup.
- **The model or the data no longer fits local RAM / VRAM.** Activation checkpointing and memory eviction extend what one machine can hold, but they do not create VRAM.
- **Runs outlast the machine's availability.** Long training belongs on a host that does not sleep, reboot for updates, or share its GPU with a desktop session.
- **A TPU is needed.** There is no local TPU; TPU runs go through Google Cloud or Colab.

Cloud deployment guides (sibling pages in this batch):

- [AWS](aws.md)
- [Azure](azure.md)
- [GCP](gcp.md)
- [Other clouds](other-clouds.md)

For a free GPU to try a larger run before paying for cloud time, see [Installation on Kaggle](kaggle.md).

---

## 6. Troubleshooting

See [Troubleshooting and FAQ](../troubleshooting.md) for the full list. The local failures seen most often:

- `ModuleNotFoundError: No module named 'litetorch'` — the extension lives under `<tree>/python/`; add it to `sys.path`.
- Training runs on CPU despite a GPU being installed — run the §3 pre-flight gate; check `nvcc --version` and rebuild.
- A local source build is killed or never finishes — the full build does not complete in one session on a small machine (`plan/build-time.md`: killed at 44.6 min). Use `LT_BUILD_JOBS=1`, or build on Kaggle CPU.
- `pip install litetorch` seems to hang — it is compiling the C++ extension from the source distribution; on a 2-CPU machine this took ~28 minutes on 2026-10-27. Wait, or build from source explicitly so the progress is visible.
