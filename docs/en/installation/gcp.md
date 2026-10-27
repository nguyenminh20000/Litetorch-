# Installation on Google Cloud Platform

Google Cloud Platform (GCP) gives LiteTorch a GPU without local hardware: a Compute Engine VM with an attached NVIDIA GPU, or a managed Vertex AI training job and endpoint. This page covers choosing the GPU, creating the VM (Deep Learning VM image or plain Ubuntu/Debian plus a manual CUDA install), the LiteTorch install, and the GPU pre-flight gate.

> Status note: every GCP provisioning step and every GCP GPU run on this page is `status=unverified`. This docs VM has no GCP account and no GCP credentials, so no cloud step on this page was executed. Cloud facts are cited to Google Cloud's official documentation per claim; check the linked official page before acting, because GPU offerings, image families, and quotas change. No prices are stated on this page — check the [Google Cloud GPU pricing page](https://cloud.google.com/compute/gpus-pricing).

---

## 1. Choose the GPU and machine series

Only accelerator-optimized (A4X, A4, A3, A2, G4, G2) and general-purpose N1 machine types support GPUs on Compute Engine. For A and G series machine types the GPU model is automatically attached to the instance; for N1 you attach a GPU model to the instance ([About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)).

| Machine series | GPU model | `accelerator` type string |
| :--- | :--- | :--- |
| A4X Max | NVIDIA GB300 Ultra Superchip | `nvidia-gb300` |
| A4X | NVIDIA GB200 Superchip | `nvidia-gb200` |
| A4 | NVIDIA B200 | `nvidia-b200` |
| A3 Ultra | NVIDIA H200 | `nvidia-h200-141gb` |
| A3 Mega | NVIDIA H100 | `nvidia-h100-mega-80gb` |
| A3 High / A3 Edge | NVIDIA H100 | `nvidia-h100-80gb` |
| A2 Ultra | NVIDIA A100 80GB | `nvidia-a100-80gb` |
| A2 Standard | NVIDIA A100 40GB | `nvidia-tesla-a100` |
| G4 | NVIDIA RTX PRO 6000 | `nvidia-rtx-pro-6000` |
| G2 | NVIDIA L4 | `nvidia-l4` |
| N1 (attachable) | NVIDIA T4 | `nvidia-tesla-t4` |
| N1 (attachable) | NVIDIA V100 | `nvidia-tesla-v100` |
| N1 (attachable) | NVIDIA P4 | `nvidia-tesla-p4` |

Source: [About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus). Two warnings from the same page: NVIDIA T4 and NVIDIA P4 are approaching end of support, and the Compute Engine SLA covers an instance with an attached GPU model only if that GPU model is generally available.

How to pick:

- **L4 (G2)** — the cheapest current-generation option for development, small-model training, and single-host inference. G2 also supports custom memory.
- **A100 (A2)** — training larger models and multi-GPU single-host work.
- **H100 / H200 / B200 (A3 / A4)** — large-model training and multi-host work; these series have consumption-option and region restrictions, so read the per-series limitations on the [create-VM overview](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus) before planning around them.
- **T4 (N1)** — the budget attachable GPU, and the closest match to the Kaggle GPU track (2x T4). Prefer L4 over T4 for new work because T4 is approaching end of support.

Not every GPU type is available in every region and zone, and new projects have a global GPU quota that limits the total number of GPUs you can create in any supported zone. When you request GPU quota, you must request quota for the GPU types you want in each region plus an additional global quota ([Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)). If instance creation fails with a quota error, that is the cause — request a quota increase; it is not a LiteTorch problem.

Spot VMs with GPUs are available at lower prices, but the VM is preempted on host maintenance and cannot be automatically restarted ([About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)). Use Spot only for fault-tolerant runs that can be re-created.

---

## 2. Path A: Deep Learning VM image (recommended)

A Deep Learning VM (DLVM) image is a VM image optimized for data science and machine learning, with key ML frameworks and tools pre-installed, usable out of the box on instances with GPUs ([Introduction to Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs/introduction)). Images are based on Debian 11 and Ubuntu 22.04 and include Python 3.10 with numpy, scipy, matplotlib, pandas, scikit-learn and others, a JupyterLab environment, and — on GPU-enabled images — the NVIDIA driver, CUDA 11.x and 12.x, cuDNN, and NCCL 2.x (versions depend on the framework).

Current image families ([Choose an image](https://cloud.google.com/deep-learning-vm/docs/images)):

| Framework | Processor | Image family |
| :--- | :--- | :--- |
| Base | GPU | `common-cu128-ubuntu-2204-nvidia-570` |
| PyTorch | GPU | `pytorch-2-7-cu128-ubuntu-2204-nvidia-570` |

Ubuntu 22.04 is the default OS and the images ship with NVIDIA driver 570; Debian images have all been deprecated. Image family names change as new framework versions are released — always confirm the current family on the [Choose an image](https://cloud.google.com/deep-learning-vm/docs/images) page before creating the instance, and reference the family (not a fixed image) to get the most recent version.

Create a DLVM instance with one T4 attached (pattern from [Create a Deep Learning VM instance from the command line](https://cloud.google.com/deep-learning-vm/docs/cli)):

<!-- ex: id=gcp-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
export IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
export ZONE="us-west1-b"
export INSTANCE_NAME="litetorch-gcp"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-t4,count=1" \
  --metadata="install-nvidia-driver=True"
```

Rules in that command, all from the official DLVM documentation:

- `--image-project` must be `deeplearning-platform-release`.
- `--maintenance-policy` must be `TERMINATE` — Compute Engine always stops instances with attached GPUs on host maintenance events.
- `--accelerator` takes the form `type=TYPE,count=COUNT`; use the type strings from the table in §1.
- `--metadata="install-nvidia-driver=True"` installs the NVIDIA driver at boot.
- Replace `us-west1-b` with a zone that supports the GPU type you chose; not all GPU types are supported in all regions.

Limit: Deep Learning VM Images cannot be used as boot disks for instances that use the G2 machine type ([Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)). For an L4, use Path B below.

---

## 3. Path B: Plain Ubuntu/Debian + manual CUDA install

Pick this path when the DLVM image does not fit: G2 (L4) instances, a CUDA version the DLVM image does not ship, or a minimal base image you control fully. You install the NVIDIA driver and the CUDA Toolkit yourself; for most public or custom images you need to install both ([Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)).

Create a G2 (L4) instance from a plain image (pattern from [Create a G2 or G4 instance](https://cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series)):

<!-- ex: id=gcp-2 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
gcloud compute instances create litetorch-gcp-l4 \
  --machine-type=g2-standard-12 \
  --zone=us-west1-b \
  --boot-disk-size=200GB \
  --image-family=rocky-linux-8-optimized-gcp \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

For a G2 instance the GPU is attached automatically with the machine type, so there is no `--accelerator` flag. `--maintenance-policy=TERMINATE` is required for GPU instances. The zone must support the L4 GPU. The `--image-family` above is the example image family from the official G2 documentation — any OS image that supports GPUs works; Ubuntu 22.04 is the other common choice.

Then, on the VM, install the NVIDIA driver and the CUDA Toolkit for your distribution and confirm them before doing anything else:

<!-- ex: id=gcp-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` verifies the driver sees the GPU; `nvcc --version` verifies the CUDA Toolkit. If `nvcc` is not found, install the CUDA Toolkit for your distribution and re-run until it prints a version. This is the standing LiteTorch rule: always check (and install if missing) the CUDA toolkit explicitly before any GPU CUDA run — a missing `nvcc` means the native GPU library never builds and the framework falls back to CPU silently.

When to pick which path:

| | Deep Learning VM (§2) | Plain image + manual install (§3) |
| :--- | :--- | :--- |
| Driver + CUDA | Pre-installed | You install both |
| Frameworks | PyTorch and data-science packages pre-installed | Only what you install |
| Works on G2 (L4) | No | Yes |
| Control over CUDA version | No (image decides) | Yes |
| Best for | Fastest path to a working GPU VM | L4, pinned CUDA, minimal images |

---

## 4. Install LiteTorch on the VM

Once the VM is up and the driver and CUDA Toolkit are confirmed, the LiteTorch install is the same as on any Linux machine. Do not re-read duplicated steps here — follow the main guide:

- **pip install:** [Installation §2.2](../installation.md#22-linux-and-google-colab) (`pip install --upgrade litetorch`).
- **Build from source:** [Installation §2.4](../installation.md#24-build-from-source) — clone, `pip install pybind11`, `python3 setup.py build_ext --inplace`.

<!-- ex: id=gcp-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
git clone https://github.com/nguyenminh20000/Litetorch-.git
cd Litetorch-
pip install pybind11
python3 setup.py build_ext --inplace
```

Two build facts from the main guide that cause most first-time failures:

1. **`setup.py` sets `package_dir={"": "python"}`, so the built `.so` lands under `<tree>/python/`.** Every script must add `<tree>/python` to `sys.path` before `import litetorch`, or the import fails with `ModuleNotFoundError`.
2. **On GPU images, `build_ext --inplace` can write `liblitetorch_gpu.so` into `build/lib.linux-*/` instead of next to the extension.** Copy it next to the extension under `python/` before import — the `python/` copy is what `dlopen` finds:

<!-- ex: id=gcp-5 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
cp build/lib.linux-*/liblitetorch_gpu.so python/
ls python/
```

---

## 5. GPU pre-flight gate

Before running any GPU example on the VM, assert all three — the same gate as the Kaggle track:

<!-- ex: id=gcp-6 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
python3 -c "import sys; sys.path.insert(0, 'python'); import litetorch as lt; print(lt.get_backend_name()); print(lt.cuda.device_count())"
```

1. The `Native GPU (CUDA/HIP)` init line is present (NOT OpenCL) — this catches the missing-native-library failure.
2. `liblitetorch_gpu.so` sits next to the extension under `python/`.
3. `lt.cuda.device_count()` returns the expected count (1 on the single-GPU VMs in §2–§3).

If any assertion fails, stop and report it as a harness / blocker issue, not as example failures. The common cause is §3's `nvcc` check skipped: no CUDA Toolkit at build time means no native GPU library, and `.to(device)` then falls back to CPU silently.

Device strings: LiteTorch's canonical GPU device string is `gpu:N`, not `cuda:N` — `str(t.device)` is `gpu:0` on the first GPU and `cpu:0` on CPU. The parser also accepts `cuda:0` as input, but never expect `cuda:0` in output. If the smoke test below prints `cpu:0` where you expected a GPU, the GPU backend did not load.

---

## 6. Smoke test

The snippet below is self-contained and is the same smoke test as the Kaggle page. On the GCP GPU VM it exercises the import path and the GPU matmul.

<!-- ex: id=gcp-7 device=gpu backend=cuda runtime=~1s torch-parity=1e-4 status=unverified -->

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

Expected output on the GPU VM:

```
Backend: CUDA
Device: gpu:0
GPU available: True
Result: [7.0, 10.0, 15.0, 22.0]
Sum: 54.0
```

(The backend name is the CUDA backend name; the result and sum are identical on CPU and GPU. The same snippet is verified on CPU locally — see [Installation §2.6](../installation.md#26-verification-script) — but the GPU run of it on GCP is unverified until a GCP GPU run executes it.)

---

## 7. Vertex AI training and endpoints (brief)

For managed training and serving instead of a self-managed VM, Vertex AI offers custom training jobs and endpoints. This section is an outline only — no LiteTorch-specific Vertex API exists yet, and no step here has been run.

The official flow ([Train and deploy your model](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/train-and-deploy-model), [Prediction classes](https://cloud.google.com/vertex-ai/docs/python-sdk/prediction-classes)) is:

1. **Custom training job** — a `CustomTrainingJob` takes your training script (`script_path`), a training container image (`container_uri`), and optional `requirements`, then `job.run(...)` starts training and returns a trained `Model`.
2. **Deploy to an endpoint** — before you can get online predictions, you must deploy the model to an endpoint; deploying associates the physical machine resources with the model so it can serve online predictions. `model.deploy(...)` creates and returns the `Endpoint`.
3. **Predict** — `endpoint.predict(instances=...)` returns predictions for new inputs.

LiteTorch-specific notes, all unverified:

- Vertex AI has no prebuilt LiteTorch training or prediction container. A custom container is required — the CUDA image in [Installation on Docker](docker.md) is the starting point for building one, but the container, the training script's LiteTorch install step, and the serving path have not been built or run.
- The same LiteTorch rules apply inside the training container: check the CUDA Toolkit before any GPU run, confirm the `Native GPU (CUDA/HIP)` init line and the `.so` placement in the pre-flight gate, and expect the `gpu:0` device string.
- Deep Learning VM images and Deep Learning Containers can also be used as part of Vertex AI work, for example as a development environment for a self-managed distributed training system ([Use Deep Learning VM Images and Deep Learning Containers with Vertex AI](https://cloud.google.com/vertex-ai/docs/general/deep-learning)).

Do not invent Vertex AI CLI flags or SDK calls beyond what the cited official pages show — if the official page does not show it, it does not belong in a runbook yet.

---

## 8. Cost and cleanup

- **No prices are stated on this page.** Check the [Google Cloud GPU pricing page](https://cloud.google.com/compute/gpus-pricing) before creating anything; GPU VMs bill per second with a 1-minute minimum, and an idle but RUNNING instance is still billed.
- **Stop the VM when you are done** — `gcloud compute instances stop <name>` preserves the boot disk and stops billing for the VM (the DLVM documentation uses this command when snapshotting an instance). Delete the instance when the work is finished so nothing keeps billing.
- **Keep data off the VM.** Store non-transient data on persistent block storage (Persistent Disk / Hyperdisk) or in Cloud Storage, not on the VM's local SSDs — local SSD data is lost when the instance stops ([About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)).

---

## 9. Troubleshooting

- `ModuleNotFoundError: No module named 'litetorch'` — `<tree>/python` is not on `sys.path` (see §4, fact 1).
- The build prints `BUILD OK` but the import fails — `setup.py` prints `BUILD OK` even when per-file compiles fail. The real gate is the extension file existing under `<tree>/python/` and `import litetorch` succeeding.
- GPU run silently uses CPU — verify `nvcc --version`, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line (§5). The usual cause on a plain image (§3) is a missing CUDA Toolkit at build time.
- Instance creation fails with a quota error — GPU quota (§1). Request a quota increase for the GPU type in the region plus the global quota.
- Instance creation fails with a zone error — the GPU type is not available in that zone. Pick a zone from the GPU regions and zones availability page linked from the [create-VM overview](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus).
- Full troubleshooting reference: [Troubleshooting](../troubleshooting.md).

---

## 10. Where to go next

- First run on the VM: [Quickstart](../quickstart.md).
- Full install reference: [Installation](../installation.md).
- Container install: [Installation on Docker](docker.md).
- Notebook install: [Installation on Kaggle](kaggle.md).

---

## Sources

All cloud facts on this page come from Google Cloud's official documentation, retrieved 2026-10-27:

- GPU models and machine series, Spot VMs, host maintenance, GPU quota, SLA coverage: [About GPU instances](https://cloud.google.com/compute/docs/gpus/about-gpus)
- Creating a GPU VM, OS choice, G2/DLVM limitation, quota: [Overview of creating an instance with attached GPUs](https://cloud.google.com/compute/docs/gpus/create-vm-with-gpus)
- DLVM contents (frameworks, Python packages, CUDA/cuDNN/NCCL): [Introduction to Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs/introduction)
- Current DLVM image families: [Choose an image](https://cloud.google.com/deep-learning-vm/docs/images)
- DLVM instance creation flags: [Create a Deep Learning VM instance from the command line](https://cloud.google.com/deep-learning-vm/docs/cli)
- G2/G4 instance creation: [Create a G2 or G4 instance](https://cloud.google.com/compute/docs/gpus/create-gpu-vm-g-series)
- GPU pricing (linked, no prices quoted): [VM instance pricing](https://cloud.google.com/compute/gpus-pricing)
- Vertex AI custom training and endpoints: [Train and deploy your model](https://cloud.google.com/vertex-ai/docs/tutorials/tabular-bq-prediction/train-and-deploy-model), [Prediction classes](https://cloud.google.com/vertex-ai/docs/python-sdk/prediction-classes)
- DLVM with Vertex AI: [Use Deep Learning VM Images and Deep Learning Containers with Vertex AI](https://cloud.google.com/vertex-ai/docs/general/deep-learning)
