# Installation on Azure

This page covers running LiteTorch on Microsoft Azure: GPU virtual machines (the primary path), Azure Machine Learning compute (brief), and Azure Container Instances (brief, retired — do not use for new deployments). For the base install steps, see [Installation Guide](../installation.md); for containers in general, see [Installation on Docker](docker.md).

> Status note: every cloud provisioning step and every cloud GPU run on this page is `status=unverified`. This docs VM has no Azure account and no Azure credentials, so nothing cloud-side was provisioned or run here. No Docker binary exists on this VM, so the Docker runtime is unverified here as well. Cloud facts on this page come from Microsoft's official documentation only, cited per claim in [Sources](#sources). No prices are stated: prices change fast — check the [Azure pricing page for Linux virtual machines](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/) before choosing a size. No quota, region availability, or SLA is stated except where the cited official page states it.

---

## 1. Which Azure path

| Path | Use it for | Status on this page |
| :--- | :--- | :--- |
| GPU virtual machine (NC/ND series) | Full control: training, multi-GPU runs, custom builds | Primary path, §2–§4 |
| Azure Machine Learning compute | Managed training jobs on a GPU cluster | Brief outline, §5 |
| Azure Container Instances with GPU | — | Retired 2025-07-14, do not use, §6 |

---

## 2. GPU VM sizes: NC and ND series

Azure's GPU-accelerated VM families are NC, ND, NG, and NV ([Virtual machine sizes overview](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)). For LiteTorch, NC and ND are the relevant families: NC is designed for compute-intensive workloads such as AI and machine learning model training, and ND for deep learning, AI research, and high-performance computing ([NC family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family), [ND family](https://learn.microsoft.com/en-gb/azure/virtual-machines/sizes/gpu-accelerated/nd-family)).

### 2.1. NC series → GPU

| Series | GPU (per official docs) | GPUs per VM |
| :--- | :--- | :--- |
| NCads_H100_v5 | NVIDIA H100 NVL, 94 GB each | 1–2 |
| NCCads_H100_v5 | NVIDIA H100 NVL, 94 GB | 1 |
| NC_A100_v4 | NVIDIA A100 PCIe, 80 GB each | 1–4 |
| NCasT4_v3 | NVIDIA Tesla T4, 16 GB each | 1–4 |
| NCv3 | NVIDIA Tesla V100, 16 GB each | 1–4 |

Source: [NC family VM size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family). NCads_H100_v5 is the current generation Microsoft names for new capacity: the NC_A100_v4 page notes Microsoft is "currently only deploying net new capacity for the latest generation of the NC product line, the NCads_H100_v5-series" ([NC_A100_v4 series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series)).

Two warnings from the official pages:

- **NCv3 is retired.** Microsoft retires the NCv3-series on September 30, 2025; see the NCv3-series retirement page for migration recommendations ([NCv3 series](https://learn.microsoft.com/sl-si/azure/virtual-machines/sizes/gpu-accelerated/ncv3-series)). Do not start new work on NCv3.
- **Quota starts at 0 for NCv3.** For NCv3, "the vCPU (core) quota in your subscription is initially set to 0 in each region. Request a vCPU quota increase for this series in an available region" ([NC family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family)). Expect a quota request for GPU series in general; check availability for your region before planning a run.

### 2.2. ND series → GPU

| Series | GPU (per official docs) | GPUs per VM |
| :--- | :--- | :--- |
| ND_H100_v5 | NVIDIA H100 Tensor Core, 80 GB each | 8 |
| ND_A100_v4 | NVIDIA Ampere A100, 40 GB each | 8 |
| NDm_A100_v4 | NVIDIA Ampere A100, 80 GB each | 8 |
| NDv2 | NVIDIA Tesla V100 NVLink, 32 GB each | 8 |
| ND (v1) | NVIDIA Tesla P40, 24 GB each | 1–4 |
| ND_MI300X_v5 | AMD Instinct MI300X, 192 GB each | 8 |

Source: [ND family VM size series](https://learn.microsoft.com/en-gb/azure/virtual-machines/sizes/gpu-accelerated/nd-family). ND_A100_v4, NDm_A100_v4, and ND_H100_v5 are built for scale-up (8 GPUs per VM) and scale-out (multiple VMs over InfiniBand, supporting GPU Direct RDMA and NCCL), per the same page. ND_MI300X_v5 is AMD hardware: it is the ROCm path, not CUDA — LiteTorch's CUDA notes on this page do not apply to it.

### 2.3. Choosing a size

- **Single-GPU development or inference:** NCasT4_v3 (T4) or NC_A100_v4 (A100). The Azure Machine Learning PyTorch training doc uses `Standard_NC4as_T4_v3` as its GPU example size ([Train deep learning PyTorch models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2&WT.mc_id=docs-article-lazzeri)).
- **Single-VM multi-GPU training:** NC_A100_v4 (up to 4 GPUs) or ND_A100_v4 / ND_H100_v5 (8 GPUs).
- **Multi-VM distributed training:** ND series with InfiniBand (ND_A100_v4, NDm_A100_v4, ND_H100_v5), which the official ND page describes as supporting NCCL-based clustering.

Check which sizes your region actually offers before provisioning; the official overview gives the Azure CLI command for that:

<!-- ex: id=az-1 device=cpu backend=cpu runtime=~10s torch-parity=n/a status=unverified -->

```bash
az vm list-skus --location <region> --output table
```

This command is quoted from [Virtual machine sizes overview](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu). It was not run here: no Azure account, no credentials.

---

## 3. VM setup: image, driver, CUDA toolkit

### 3.1. Base image

Use an Ubuntu LTS image. The official driver page lists Ubuntu 24.04 LTS, 22.04 LTS, and 20.04 LTS as supported Linux OS versions for the GRID drivers on NCasT4_v3 ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Ubuntu 22.04 matches the LiteTorch Docker CUDA image ([Installation on Docker](docker.md)) and the base install guide ([Installation Guide](../installation.md) §2.2).

### 3.2. NVIDIA driver

To use the GPU on an Azure N-series VM backed by NVIDIA GPUs, NVIDIA GPU drivers must be installed. The NVIDIA GPU Driver Extension installs the appropriate NVIDIA CUDA or GRID drivers on an N-series VM, and can be installed and managed through the Azure portal or Azure PowerShell ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Two limits from the official extension page ([NVIDIA GPU Driver Extension — Azure Linux VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux)):

- The extension does not automatically update the driver after installation; to move to a newer driver, manually download and install it, or remove and re-add the extension.
- Secure Boot is not supported using the Windows or Linux extensions.

Which driver type: NVIDIA distributes CUDA drivers for NCasT4_v3, NC_A100_v4, and NCads_H100_v5 VMs; for CUDA driver installation the official page refers to the NVIDIA website ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). On NCasT4_v3, the Azure NVIDIA GPU driver extension installs CUDA drivers; for graphics/visualization workloads on the same series, manually install the Azure-supported GRID drivers instead ([NC family](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family)). For LiteTorch compute workloads, install the CUDA driver, not GRID.

Verify the driver after installation:

<!-- ex: id=az-2 device=gpu backend=cuda runtime=~10s torch-parity=n/a status=unverified -->

```bash
nvidia-smi
```

Per the official driver page, if the driver is installed, NVIDIA SMI lists **GPU-Util** as N/A until you run a GPU workload on the VM ([Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)). Not run here: no Azure VM exists.

### 3.3. CUDA toolkit — check before any GPU run

The driver alone is not enough. LiteTorch needs the CUDA toolkit (`nvcc`, cuBLAS, cuBLASLt) to build its native GPU library: CUDA Toolkit 11.8 or 12.x ([Installation Guide](../installation.md) §2.1). This is a standing rule, not a suggestion: always check (and install if missing) the CUDA toolkit explicitly before a GPU CUDA run. A missing `nvcc` means the native GPU library never builds, and the framework then falls back to CPU silently.

<!-- ex: id=az-3 device=gpu backend=cuda runtime=~5s torch-parity=n/a status=unverified -->

```bash
nvcc --version
```

If `nvcc` is not found, install the CUDA Toolkit for your distribution (for example, the NVIDIA CUDA 12.x network repository for Ubuntu) and re-run `nvcc --version` until it prints a version. Only then build or run LiteTorch on GPU. Not run here: no Azure VM exists.

---

## 4. Install LiteTorch on the VM

Do not duplicate the base steps here: install LiteTorch exactly as in the base guide — pip install ([Installation Guide](../installation.md) §2.2) or build from source ([Installation Guide](../installation.md) §2.4). The VM-specific facts are:

1. **Extension placement.** `setup.py` sets `package_dir={"": "python"}`, so `build_ext --inplace` puts the importable extension under `<tree>/python/`, not the tree root. Every script must add `<tree>/python` to `sys.path` before `import litetorch` ([Installation Guide](../installation.md) §2.4). Confirm the built files are there:

<!-- ex: id=az-4 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=unverified -->

```bash
find python -maxdepth 1 -name "*.so"
```

2. **Pre-flight gate.** Before any GPU run, confirm the `Native GPU (CUDA/HIP)` init line appears at import and that `liblitetorch_gpu.so` sits next to the extension under `python/`. If the init line is absent, the GPU backend did not load — check `nvcc --version`, rebuild, and re-check. Do not assume the GPU is in use: `.to(device)` falls back to CPU silently when the accelerator is unavailable ([Installation Guide](../installation.md) §2.6).
3. **Device string.** LiteTorch's canonical device string is `gpu:N`, not `cuda:N`: `str(t.device)` prints `gpu:0` for the first GPU. The parser accepts both `cuda` and `gpu` as input, but output and `auto_device()` use `gpu:N`. Do not write `cuda:0` in LiteTorch code and do not compare against it.

Verification snippet (same code as [Installation Guide](../installation.md) §2.6, run here on the VM):

<!-- ex: id=az-5 device=gpu backend=cuda runtime=~1s torch-parity=1e-4 status=unverified -->

```python
import litetorch as lt

is_gpu = lt.is_gpu_available()
backend = lt.get_backend_name()
device = lt.auto_device()

print(f"GPU Available: {is_gpu}")
print(f"Backend Name: {backend}")
print(f"Active Device: {device}")

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
loss = lt.Ops.sum(y)
loss.backward()

print("Result shape:", y.shape)
print("Loss value:", loss.item())
print("Gradient:", x.grad.to_vector())
```

Expected output on a working Azure GPU VM (values are exact for this input; the first three lines are the gate):

<!-- ex: id=az-5-out device=gpu backend=cuda runtime=n/a torch-parity=1e-4 status=unverified -->

```
GPU Available: True
Backend Name: CUDA
Active Device: gpu:0
Result shape: [2, 2]
Loss value: 54.0
Gradient: [7.0, 11.0, 9.0, 13.0]
```

If `Active Device` prints `cpu:0` on a VM where you expected a GPU, the GPU backend did not load — re-check §3.3, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line. The loss and gradient values above are the torch-parity reference from the CPU-verified run in the base guide; the GPU run itself is unverified — no Azure GPU run has been executed.

When you are done, stop or delete resources you no longer need. Do not leave a GPU VM running idle.

---

## 5. Azure Machine Learning compute and jobs (brief)

Azure Machine Learning manages compute for you. The managed compute types are compute clusters, serverless compute, and compute instances ([Understand compute targets](https://learn.microsoft.com/da-dk/azure/machine-learning/concept-compute-target?view=azureml-api-2)):

- **Compute cluster:** a fully managed, autoscaling cluster for training jobs; single-node or multinode, CPU and GPU. Set the minimum number of nodes to 0 to avoid charges when the cluster is idle (same source).
- **Compute instance:** a single-node cluster, the Jupyter-notebook development machine attached to your workspace (same source). Enable idle shutdown to avoid charges when idle (same source).
- **Serverless compute:** offloads compute lifecycle management to Azure Machine Learning instead of creating a cluster (same source).

Creating a GPU cluster follows the official PyTorch training doc ([Train deep learning PyTorch models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2&WT.mc_id=docs-article-lazzeri)), which creates a Linux compute cluster of size `Standard_NC4as_T4_v3` (a T4 size from §2.1). Condensed from that doc, comments stripped:

<!-- ex: id=az-6 device=gpu backend=cuda runtime=~10m torch-parity=n/a status=unverified -->

```python
from azure.ai.ml.entities import AmlCompute

gpu_cluster = AmlCompute(
    name="gpu-cluster",
    type="amlcompute",
    size="STANDARD_NC4AS_T4_V3",
    min_instances=0,
    max_instances=4,
    idle_time_before_scale_down=180,
    tier="Dedicated",
)
gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()
```

`ml_client` is the workspace client from the same doc (`MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)`). The parameter names and values above are quoted from the official doc; nothing else was added. Job submission itself is outline only: define the training job (training script, environment, and the `gpu-cluster` compute target) and submit it from the studio, the Python SDK, or the Azure CLI, following the same official training doc. No CLI flags or SDK calls beyond the cited doc are given here.

LiteTorch-specific notes apply unchanged on Azure ML compute: check the CUDA toolkit before any GPU run, pass the pre-flight gate (`Native GPU (CUDA/HIP)` init line + `.so` placement), and use the `gpu:0` device string. The environment (container image or conda environment) must contain the CUDA toolkit and the built LiteTorch extension under `<tree>/python/` on `sys.path`, exactly as in §4. None of this has been run on Azure ML: no workspace, no account, no credentials.

---

## 6. Azure Container Instances (brief — retired)

Do not use Azure Container Instances GPU resources for new deployments. The official page states: "This product is retired as of July 14, 2025" ([Deploy GPU-Enabled Container Instance](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-gpu)). For the record, what the retired feature offered:

- GPU version V100 only, mapping to the NCv3 VM family; Linux only; GPU count 1, 2, or 4.
- Maximum resources per SKU: 1 GPU → 6 CPU / 112 GB memory / 50 GB storage; 2 GPUs → 12 CPU / 224 GB; 4 GPUs → 24 CPU / 448 GB.
- Container instances with GPU resources were pre-provisioned with NVIDIA CUDA drivers and container runtimes; the page supported up through CUDA 11.
- Default subscription limits (quotas) for GPU resources differed by version; the default CPU limits for V100 versions were initially set to 0, and a quota increase required an Azure Support request.

All of the above is quoted from the same retired official page, for historical reference only. Use a GPU VM (§2–§4) or Azure Machine Learning compute (§5) instead. The Docker runtime is unverified on this VM in any case: no Docker binary exists here, so no image build or container run in this guide was executed.

---

## 7. Troubleshooting

Full troubleshooting lives in [Troubleshooting](../troubleshooting.md). Azure-specific symptoms:

| Symptom | Check |
| :--- | :--- |
| `Active Device` prints `cpu:0` on a GPU VM | §3.3: `nvcc --version`; rebuild; confirm the `Native GPU (CUDA/HIP)` init line (§4) |
| `nvidia-smi` fails or shows no GPU | §3.2: driver extension installed; correct driver type (CUDA, not GRID, for compute) |
| `ModuleNotFoundError: No module named 'litetorch'` | `<tree>/python` not on `sys.path` (§4.1) |
| VM size unavailable, or quota error at creation | §2: `az vm list-skus` for the region; request a quota increase (NCv3 quota starts at 0 per §2.1) |
| ACI GPU deployment fails | §6: the feature is retired (2025-07-14); use a VM or Azure ML compute |

---

## Sources

Microsoft official documentation only. Fetched and read 2026-10-27. No prices, quotas, region availability, or SLAs are stated on this page beyond what these pages state.

- [Virtual machine sizes overview](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu) — GPU families (NC, ND, NG, NV); `az vm list-skus` command; pricing-page pointer.
- [NC family VM size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-family) — NCads_H100_v5 / NCCads_H100_v5 / NCv3 / NCasT4_v3 / NC_A100_v4 GPU mappings; NCv3 quota starts at 0; CUDA vs GRID driver note for NCasT4_v3.
- [NC_A100_v4 size series](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series) — NCads_H100_v5 is the current generation receiving new capacity.
- [NCv3 size series](https://learn.microsoft.com/sl-si/azure/virtual-machines/sizes/gpu-accelerated/ncv3-series) — NCv3 retirement on September 30, 2025.
- [ND family VM size series](https://learn.microsoft.com/en-gb/azure/virtual-machines/sizes/gpu-accelerated/nd-family) — ND / NDv2 / ND_A100_v4 / NDm_A100_v4 / ND_H100_v5 / ND_MI300X_v5 GPU mappings; InfiniBand and NCCL scale-out.
- [Azure N-series GPU driver setup for Linux](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup) — driver extension installs CUDA or GRID drivers; CUDA drivers distributed for NCasT4_v3, NC_A100_v4, NCads_H100_v5; `nvidia-smi` verification; supported Ubuntu LTS versions.
- [NVIDIA GPU Driver Extension — Azure Linux VMs](https://learn.microsoft.com/en-us/azure/virtual-machines/extensions/hpccompute-gpu-linux) — no automatic driver updates; Secure Boot not supported with the extensions.
- [Train deep learning PyTorch models (SDK v2)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch?view=azureml-api-2&WT.mc_id=docs-article-lazzeri) — `AmlCompute` GPU cluster example (`STANDARD_NC4AS_T4_V3`, min 0, max 4, idle 180 s, Dedicated).
- [Understand compute targets](https://learn.microsoft.com/da-dk/azure/machine-learning/concept-compute-target?view=azureml-api-2) — compute cluster vs compute instance vs serverless compute; min nodes 0 and idle shutdown.
- [Deploy GPU-Enabled Container Instance](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-gpu) — ACI GPU retired 2025-07-14; V100 only, NCv3 family, Linux only, 1/2/4 GPUs, per-SKU maxima, CUDA 11 support, default quota 0.
- [Pricing — Linux Virtual Machines](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/) — check current prices here; no price is quoted on this page.
