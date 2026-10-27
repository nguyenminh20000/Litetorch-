# Installation on Other Clouds

This page covers running LiteTorch on GPU clouds beyond Docker and Kaggle: Lambda Labs (Lambda Cloud), Oracle Cloud Infrastructure (OCI), and a short table of other providers. The LiteTorch install itself is the same on every cloud: pip install ([Installation Guide](../installation.md) §2.2) or build from source ([Installation Guide](../installation.md) §2.4). This page only covers getting the GPU machine and the LiteTorch-specific checks that follow. It does not duplicate the install steps.

> Status note: every cloud provisioning step and every cloud GPU run on this page is `status=unverified`. The author has no account and no credentials on any of these platforms, so nothing cloud-side was run. Cloud facts below come only from each provider's official documentation, cited per claim. No prices are stated: check the provider's pricing page at run time.

---

## 1. Lambda Labs (Lambda Cloud)

Lambda's public cloud lets you launch individual virtual machines or clusters and turn them down on your schedule ([Lambda Docs: Public Cloud](https://docs.lambda.ai/public-cloud/)).

- **On Demand Cloud**: instantly launch a Linux-based, GPU-backed virtual machine instance using a variety of NVIDIA chip configurations ([Lambda Docs: Public Cloud](https://docs.lambda.ai/public-cloud/)).
- **1-Click Clusters**: clusters of GPU instances consisting of 16 to 512 NVIDIA H100 SXM Tensor Core GPUs ([Lambda Docs: Public Cloud](https://docs.lambda.ai/public-cloud/)). 1-Click Cluster nodes ship with Ubuntu 22.04 LTS and Lambda Stack preinstalled, including NCCL, Open MPI, PyTorch with DDP and FSDP support, and TensorFlow ([Lambda Docs: 1-Click Clusters](https://docs.lambda.ai/public-cloud/1-click-clusters/)).

How it works, at outline level:

1. Create a Lambda account and add an SSH key.
2. Launch an On Demand instance (or a 1-Click Cluster for multi-node work) from the console, CLI, or API.
3. Connect over SSH.
4. Install LiteTorch with pip ([Installation Guide](../installation.md) §2.2.2) or build it from source ([Installation Guide](../installation.md) §2.4). Check the CUDA toolkit first — see §4 below.
5. Terminate the instance when done. On Demand instances bill while they exist; confirm what keeps billing after you stop one (storage, IPs) on the provider's pricing and billing pages before you walk away.

Connect and check the toolkit:

<!-- ex: id=other-1 device=gpu backend=cuda runtime=~5s torch-parity=n/a status=unverified -->

```bash
ssh ubuntu@<INSTANCE-IP>
nvcc --version
nvidia-smi
```

The exact GPU models available on demand change over time; check the provider's instance list and pricing page at run time instead of trusting a snapshot here.

---

## 2. Oracle Cloud Infrastructure (OCI)

OCI rents GPU compute as named shapes. The shape name encodes the form factor (`VM.` virtual machine, `BM.` bare metal), the GPU family, and the GPU count. Bare metal shapes provide dedicated hardware with no virtualization overhead and support multi-GPU tensor parallelism; virtual machine shapes are more cost-effective for smaller models ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).

| Shape | GPUs | GPU | Total GPU memory |
| :--- | :---: | :--- | :---: |
| `VM.GPU.A10.1` | 1 | NVIDIA A10 | 24 GB |
| `VM.GPU.A10.2` | 2 | NVIDIA A10 | 48 GB |
| `BM.GPU4.8` | 8 | NVIDIA A100 40 GB | 320 GB |
| `BM.GPU.A100-v2.8` | 8 | NVIDIA A100 80 GB | 640 GB |
| `BM.GPU.H100.8` | 8 | NVIDIA H100 | 640 GB |

Table source: [Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/). `BM.GPU.H100.8` is listed there with RDMA support. `BM.GPU.GB200.4` is additionally listed by Oracle as a shape that supports compute clusters ([Oracle Docs: Compute Clusters](https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/compute-clusters.htm)); this page states no GPU count or memory for it because the cited page does not give them. OCI adds shapes over time — confirm the current shape list in the OCI Console or CLI before planning.

How it works, at outline level:

1. Create an OCI account and a compartment with permission to create compute instances.
2. Request GPU compute quota for the shape you want (for example `VM.GPU.A10.1` or `BM.GPU.A100-v2.8`) via a support ticket if you do not already have it ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).
3. Launch the instance in a region and availability domain where the shape is available. GPU capacity varies by region and availability domain; be prepared to try a different availability domain if you hit capacity errors ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).
4. Connect over SSH.
5. Install LiteTorch with pip ([Installation Guide](../installation.md) §2.2.2) or build it from source ([Installation Guide](../installation.md) §2.4). Check the CUDA toolkit first — see §4 below.
6. Terminate the instance (and any attached volumes) when done. These are paid GPU resources; always run the cleanup steps to avoid ongoing charges ([Oracle Docs: Deploy OpenAI vLLM Production Stack on OKE](https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/)).

Check the GPU and the toolkit on the instance:

<!-- ex: id=other-2 device=gpu backend=cuda runtime=~5s torch-parity=n/a status=unverified -->

```bash
nvidia-smi
nvcc --version
```

No prices are stated here: check the provider's pricing page at run time.

---

## 3. Other providers

| Provider | What it offers | Official docs | Status |
| :--- | :--- | :--- | :--- |
| RunPod | GPU Pods (container instances with a chosen GPU type and count) plus Serverless endpoints; deploy a Pod from the console, the `runpodctl` CLI, or the REST API | [RunPod Docs: Manage Pods](https://docs.runpod.io/pods/manage-pods) | `status=unverified` |
| Paperspace | GPU-powered virtual machines (Machines) and a Kubernetes-based container service, plus Notebooks, Deployments, and Workflows, for developing, training, and deploying AI applications | [DigitalOcean Docs: Paperspace](https://docs.digitalocean.com/products/paperspace/) | `status=unverified` |
| CoreWeave | Cloud platform purpose-built for AI and high-performance compute; CoreWeave Kubernetes Service (CKS) runs bare-metal clusters on its GPU and CPU fleet, and SUNK runs Slurm on Kubernetes for distributed training | [CoreWeave Docs: Get started](https://docs.coreweave.com/get-started) | `status=unverified` |

The rows above state only what the linked official docs page establishes. GPU lineups, regions, quotas, and prices for these providers were not verified for this page; check the provider's own docs and pricing page at run time.

---

## 4. LiteTorch-specific notes (all clouds)

These apply on every cloud above, exactly as on Kaggle and Docker.

- **Check the CUDA toolkit before any GPU run.** Run `nvcc --version` first and install the CUDA Toolkit if it is missing. A missing `nvcc` means the native GPU library never builds, and the framework then falls back to CPU silently.
- **GPU pre-flight gate.** Before trusting any GPU result, confirm both of these:
  1. The `Native GPU (CUDA/HIP)` init line is present (NOT OpenCL).
  2. `liblitetorch_gpu.so` sits next to the extension under `python/` (the `python/` copy is what `dlopen` finds). On some images `build_ext --inplace` writes it under `build/` instead; copy it to `python/` before importing.
- **Device string is `gpu:N`, not `cuda:N`.** `str(t.device)` returns `gpu:0`, `gpu:1`, and so on (D-50). The parser also accepts `cuda` as input, but the canonical string a device prints is `gpu:N`. Do not assert `cuda:0` in tests or scripts.

Run this pre-flight check after install (add `<tree>/python` to `sys.path` first when importing a source-built tree):

<!-- ex: id=other-3 device=gpu backend=cuda runtime=~1s torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "python")
import litetorch as lt

print(lt.get_backend_name())
print(lt.auto_device())
print(lt.is_gpu_available())
```

On a GPU machine the lines read a CUDA backend name, `gpu:0`, and `True`. If the device prints `cpu:0` where you expected a GPU, the GPU backend did not load — check `nvcc --version`, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line appears. Do not assume the GPU is in use: `.to(device)` falls back to CPU silently when the accelerator is unavailable.

---

## 5. Troubleshooting

See [Troubleshooting and FAQ](../troubleshooting.md) for build failures, out of memory, silent CPU fallback, and backend detection. Cloud-specific additions:

- The GPU backend did not load (`cpu:0` on a GPU machine) — re-check `nvcc --version` on the instance, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line and the `liblitetorch_gpu.so` placement in §4.
- `ModuleNotFoundError: No module named 'litetorch'` — `sys.path` does not include `<tree>/python` (source build) or the pip install did not complete; see [Installation Guide](../installation.md) §2.4.
- Out of host capacity / shape unavailable in the chosen region or availability domain — try a different availability domain or a different shape; see the OCI section above.

---

## Sources

Official documentation consulted 2026-10-27 (all cloud steps `status=unverified`; no account, no credentials, no cloud run):

- Lambda Docs, Public Cloud overview: https://docs.lambda.ai/public-cloud/
- Lambda Docs, 1-Click Clusters: https://docs.lambda.ai/public-cloud/1-click-clusters/
- Oracle Docs, Deploy OpenAI vLLM Production Stack on Oracle Kubernetes Engine (OKE): https://docs.oracle.com/en/learn/deploy-vllm-production-stack-oke/
- Oracle Docs, Compute Clusters: https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/compute-clusters.htm
- RunPod Docs, Manage Pods: https://docs.runpod.io/pods/manage-pods
- RunPod Docs, Deploy your first Pod: https://docs.runpod.io/get-started
- DigitalOcean Docs, Paperspace: https://docs.digitalocean.com/products/paperspace/
- CoreWeave Docs, Get started with CoreWeave: https://docs.coreweave.com/get-started
