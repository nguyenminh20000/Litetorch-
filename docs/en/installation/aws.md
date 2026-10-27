# Installation on AWS

AWS gives LiteTorch GPU compute without local hardware, through four paths: a raw EC2 GPU instance, a managed SageMaker training job or endpoint, or a container on ECS or EKS. This page maps the paths, names the current EC2 GPU families, and gives the LiteTorch steps after the instance is up. It does not duplicate the pip and source-build steps: those live in [Installation](installation.md) §2.2 (pip) and §2.4 (build from source).

> Status note: every cloud provisioning step and every cloud GPU run on this page is `status=unverified`. The author has no AWS account and no credentials, so no instance was launched, no AMI was booted, and no GPU run was executed on AWS. The cloud facts below come from AWS official documentation, cited per claim; the LiteTorch checks are the same checks verified locally on CPU and on Kaggle GPU in the project runbook, applied here to an unlaunched target. Do not treat any step on this page as run-verified until a Batch B5 AWS verification item closes it. Docker is not installed on the docs verification VM, so the Docker runtime on this page is likewise unverified.

---

## 1. Which path to pick

| Path | What AWS runs for you | Pick it when |
| :--- | :--- | :--- |
| EC2 GPU instance (§2, §3) | A virtual server with NVIDIA GPUs; you manage the OS, drivers, and LiteTorch install | You want full control, interactive development, or a custom build |
| SageMaker training job (§4) | Managed training compute; you supply a script or a container | You want training without managing servers |
| SageMaker endpoint (§4) | A managed HTTPS inference endpoint | You want to serve a trained model |
| ECS / EKS (§5) | Container orchestration on GPU instances | You already ship LiteTorch as a container and want scaling |

No prices are stated on this page: they change fast. Check the [EC2 On-Demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for the instance type and region you plan to use before launching anything. No quota, region availability, or SLA is stated either; confirm those in the AWS console for your account and region at launch time.

---

## 2. EC2 GPU instances

AWS splits GPU instances into two series: **P** for ML training and HPC, **G** for graphics-intensive work and ML inference. The family-to-GPU mapping below is from the AWS Deep Learning AMI (DLAMI) recommended-GPU page, cross-checked against the per-family EC2 pages:

| Family | GPU | GPUs per instance (max) | Source |
| :--- | :--- | :--- | :--- |
| G4dn | NVIDIA T4 | 4 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| G5 | NVIDIA A10G | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| G6 | NVIDIA L4 Tensor Core, 24 GB per GPU | 8 | [EC2 G6 instances](http://aws.amazon.com/ec2/instance-types/g6/), [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| G6e | NVIDIA L40S Tensor Core | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P3 | NVIDIA V100 | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P4d | NVIDIA A100 Tensor Core | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) (P4 family), [ECS GPU instances](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu.html) (p4d: A100) |
| P5 | NVIDIA H100 Tensor Core, 80 GB HBM3 per GPU | 8 | [EC2 P5 instances](http://aws.amazon.com/ec2/instance-types/p5/), [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P5e / P5en | NVIDIA H200 Tensor Core | 8 | [EC2 P5 instances](http://aws.amazon.com/ec2/instance-types/p5/), [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |
| P6-B200 | NVIDIA Blackwell B200 | 8 | [DLAMI recommended GPU instances](https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html) |

Notes on the table:

- G6 is the current general-purpose inference family: the EC2 G6 page states the instances are "powered by NVIDIA L4 Tensor Core GPUs" for "deep learning inference and graphics-intensive applications", with up to 8 L4 GPUs at 24 GB each, plus fractional-GPU sizes (as little as 1/8 of an L4, 3 GB) for workloads that cannot use a whole GPU.
- P5 is the current large-training family: the EC2 P5 page states P5 provides up to 8 NVIDIA H100 GPUs (640 GB HBM3 total per instance) and P5e/P5en up to 8 NVIDIA H200 GPUs (1128 GB HBM3e total), with up to 3,200 Gbps EFA networking and NVSwitch GPU interconnect.
- P3 (V100) is a previous generation, still listed by the DLAMI guide; prefer G6 or P5 for new work unless you have a specific reason to use V100.
- P6-B200 (Blackwell) is the newest family listed by the DLAMI guide. LiteTorch's native GPU library has not been built or run against Blackwell on this project, so a P6-B200 run is unverified in both senses: no AWS run and no Blackwell build.
- G5g (Arm64, Graviton2) is listed by the DLAMI guide as a separate Arm64 family with its own DLAMI; the LiteTorch build steps in [Installation](installation.md) §2.4 are written for x86_64, so G5g is out of scope for this page.

LiteTorch guidance: start on a single-GPU G6 (inference, small training, development) or G5 (larger models, faster training); move to P4d or P5 when you need multi-GPU training or the largest GPU memory. The DLAMI guide itself advises that model size should drive the choice: if the model exceeds the instance's available memory, pick a different instance type.

---

## 3. Launching an EC2 instance: Deep Learning AMI vs plain Ubuntu

### 3.1. Deep Learning AMI (DLAMI)

A DLAMI is an AWS-provided machine image with the NVIDIA driver and deep learning software pre-installed: the GPU AMIs ship with the NVIDIA driver, CUDA Toolkit, NCCL, cuDNN, Docker, and the NVIDIA Container Toolkit pre-installed, plus a framework environment (PyTorch DLAMIs ship a PyTorch environment). Exact driver, CUDA, cuDNN, and PyTorch versions differ per DLAMI release: read them from the DLAMI release notes at launch time; this page pins no versions.

Launch steps, from the DLAMI developer guide:

1. Open the EC2 console.
2. Choose **Launch Instance**.
3. Enter a name and select the DLAMI (search by DLAMI ID, or pick from Quick Start / My AMIs).
4. Choose an instance type. The DLAMI release notes list the recommended instance families for each DLAMI.
5. Choose **Launch Instance**, then wait for the instance to be ready (usually a few minutes) and connect to it.

Pick the DLAMI when you want the fastest start: the driver and CUDA stack are already installed and matched to each other, so the usual failure (driver/CUDA version mismatch) is removed. Pick the PyTorch DLAMI, not the Base DLAMI, if you want a ready PyTorch environment alongside LiteTorch; pick the Base DLAMI if you want only the NVIDIA stack and intend to install everything else yourself.

### 3.2. Plain Ubuntu + manual CUDA install

Launch a plain Ubuntu AMI (Ubuntu 22.04 is the version the LiteTorch Docker images and the DLAMI PyTorch images use) and install the NVIDIA driver, CUDA Toolkit, and NVIDIA Container Toolkit yourself, following the AWS re:Post install guide for Ubuntu EC2 instances.

Pick plain Ubuntu when you need full control: a specific CUDA version, a minimal image with no framework pre-installed, or an install that matches the local build-from-source procedure in [Installation](installation.md) §2.4 exactly. The cost is that you own the driver/CUDA pairing; a mismatch fails the same way it fails locally (native GPU library never builds, silent CPU fallback — see §6).

### 3.3. After the instance is up: installing LiteTorch

Once you can SSH into the instance, the LiteTorch steps are the same as on any Ubuntu machine. Do not follow a separate AWS-only install procedure; follow [Installation](installation.md) §2.2 for pip (`pip install --upgrade litetorch`) or §2.4 for a source build (`git clone`, `pip install pybind11`, `python3 setup.py build_ext --inplace`). Before either path, run the checks in §6 — the CUDA toolkit check and the pre-flight gate — because they are the steps that catch a broken GPU setup before any training time is spent.

<!-- ex: id=aws-1 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
nvidia-smi
nvcc --version
ls /opt/litetorch/python/
```

`nvidia-smi` proves the driver sees the GPU; `nvcc --version` proves the CUDA toolkit is installed (the standing rule: check the CUDA toolkit before any GPU run). The `ls` proves the built extension lives under `<tree>/python/` — `setup.py` sets `package_dir={"": "python"}`, so the importable extension is under `<tree>/python/`, not the tree root. If `nvcc` is not found, install the CUDA Toolkit and re-run until it prints a version; only then build or run LiteTorch on GPU. If `nvidia-smi` fails, the problem is the driver, not LiteTorch: fix the driver first (DLAMI users: relaunch from the DLAMI rather than debugging the driver by hand).

---

## 4. SageMaker training jobs and endpoints (brief)

Amazon SageMaker AI is the managed ML service: it provisions the compute, runs your training, and serves the model, without you managing servers. SageMaker accepts any model, any framework, and any container.

- **Training jobs.** A training job runs your training code on managed compute (CPU, GPU, or distributed clusters). Two ways to bring LiteTorch code:
  - **Script mode:** bring your own training script; SageMaker pulls the appropriate AWS-managed container, downloads your script, and runs it. This is the low-effort route, but there is no AWS-managed LiteTorch container, so script mode fits only if the managed container already contains everything LiteTorch needs — it does not, for a source-built LiteTorch.
  - **Bring your own container (BYOC):** package your algorithm and its dependencies in a Docker image, push it to Amazon ECR, and SageMaker pulls it and runs training. This is the LiteTorch route: build the image from the LiteTorch CUDA Dockerfile in [Installation on Docker](installation/docker.md), push it to ECR, and point the training job at it. Outline only — the exact training-job definition and the container contract are unverified (no AWS run) and are not specified here; follow the SageMaker developer guide for the container requirements.
- **Endpoints.** A trained model is deployed as a real-time inference endpoint (a managed HTTPS endpoint) or as a batch transform job for predictions over a large dataset without a persistent endpoint. For a bring-your-own inference container, SageMaker requires the container to serve a web server on port `8080` and accept `POST` requests to the `/invocations` and `/ping` endpoints. LiteTorch has no inference server on this page, so an endpoint needs a serving wrapper around LiteTorch inference; that wrapper is out of scope and unverified.

LiteTorch fit, in one line: SageMaker training via BYOC with the LiteTorch CUDA image; SageMaker script mode does not apply to a source-built LiteTorch; endpoints need a serving wrapper that does not exist yet. No SageMaker API calls are given on this page — every SageMaker step is unverified.

---

## 5. ECS and EKS with the LiteTorch Dockerfile (brief)

### 5.1. ECS

Amazon Elastic Container Service runs containers on managed EC2 capacity. GPU support on ECS Managed Instances covers the `g4dn` (NVIDIA T4), `g5` (NVIDIA A10G), `p3` (NVIDIA V100), `p4d` (NVIDIA A100), and `g6f` (fractional NVIDIA L4) families, and the NVIDIA drivers and CUDA toolkit are pre-installed on the GPU-enabled instances. The outline: push the LiteTorch CUDA image (built from [Installation on Docker](installation/docker.md)) to ECR, write a task definition that requests GPU resources, and run it as an ECS service or task on a GPU capacity provider. Outline only — no task definition is given, and no ECS run has been executed.

### 5.2. EKS

Amazon Elastic Kubernetes Service runs Kubernetes with GPU node groups. The EKS documentation shows GPU NodePools selecting the `g6` and `g6e` families, pods requesting GPUs through the `nvidia.com/gpu` resource, and the NVIDIA device plugin exposing node GPUs to the scheduler. A pod requesting one GPU looks like this (trimmed from the EKS accelerated-workload guide):

<!-- ex: id=aws-2 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: litetorch-gpu
spec:
  restartPolicy: OnFailure
  containers:
    - name: litetorch
      image: <your-ecr-registry>/litetorch:cuda
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
```

Replace `<your-ecr-registry>` with your own ECR registry URI; the image is the LiteTorch CUDA image from [Installation on Docker](installation/docker.md). The `nvidia.com/gpu: 1` request is what schedules the pod onto a GPU node; without it the pod lands on CPU capacity and LiteTorch falls back to CPU silently (§6).

> Docker runtime unverified: there is no Docker binary on the docs verification VM, so the LiteTorch CUDA image has not been built or run here, and no ECS or EKS run has been executed. The Dockerfile contents are the reviewed deliverable; the build and the ECS/EKS runs are Batch B5 verification items.

---

## 6. LiteTorch notes on every path

These apply identically on EC2, SageMaker BYOC, ECS, and EKS. They are the checks that separate "the GPU is in use" from "the run silently used CPU".

1. **Check the CUDA toolkit before any GPU run.** Run `nvcc --version` first, on the instance or inside the container. A missing `nvcc` means the native GPU library never builds, and the framework falls back to CPU silently.
2. **Run the pre-flight gate.** Confirm the `Native GPU (CUDA/HIP)` init line appears at import, and confirm the built extension and `liblitetorch_gpu.so` sit next to it under `<tree>/python/`. The gate is: init line present AND `.so` placement correct. Never trust a `BUILD OK` line alone; `setup.py` prints it even when per-file compiles fail.
3. **The device string is `gpu:0`, not `cuda:0`.** LiteTorch device strings use the `gpu:` prefix: `str(t.device)` is `gpu:0` for the first GPU. Code that writes `"cuda:0"` as a LiteTorch device string is wrong; the parser accepts `cuda` as input, but the canonical string a device prints is `gpu:N`.
4. **Silent CPU fallback is the default failure.** If the GPU backend did not load, `.to(device)` falls back to CPU silently. The symptom is a run that completes with CPU speed and CPU results; the diagnosis is the missing init line in step 2, not the training output.

<!-- ex: id=aws-3 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```python
import sys
sys.path.insert(0, "/opt/litetorch/python")
import litetorch as lt

print(lt.is_gpu_available())
print(lt.get_backend_name())
print(lt.auto_device())
```

Expected on a working GPU path: `True`, a CUDA backend name, and `gpu:0`. If the third line prints `cpu:0` on a machine where a GPU was expected, the GPU backend did not load — check `nvcc --version`, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line. The `sys.path` line is required only for a source-built tree; a pip install does not need it. Adjust the path to wherever the tree lives on your instance or in your image.

5. **Stop what you are not using.** GPU instances bill while running. Stop (not just disconnect from) an EC2 instance when work pauses, and delete endpoints and idle ECS/EKS GPU capacity when they are no longer needed:

<!-- ex: id=aws-4 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->

```bash
aws ec2 stop-instances --instance-ids i-0123456789abcdef0
```

Replace the instance ID with your own. Stopping preserves the EBS volume; terminating deletes the instance. Either way, confirm in the console that the instance is actually stopped — a running GPU instance left overnight is the most common AWS cost surprise, and this page states no prices to warn you with.

---

## 7. Troubleshooting

- GPU run silently uses CPU — see §6, steps 1–4: `nvcc`, the init line, the `.so` placement, and the `gpu:0` device string.
- `ModuleNotFoundError: No module named 'litetorch'` — `<tree>/python` is not on `sys.path` (source build) or `PYTHONPATH` (container). See [Installation](installation.md) §2.4 and [Installation on Docker](installation/docker.md).
- `nvidia-smi` fails on EC2 — driver problem, not a LiteTorch problem. On a DLAMI, relaunch from the DLAMI; on plain Ubuntu, reinstall the driver following the AWS install guide.
- Build prints `BUILD OK` but the import fails — per-file compiles failed silently; the real gate is the extension file existing under `<tree>/python/` and `import litetorch` succeeding.
- Anything else: [Troubleshooting](troubleshooting.md).

---

## Sources

Official AWS documentation consulted for this page (accessed 2026-10-27). Every instance-family, GPU-model, AMI, and service claim above traces to one of these pages; no price, quota, region-availability, or SLA claim is made anywhere on this page.

- EC2 G6 instances (L4, up to 8 GPUs, 24 GB per GPU, fractional sizes): <http://aws.amazon.com/ec2/instance-types/g6/>
- EC2 P5 instances (H100/H200, up to 8 GPUs, 640 GB HBM3 / 1128 GB HBM3e): <http://aws.amazon.com/ec2/instance-types/p5/>
- DLAMI recommended GPU instances (P6-B200/B300, P6e-GB200, P5e, P5, P4, P3, G3, G4, G5, G6, G6e): <https://docs.aws.amazon.com/dlami/latest/devguide/gpu.html>
- DLAMI launching an instance (console/CLI steps): <https://docs.aws.amazon.com/dlami/latest/devguide/launch.html>
- DLAMI GPU AMI contents (NVIDIA driver, CUDA Toolkit, NCCL, cuDNN, Docker, NVIDIA Container Toolkit): AWS re:Post, "Deep Learning graphical desktop on Ubuntu Linux with AWS Deep Learning AMI (DLAMI)": <https://repost.aws/articles/AR6RrDeUL1Tq6R8TgDs59iEA/deep-learning-graphical-desktop-on-ubuntu-linux-with-aws-deep-learning-ami-dlami>
- Installing the NVIDIA driver, CUDA Toolkit, and NVIDIA Container Toolkit on Ubuntu EC2 instances: AWS re:Post: <https://repost.aws/articles/ARWGxLArMBQ4y1MKoSHTq3gQ/install-nvidia-gpu-driver-cuda-toolkit-nvidia-container-toolkit-on-amazon-ec2-instances-running-ubuntu-linux>
- SageMaker AI overview (bring any model, framework, container): <https://aws.amazon.com/sagemaker/ai/>
- SageMaker script mode vs bring-your-own-container: <https://aws.amazon.com/blogs/machine-learning/use-your-own-training-scripts-and-automatically-select-the-best-model-using-hyperparameter-optimization-in-amazon-sagemaker/> and <https://aws.amazon.com/ko/blogs/machine-learning/introducing-amazon-sagemaker-components-for-kubeflow-pipelines/>
- SageMaker bring-your-own inference container (port 8080, `/invocations`, `/ping`): <https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html>
- ECS GPU instances (g4dn T4, g5 A10G, p3 V100, p4d A100, g6f fractional L4; drivers and CUDA pre-installed): <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/managed-instances-gpu.html>
- EKS accelerated workloads (GPU NodePool, `nvidia.com/gpu`): <https://docs.aws.amazon.com/eks/latest/userguide/auto-accelerated.html>
- EKS NVIDIA device plugin: <https://docs.aws.amazon.com/eks/latest/userguide/device-management-nvidia-dra-device-plugin.html>
- EC2 On-Demand pricing (linked, no prices quoted): <https://aws.amazon.com/ec2/pricing/on-demand/>
