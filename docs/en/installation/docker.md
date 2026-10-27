# Installation on Docker

Docker gives a reproducible LiteTorch install: the compiler, Python, and (for the CUDA image) the CUDA toolkit are pinned in the image instead of installed on the host. Two images are provided: a CPU image and a CUDA image.

> Status note: the Docker commands below were written from the LiteTorch source-build procedure and the Docker CLI contract. Docker is not installed on the docs verification VM, so the image builds on this page were not executed here; the CPU and CUDA `Dockerfile` contents are the reviewed deliverable, and the build itself is a Batch B5 verification item. Do not treat the commands as run-verified until that item closes.

---

## CPU image

`Dockerfile.cpu`:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-pip libopenblas-dev git cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pybind11 numpy

WORKDIR /opt
RUN git clone https://github.com/nguyenminh20000/Litetorch-.git litetorch
WORKDIR /opt/litetorch

RUN LITETORCH_NO_NATIVE_GPU=1 python3 setup.py build_ext --inplace

ENV PYTHONPATH=/opt/litetorch/python:$PYTHONPATH
CMD ["python3", "-c", "import litetorch as lt; print(lt.get_backend_name(), lt.auto_device())"]
```

Build and run:

```bash
docker build -f Dockerfile.cpu -t litetorch:cpu .
docker run --rm litetorch:cpu
```

Expected output of the run:

```
CPU cpu:0
```

Notes:

- The build sets `LITETORCH_NO_NATIVE_GPU=1`, so no CUDA toolkit is needed and the native GPU library is skipped.
- `PYTHONPATH` points at `/opt/litetorch/python` because `setup.py` uses `package_dir={"": "python"}` — the built extension lives under `<tree>/python/`, not the tree root. Without that entry, `import litetorch` fails with `ModuleNotFoundError`.
- On a small host, add `--build-arg` free memory pressure by limiting build jobs: pass `LT_BUILD_JOBS=1` in the `RUN` line (it also forces `-O1` on the binding translation units).

---

## CUDA image

`Dockerfile.cuda`:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential python3-dev python3-pip libopenblas-dev git cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pybind11 numpy

WORKDIR /opt
RUN git clone https://github.com/nguyenminh20000/Litetorch-.git litetorch
WORKDIR /opt/litetorch

RUN python3 setup.py build_ext --inplace

ENV PYTHONPATH=/opt/litetorch/python:$PYTHONPATH
CMD ["python3", "-c", "import litetorch as lt; print(lt.get_backend_name(), lt.auto_device(), lt.is_gpu_available())"]
```

Build and run (requires the NVIDIA Container Toolkit on the host and a GPU):

```bash
docker build -f Dockerfile.cuda -t litetorch:cuda .
docker run --rm --gpus all litetorch:cuda
```

Check the CUDA toolkit inside the image before trusting a GPU run — the standing rule applies inside containers too:

```bash
docker run --rm litetorch:cuda nvcc --version
```

If `nvcc` is missing, the native GPU library never builds and the framework falls back to CPU silently. Expected output of the run on a GPU host:

```
CUDA cuda:0 True
```

(Exact backend name and device index depend on the host GPU. If the output shows `CPU cpu:0 False`, the GPU backend did not load — check `nvcc --version` inside the image, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line appears.)

---

## Using your own code with the image

Mount a working directory and run a script inside the container:

```bash
docker run --rm --gpus all -v "$PWD":/work -w /work litetorch:cuda python3 train.py
```

For CPU-only work, drop `--gpus all` and use `litetorch:cpu`. Your script still needs `<tree>/python` on `sys.path` only when it imports a source-built tree; inside these images `PYTHONPATH` already covers it.

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'litetorch'` — `PYTHONPATH` does not include `<tree>/python`. Check `echo $PYTHONPATH` inside the container and confirm the `.so` exists under `/opt/litetorch/python/`.
- The build prints `BUILD OK` but the import fails — `setup.py` prints `BUILD OK` even when per-file compiles fail. The real gate is the extension file existing and `import litetorch` succeeding.
- GPU run silently uses CPU — see the CUDA image section above: verify `nvcc`, rebuild, and confirm the `Native GPU (CUDA/HIP)` init line.
- Out of memory during the build — set `LT_BUILD_JOBS=1` in the image build to lower peak memory.
