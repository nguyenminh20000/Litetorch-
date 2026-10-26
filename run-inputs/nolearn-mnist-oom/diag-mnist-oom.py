import gc
import gzip
import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as tnn

import litetorch as lt

DIAG_SCRIPT = "diag-mnist-oom"
BATCH = 128
EPOCHS = 2
LR = 1e-3
SKIP_TORCH = os.environ.get("DIAG_MNIST_SKIP_TORCH", "0") == "1"
DATA_CANDIDATES = [
    "/kaggle/working/data/mnist",
    "/kaggle/input/mnist-dataset",
    "/content/data/mnist",
]
FILES = {
    "train-images-idx3-ubyte": (60000, 28, 28),
    "train-labels-idx1-ubyte": (60000,),
    "t10k-images-idx3-ubyte": (10000, 28, 28),
    "t10k-labels-idx1-ubyte": (10000,),
}


def nvidia_smi(tag):
    print(f"DIAG-NVIDIA-SMI {tag} begin", flush=True)
    for args in (
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"],
        ["nvidia-smi"],
    ):
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=30)
            print(f"DIAG-NVIDIA-SMI {tag} cmd={' '.join(args)} rc={out.returncode}", flush=True)
            print(out.stdout, flush=True)
            if out.stderr:
                print(out.stderr, flush=True)
            if out.returncode == 0:
                break
        except Exception as e:
            print(f"DIAG-NVIDIA-SMI {tag} cmd={' '.join(args)} failed: {e}", flush=True)
    print(f"DIAG-NVIDIA-SMI {tag} end", flush=True)


def mem_snapshot(tag):
    try:
        lt_used = lt.get_gpu_memory_used()
    except Exception as e:
        lt_used = f"<lt.get_gpu_memory_used failed: {e}>"
    try:
        t_alloc = torch.cuda.memory_allocated()
        t_res = torch.cuda.memory_reserved()
    except Exception as e:
        t_alloc = t_res = f"<torch mem failed: {e}>"
    print(f"DIAG-MNIST-PHASE {tag} lt_gpu_used={lt_used} torch_alloc={t_alloc} torch_reserved={t_res}", flush=True)


def find_data_dir():
    for root in DATA_CANDIDATES:
        if os.path.isdir(root):
            names = os.listdir(root)
            if any(n.startswith("train-images-idx3-ubyte") for n in names):
                return root
    import glob
    for path in sorted(glob.glob("/kaggle/input/*/train-images-idx3-ubyte*")):
        return os.path.dirname(path)
    return None


def load_idx(path):
    try:
        f = gzip.open(path, "rb")
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            n, r, c = struct.unpack(">III", f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
        else:
            n = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
        f.close()
        return data
    except Exception:
        with open(path, "rb") as f:
            magic = struct.unpack(">I", f.read(4))[0]
            if magic == 2051:
                n, r, c = struct.unpack(">III", f.read(12))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
            else:
                n = struct.unpack(">I", f.read(4))[0]
                data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def resolve_path(data_dir, name):
    for suffix in ("", ".gz"):
        path = os.path.join(data_dir, name + suffix)
        if os.path.exists(path):
            return path
    raise RuntimeError(f"MNIST file {name} not found under {data_dir}")


def get_data():
    data_dir = find_data_dir()
    if data_dir is None:
        raise RuntimeError("MNIST bundled data not found; stage hojjatk/mnist-dataset under /kaggle/working/data/mnist or /kaggle/input/mnist-dataset")
    print(f"DIAG-SCRIPT {DIAG_SCRIPT} data_dir={data_dir} batch={BATCH}", flush=True)
    x_train = load_idx(resolve_path(data_dir, "train-images-idx3-ubyte")).astype(np.float32) / 255.0
    y_train = load_idx(resolve_path(data_dir, "train-labels-idx1-ubyte")).astype(np.int64)
    x_test = load_idx(resolve_path(data_dir, "t10k-images-idx3-ubyte")).astype(np.float32) / 255.0
    y_test = load_idx(resolve_path(data_dir, "t10k-labels-idx1-ubyte")).astype(np.int64)
    return (x_train, y_train), (x_test, y_test)


def batches(x, y, batch, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(x), batch):
        j = idx[i : i + batch]
        yield x[j], y[j]


def build_torch():
    return tnn.Sequential(
        tnn.Conv2d(1, 32, 3, padding=1),
        tnn.ReLU(),
        tnn.MaxPool2d(2),
        tnn.Conv2d(32, 64, 3, padding=1),
        tnn.ReLU(),
        tnn.MaxPool2d(2),
        tnn.Flatten(),
        tnn.Linear(64 * 7 * 7, 128),
        tnn.ReLU(),
        tnn.Linear(128, 10),
    ).cuda()


def build_lt():
    dev = lt.Device("gpu:0")
    m = lt.nn.Sequential([
        lt.nn.Conv2d(1, 32, 3, padding=1),
        lt.nn.ReLU(),
        lt.nn.MaxPool2d(2),
        lt.nn.Conv2d(32, 64, 3, padding=1),
        lt.nn.ReLU(),
        lt.nn.MaxPool2d(2),
        lt.nn.Flatten(),
        lt.nn.Linear(64 * 7 * 7, 128),
        lt.nn.ReLU(),
        lt.nn.Linear(128, 10),
    ])
    m.to(dev)
    return m


def copy_init_from_torch(torch_model, lt_model, dev):
    cpu = lt.Device("cpu")
    for tp, lp in zip(torch_model.parameters(), lt_model.parameters()):
        arr = tp.detach().cpu().numpy().astype(np.float32)
        lp.copy_(lt.Tensor.from_vector(arr.reshape(-1).tolist(), list(lp.shape), cpu).to(dev))


def sync():
    torch.cuda.synchronize()
    lt.cuda_synchronize()


def train_torch(data):
    (x_train, y_train), (x_test, y_test) = data
    torch.manual_seed(0)
    model = build_torch()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = tnn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xb, yb in batches(x_train, y_train, BATCH):
            xt = torch.from_numpy(xb).unsqueeze(1).cuda()
            yt = torch.from_numpy(yb).cuda()
            opt.zero_grad()
            loss = ce(model(xt), yt)
            loss.backward()
            opt.step()
        sync()
        print(f"[torch] epoch {ep+1}/{EPOCHS}: {time.perf_counter()-t0:.2f}s", flush=True)
    mem_snapshot("after-torch-train")
    return model


def train_lt(data, torch_model=None):
    (x_train, y_train), (x_test, y_test) = data
    dev = lt.Device("gpu:0")
    print("[lt] build...", flush=True)
    model = build_lt()
    sync()
    mem_snapshot("after-lt-build")
    nvidia_smi("after-lt-build")
    if torch_model is not None:
        print("[lt] copy_init...", flush=True)
        copy_init_from_torch(torch_model, model, dev)
        sync()
        mem_snapshot("after-copy-init")
        nvidia_smi("after-copy-init")
    print("[lt] Adam...", flush=True)
    opt = lt.optim.Adam(model.parameters(), lr=LR)
    sync()
    mem_snapshot("after-adam")
    nvidia_smi("after-adam")
    print("[lt] preloading batches...", flush=True)
    t_pre = time.perf_counter()
    lt_batches = []
    for k, (xb, yb) in enumerate(batches(x_train, y_train, BATCH)):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 1, 28, 28], dev)
        yt = lt.Tensor.from_vector([float(v) for v in yb], [b], dev)
        lt_batches.append((xt, yt))
        if (k + 1) % 100 == 0:
            mem_snapshot(f"preload-batch-{k+1}")
    sync()
    print(f"[lt] preload done in {time.perf_counter()-t_pre:.2f}s batches={len(lt_batches)}", flush=True)
    mem_snapshot("after-preload")
    nvidia_smi("after-preload")
    forward = model.forward
    xent = lt.Ops.cross_entropy_loss
    zero_grad = opt.zero_grad
    step = opt.step
    print("[lt] first training batches...", flush=True)
    for i, (xt, yt) in enumerate(lt_batches[:10]):
        print(f"DIAG-MNIST-BATCH i={i}", flush=True)
        mem_snapshot(f"before-batch-{i}")
        zero_grad()
        out = forward(xt)
        loss = xent(out, yt)
        sync()
        mem_snapshot(f"after-fwd-{i}")
        loss.backward()
        sync()
        mem_snapshot(f"after-bwd-{i}")
        step()
        sync()
        mem_snapshot(f"after-step-{i}")
    print("[lt] first 10 batches done, no OOM", flush=True)


def main():
    print(f"DIAG-SCRIPT {DIAG_SCRIPT} device=gpu batch={BATCH} skip_torch={SKIP_TORCH}", flush=True)
    nvidia_smi("start")
    mem_snapshot("start")
    data = get_data()
    torch_model = None
    try:
        if not SKIP_TORCH:
            print("== MNIST: pytorch ==", flush=True)
            torch_model = train_torch(data)
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            sync()
            mem_snapshot("after-torch-empty-cache")
            nvidia_smi("after-torch-empty-cache")
        print("== MNIST: litetorch ==", flush=True)
        train_lt(data, torch_model)
    except RuntimeError as e:
        print(f"DIAG-MNIST-OOM-EXC {e}", flush=True)
        nvidia_smi("on-oom")
        mem_snapshot("on-oom")
        print("DIAG-MNIST-OOM REPRODUCED", flush=True)
        return 0
    print("DIAG-MNIST-OOM NOT REPRODUCED", flush=True)
    nvidia_smi("end")
    mem_snapshot("end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
