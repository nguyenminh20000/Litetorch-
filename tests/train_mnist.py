import gzip
import os
import struct
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as tnn

import litetorch as lt

DATA_DIR = "/content/data/mnist"
WEIGHT_DIR = "/content/weights"
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = {
    "train-images-idx3-ubyte.gz": (60000, 28, 28),
    "train-labels-idx1-ubyte.gz": (60000,),
    "t10k-images-idx3-ubyte.gz": (10000, 28, 28),
    "t10k-labels-idx1-ubyte.gz": (10000,),
}

BATCH = 128
EPOCHS = 2
LR = 1e-3


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"downloading {name}...", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{name}", path)


def load(name):
    path = os.path.join(DATA_DIR, name)
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2051:
            n, r, c = struct.unpack(">III", f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
        else:
            n = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def get_data():
    download()
    x_train = load("train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y_train = load("train-labels-idx1-ubyte.gz").astype(np.int64)
    x_test = load("t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y_test = load("t10k-labels-idx1-ubyte.gz").astype(np.int64)
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


def train_torch(data):
    (x_train, y_train), (x_test, y_test) = data
    torch.manual_seed(0)
    model = build_torch()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = tnn.CrossEntropyLoss()
    n = len(x_train)
    times = []
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
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[torch] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s)", flush=True)
    acc = eval_torch(model, x_test, y_test)
    print(f"[torch] test acc: {acc:.4f}", flush=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "mnist_torch.pt"))
    return times, acc, model


@torch.no_grad()
def eval_torch(model, x_test, y_test):
    model.eval()
    correct = total = 0
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        xt = torch.from_numpy(xb).unsqueeze(1).cuda()
        pred = model(xt).argmax(1).cpu().numpy()
        correct += (pred == yb).sum()
        total += len(yb)
    model.train()
    return correct / total


def train_lt(data, torch_model=None):
    (x_train, y_train), (x_test, y_test) = data
    dev = lt.Device("gpu:0")
    model = build_lt()
    if torch_model is not None:
        copy_init_from_torch(torch_model, model, dev)
    opt = lt.optim.Adam(model.parameters(), lr=LR)
    n = len(x_train)
    print("[lt] preloading batches...", flush=True)
    t_pre = time.perf_counter()
    lt_batches = []
    for xb, yb in batches(x_train, y_train, BATCH):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 1, 28, 28], dev)
        yt = lt.Tensor.from_vector([float(v) for v in yb], [b], dev)
        lt_batches.append((xt, yt))
    sync()
    print(f"[lt] preload done in {time.perf_counter()-t_pre:.2f}s", flush=True)
    times = []
    forward = model.forward
    xent = lt.Ops.cross_entropy_loss
    zero_grad = opt.zero_grad
    step = opt.step
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        t_fwd = t_bwd = t_opt = 0.0
        for xt, yt in lt_batches:
            zero_grad()
            ta = time.perf_counter()
            out = forward(xt)
            loss = xent(out, yt)
            sync(); t_fwd += time.perf_counter() - ta
            ta = time.perf_counter()
            loss.backward()
            sync(); t_bwd += time.perf_counter() - ta
            ta = time.perf_counter()
            step()
            sync(); t_opt += time.perf_counter() - ta
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[lt] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s) fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s opt={t_opt:.2f}s", flush=True)
    acc = eval_lt(model, x_test, y_test, dev)
    print(f"[lt] test acc: {acc:.4f}", flush=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    lt.save_parameters(model.parameters(), os.path.join(WEIGHT_DIR, "mnist_lt.bin"))
    return times, acc


def eval_lt(model, x_test, y_test, dev):
    cpu = lt.Device("cpu")
    correct = total = 0
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 1, 28, 28], dev)
        out = model.forward(xt)
        pred = np.array(out.to(cpu).to_vector()).reshape(b, 10).argmax(1)
        correct += (pred == yb).sum()
        total += len(yb)
    sync()
    return correct / total


def main():
    data = get_data()
    print("== MNIST: pytorch ==", flush=True)
    t_times, t_acc, t_model = train_torch(data)
    print("== MNIST: litetorch ==", flush=True)
    l_times, l_acc = train_lt(data, t_model)
    tt, ll = sum(t_times) / len(t_times), sum(l_times) / len(l_times)
    print(f"MNIST RESULT: torch {tt:.2f}s/epoch acc={t_acc:.4f} | lt {ll:.2f}s/epoch acc={l_acc:.4f} | speedup x{tt/ll:.2f}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
