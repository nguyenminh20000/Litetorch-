import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as tnn

import litetorch as lt

DATA_DIR = "/content/data/cifar10"
WEIGHT_DIR = "/content/weights"
URLS = [
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-batches-py.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]

BATCH = 128
EPOCHS = 2
LR = 1e-3
NUM_CLASSES = 10


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    out = os.path.join(DATA_DIR, "cifar-10-batches-py")
    if not os.path.exists(out):
        for url in URLS:
            try:
                print(f"downloading cifar-10 from {url}...", flush=True)
                urllib.request.urlretrieve(url, tgz)
                break
            except Exception as e:
                print(f"failed {url}: {e}", flush=True)
        else:
            raise RuntimeError("all CIFAR-10 mirrors failed")
        with tarfile.open(tgz) as tf:
            tf.extractall(DATA_DIR)


def load_batch(name):
    with open(os.path.join(DATA_DIR, "cifar-10-batches-py", name), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.array(d[b"labels"], dtype=np.int64)
    return x, y


def get_data():
    download()
    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_batch(f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    x_test, y_test = load_batch("test_batch")
    keep_tr = np.ones(len(y_train), dtype=bool)
    keep_te = np.ones(len(y_test), dtype=bool)
    x_train, y_train = x_train[keep_tr], y_train[keep_tr]
    x_test, y_test = x_test[keep_te], y_test[keep_te]
    print(f"cifar10-10class: train {len(x_train)}, test {len(x_test)}", flush=True)
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True)
    std = x_train.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return (x_train - mean) / std, y_train, (x_test - mean) / std, y_test


def batches(x, y, batch, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(x), batch):
        j = idx[i : i + batch]
        yield x[j], y[j]


def build_torch():
    return tnn.Sequential(
        tnn.Conv2d(3, 32, 3, padding=1), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Conv2d(32, 64, 3, padding=1), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Conv2d(64, 128, 3, padding=1), tnn.ReLU(),
        tnn.AdaptiveAvgPool2d(4),
        tnn.Flatten(),
        tnn.Linear(128 * 16, 64), tnn.ReLU(),
        tnn.Linear(64, 10),
    ).cuda()


def build_lt():
    dev = lt.Device("gpu:0")
    m = lt.nn.Sequential([
        lt.nn.Conv2d(3, 32, 3, padding=1), lt.nn.ReLU(), lt.nn.MaxPool2d(2),
        lt.nn.Conv2d(32, 64, 3, padding=1), lt.nn.ReLU(), lt.nn.MaxPool2d(2),
        lt.nn.Conv2d(64, 128, 3, padding=1), lt.nn.ReLU(),
        lt.nn.AdaptiveAvgPool2d(4, 4),
        lt.nn.Flatten(),
        lt.nn.Linear(128 * 16, 64), lt.nn.ReLU(),
        lt.nn.Linear(64, 10),
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


def train_torch(x_train, y_train, x_test, y_test):
    torch.manual_seed(0)
    model = build_torch()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = tnn.CrossEntropyLoss()
    n = len(x_train)
    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xb, yb in batches(x_train, y_train, BATCH):
            xt = torch.from_numpy(np.ascontiguousarray(xb)).cuda()
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
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "cats_dogs_torch.pt"))
    return times, acc, model


@torch.no_grad()
def eval_torch(model, x_test, y_test):
    model.eval()
    correct = total = 0
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        xt = torch.from_numpy(np.ascontiguousarray(xb)).cuda()
        pred = model(xt).argmax(1).cpu().numpy()
        correct += (pred == yb).sum()
        total += len(yb)
    model.train()
    return correct / total


def train_lt(x_train, y_train, x_test, y_test, torch_model=None):
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
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 3, 32, 32], dev)
        yt = lt.Tensor.from_vector([float(v) for v in yb], [b], dev)
        lt_batches.append((xt, yt))
    sync()
    print(f"[lt] preload done in {time.perf_counter()-t_pre:.2f}s", flush=True)
    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xt, yt in lt_batches:
            opt.zero_grad()
            loss = lt.Ops.cross_entropy_loss(model.forward(xt), yt)
            loss.backward()
            opt.step()
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[lt] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s)", flush=True)
    acc = eval_lt(model, x_test, y_test, dev)
    print(f"[lt] test acc: {acc:.4f}", flush=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    lt.save_parameters(model.parameters(), os.path.join(WEIGHT_DIR, "cats_dogs_lt.bin"))
    return times, acc


def eval_lt(model, x_test, y_test, dev):
    correct = total = 0
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 3, 32, 32], dev)
        out = model.forward(xt)
        pred = np.array(out.to(lt.Device("cpu")).to_vector()).reshape(b, 10).argmax(1)
        correct += (pred == yb).sum()
        total += len(yb)
    sync()
    return correct / total


def main():
    x_train, y_train, x_test, y_test = get_data()
    print("== cifar10: pytorch ==", flush=True)
    t_times, t_acc, t_model = train_torch(x_train, y_train, x_test, y_test)
    print("== cifar10: litetorch ==", flush=True)
    l_times, l_acc = train_lt(x_train, y_train, x_test, y_test, t_model)
    tt, ll = sum(t_times) / len(t_times), sum(l_times) / len(l_times)
    print(f"CIFAR10 RESULT: torch {tt:.2f}s/epoch acc={t_acc:.4f} | lt {ll:.2f}s/epoch acc={l_acc:.4f} | speedup x{tt/ll:.2f}", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
