import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

import litetorch as lt
import torch
import torch.nn as tnn

DATA_DIR = os.environ.get("LT_DATA_DIR", "/content/data/mnist")
MIRRORS = [
    "https://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
BATCH = 256
EPOCHS = 3
LR = 1e-3


def load_idx(name):
    path = C.download([m + name for m in MIRRORS], os.path.join(DATA_DIR, name))
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def forward(lin1, relu, lin2, x):
    return lin2.forward(relu.forward(lin1.forward(x)))


def torch_parity_check(lin1, lin2, dev):
    tm = tnn.Sequential(tnn.Linear(784, 128), tnn.ReLU(), tnn.Linear(128, 10)).eval()
    C.copy_linear(lin1, tm[0])
    C.copy_linear(lin2, tm[2])
    xb = np.random.RandomState(0).rand(4, 784).astype(np.float32)
    xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [4, 784], dev)
    with torch.no_grad():
        tout = tm(torch.from_numpy(xb))
    C.torch_forward_parity(forward(lin1, lt.nn.ReLU(), lin2, xt), tout, 1e-4, "mnist-mlp")


def main():
    xtr = load_idx("train-images-idx3-ubyte").reshape(-1, 784)
    ytr = load_idx("train-labels-idx1-ubyte")
    xte = load_idx("t10k-images-idx3-ubyte").reshape(-1, 784)
    yte = load_idx("t10k-labels-idx1-ubyte")
    print(f"mnist: train {xtr.shape} test {xte.shape}", flush=True)
    dev = C.pick_device()
    lin1 = lt.nn.Linear(784, 128)
    relu = lt.nn.ReLU()
    lin2 = lt.nn.Linear(128, 10)
    for m in (lin1, lin2):
        m.to(dev)
    torch_parity_check(lin1, lin2, dev)
    opt = lt.optim.Adam(lin1.parameters() + lin2.parameters(), lr=LR)
    first_loss = None
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        tot, nb = 0.0, 0
        for xb, yb in C.batches(xtr, ytr, BATCH):
            xt, yt = C.lt_batch(xb, yb, dev)
            opt.zero_grad()
            loss = lt.Ops.cross_entropy_loss(forward(lin1, relu, lin2, xt), yt)
            loss.backward()
            opt.step()
            tot += C.scalar(loss)
            nb += 1
        C.sync()
        avg = tot / nb
        if first_loss is None:
            first_loss = avg
        print(f"[lt] mnist epoch {ep+1}/{EPOCHS}: loss={avg:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)
    acc = C.accuracy_litetorch(lambda x: forward(lin1, relu, lin2, x), xte, yte, 512, dev, 10)
    print(f"MNIST MLP RESULT: test accuracy = {acc*100:.2f}%", flush=True)
    C.check(avg < first_loss * 0.9, "loss did not decrease")
    C.check(acc > 0.90, f"test accuracy {acc} <= 0.90")
    print("MNIST MLP PASS", flush=True)


if __name__ == "__main__":
    main()
