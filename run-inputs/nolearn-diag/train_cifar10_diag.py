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

PARAM_NAMES = [
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "conv3.weight", "conv3.bias",
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
]


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


def diag_snapshot(params, cpu):
    snaps = []
    for p in params:
        snaps.append(np.array(p.to(cpu).to_vector(), dtype=np.float64))
    return snaps


def diag_gradcheck(torch_model, lt_model, opt, x_train, y_train, dev):
    cpu = lt.Device("cpu")
    xb0, yb0 = next(batches(x_train, y_train, BATCH, shuffle=False))
    ce = tnn.CrossEntropyLoss()
    torch_model.zero_grad()
    xt = torch.from_numpy(np.ascontiguousarray(xb0)).cuda()
    yt = torch.from_numpy(yb0).cuda()
    tloss = ce(torch_model(xt), yt)
    tloss.backward()
    tgrads = []
    for p in torch_model.parameters():
        if p.grad is None:
            tgrads.append(None)
        else:
            tgrads.append(p.grad.detach().cpu().numpy().astype(np.float64).reshape(-1))
    opt.zero_grad()
    b = len(xb0)
    lxt = lt.Tensor.from_vector(xb0.reshape(-1).tolist(), [b, 3, 32, 32], dev)
    lyt = lt.Tensor.from_vector([float(v) for v in yb0], [b], dev)
    lloss = lt.Ops.cross_entropy_loss(lt_model.forward(lxt), lyt)
    lloss.backward()
    sync()
    lgrads = []
    for p in lt_model.parameters():
        g = p.grad
        if g is None:
            lgrads.append(None)
        else:
            lgrads.append(np.array(g.to(cpu).to_vector(), dtype=np.float64))
    print("[diag] gradcheck first batch (identical weights, dataset order):", flush=True)
    for i, name in enumerate(PARAM_NAMES):
        tg = tgrads[i]
        lg = lgrads[i]
        if tg is None or lg is None:
            print(f"[diag] gradcheck {name}: torch_null={tg is None} lt_null={lg is None}", flush=True)
        else:
            diff = float(np.max(np.abs(tg - lg)))
            print(f"[diag] gradcheck {name}: max_abs_diff={diff:.6e} torch_norm={float(np.linalg.norm(tg)):.6e} lt_norm={float(np.linalg.norm(lg)):.6e}", flush=True)
    opt.zero_grad()


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
    cpu = lt.Device("cpu")
    model = build_lt()
    if torch_model is not None:
        copy_init_from_torch(torch_model, model, dev)
    params = model.parameters()
    opt = lt.optim.Adam(params, lr=LR)
    copy_acc = eval_lt(model, x_test, y_test, dev)
    print(f"[diag] copy-eval lt acc (untrained, copied weights): {copy_acc:.4f}", flush=True)
    diag_gradcheck(torch_model, model, opt, x_train, y_train, dev)
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
    step_idx = 0
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        ep_before = diag_snapshot(params, cpu)
        loss_sum = 0.0
        loss_cnt = 0
        for xt, yt in lt_batches:
            opt.zero_grad()
            loss = lt.Ops.cross_entropy_loss(model.forward(xt), yt)
            loss_sum += float(loss.to(cpu).to_vector()[0])
            loss_cnt += 1
            if step_idx < 3:
                loss.backward()
                for i, p in enumerate(params):
                    g = p.grad
                    if g is None:
                        print(f"[diag] step {step_idx+1} {PARAM_NAMES[i]}: grad_null=True", flush=True)
                    else:
                        gv = np.array(g.to(cpu).to_vector(), dtype=np.float64)
                        print(f"[diag] step {step_idx+1} {PARAM_NAMES[i]}: grad_null=False grad_norm={float(np.linalg.norm(gv)):.6e}", flush=True)
                before = diag_snapshot(params, cpu)
                opt.step()
                sync()
                after = diag_snapshot(params, cpu)
                for i, name in enumerate(PARAM_NAMES):
                    dn = float(np.linalg.norm(after[i] - before[i]))
                    print(f"[diag] step {step_idx+1} {name}: weight_delta_norm={dn:.6e}", flush=True)
                print(f"[diag] step {step_idx+1} lr={opt.lr} step_count={step_idx+1}", flush=True)
            else:
                loss.backward()
                opt.step()
            step_idx += 1
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        ep_after = diag_snapshot(params, cpu)
        for i, name in enumerate(PARAM_NAMES):
            dn = float(np.linalg.norm(ep_after[i] - ep_before[i]))
            print(f"[diag] epoch {ep+1} {name}: weight_delta_norm={dn:.6e}", flush=True)
        print(f"[diag] epoch {ep+1} lr={opt.lr} step_count={step_idx}", flush=True)
        print(f"[diag] lt epoch {ep+1}/{EPOCHS} loss mean: {loss_sum/max(loss_cnt, 1):.4f}", flush=True)
        print(f"[lt] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s)", flush=True)
    acc, hist = diag_eval_lt(model, x_test, y_test, dev)
    print(f"[lt] test acc: {acc:.4f}", flush=True)
    print(f"[diag] lt final pred histogram: {hist.tolist()}", flush=True)
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


def diag_eval_lt(model, x_test, y_test, dev):
    correct = total = 0
    preds = []
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 3, 32, 32], dev)
        out = model.forward(xt)
        pred = np.array(out.to(lt.Device("cpu")).to_vector()).reshape(b, 10).argmax(1)
        preds.append(pred)
        correct += (pred == yb).sum()
        total += len(yb)
    sync()
    hist = np.bincount(np.concatenate(preds), minlength=10)
    return correct / total, hist


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
