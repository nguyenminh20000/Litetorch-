import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as tnn

os.environ.setdefault("LITETORCH_CUDNN_ALGO_LOG", "1")

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

DIAG_DEVICE = os.environ.get("DIAG_DEVICE", "gpu").lower()
DIAG_MAX_STEPS = int(os.environ.get("DIAG_MAX_STEPS", "0"))
DIAG_FORCE_PER_PARAM = os.environ.get("LITETORCH_ADAM_FORCE_PER_PARAM", "")


def diag_device():
    if DIAG_DEVICE == "cpu":
        return lt.Device("cpu")
    return lt.Device("gpu:0")


def sync():
    torch.cuda.synchronize()
    if DIAG_DEVICE != "cpu":
        lt.cuda_synchronize()


def tensor_stats(vec):
    a = np.asarray(vec, dtype=np.float64)
    n = a.size
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    nan = int(np.isnan(a).sum())
    finite = a[np.isfinite(a)]
    norm = float(np.linalg.norm(a)) if nan == 0 else float("nan")
    if finite.size == 0:
        return norm, float("nan"), float("nan"), nan
    return norm, float(finite.min()), float(finite.max()), nan


def snapshot_params(params, cpu):
    out = []
    for p in params:
        out.append(np.array(p.to(cpu).to_vector(), dtype=np.float32))
    return out


def scalar_value(t):
    return float(t.to(lt.Device("cpu")).to_vector()[0])


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


def build_lt(dev):
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


def gradcheck(torch_model, lt_model, lt_opt, xb0, yb0, first_lt_batch, dev):
    cpu = lt.Device("cpu")
    ce = tnn.CrossEntropyLoss()
    torch_model.zero_grad()
    xt = torch.from_numpy(np.ascontiguousarray(xb0)).cuda()
    yt = torch.from_numpy(yb0).cuda()
    loss_t = ce(torch_model(xt), yt)
    loss_t.backward()
    t_grads = []
    for tp in torch_model.parameters():
        g = tp.grad.detach().cpu().numpy().astype(np.float64) if tp.grad is not None else None
        t_grads.append(g)
    torch_model.zero_grad()
    xt_lt, yt_lt = first_lt_batch
    lt_opt.zero_grad()
    loss_l = lt.Ops.cross_entropy_loss(lt_model.forward(xt_lt), yt_lt)
    loss_l.backward()
    sync()
    for i, (name, tg) in enumerate(zip(PARAM_NAMES, t_grads)):
        lp = lt_model.parameters()[i]
        lg = lp.grad
        if tg is None:
            print(f"DIAG-GRADCHECK {name}: torch_grad=None", flush=True)
            continue
        if lg is None:
            print(f"DIAG-GRADCHECK {name}: torch_norm={np.linalg.norm(tg):.6g} lt_grad=None", flush=True)
            continue
        la = np.array(lg.to(cpu).to_vector(), dtype=np.float64)
        diff = np.abs(la.reshape(tg.shape) - tg)
        print(f"DIAG-GRADCHECK {name}: torch_norm={np.linalg.norm(tg):.6g} lt_norm={np.linalg.norm(la):.6g} max_abs_diff={diff.max():.6g} ratio={np.linalg.norm(tg)/max(np.linalg.norm(la), 1e-30):.4g}", flush=True)
    lt_opt.zero_grad()


def diag_counters(tag):
    try:
        print(f"DIAG-COUNTERS {tag}: gpu_used={lt.get_gpu_memory_used()} gpu_limit={lt.get_gpu_limit()} last_evicted_bytes={lt.get_last_evicted_bytes()} last_eviction_required={lt.get_last_eviction_required()} gpu_limit_calls={lt.get_gpu_limit_calls()} last_set_gpu_limit={lt.get_last_set_gpu_limit()} cached_gpu_bytes={lt.get_cached_gpu_bytes()}", flush=True)
    except AttributeError as e:
        print(f"DIAG-COUNTERS {tag}: binding-unavailable ({e})", flush=True)


def diag_mirrors(tag, tensors):
    try:
        has_fn = lt._diag_has_cpu_mirror
    except AttributeError:
        print(f"DIAG-MIRROR {tag}: binding-unavailable", flush=True)
        return
    flagged = []
    checked = 0
    for name, t in tensors:
        if t is None:
            continue
        checked += 1
        try:
            flag = bool(has_fn(t))
        except Exception as e:
            print(f"DIAG-MIRROR {tag} {name}: error={e}", flush=True)
            continue
        if flag:
            flagged.append(name)
            print(f"DIAG-MIRROR {tag} {name}: cpu_mirror=1", flush=True)
    print(f"DIAG-MIRROR {tag}: flagged={len(flagged)}/{checked} {flagged}", flush=True)


def stagedump_gradcheck(torch_model, lt_model, lt_opt, xb0, yb0, first_lt_batch, dev):
    cpu = lt.Device("cpu")
    ce = tnn.CrossEntropyLoss()
    torch_model.zero_grad()
    xt = torch.from_numpy(np.ascontiguousarray(xb0)).cuda()
    yt = torch.from_numpy(yb0).cuda()
    t1 = torch_model[0](xt)
    t2 = torch_model[1](t1)
    t3 = torch_model[2](t2)
    t4 = torch_model[3](t3)
    t5 = torch_model[4](t4)
    t6 = torch_model[5](t5)
    t7 = torch_model[6](t6)
    t8 = torch_model[7](t7)
    t9 = torch_model[8](t8)
    t10 = torch_model[9](t9)
    t11 = torch_model[10](t10)
    t12 = torch_model[11](t11)
    t13 = torch_model[12](t12)
    loss_t = ce(t13, yt)
    t_acts = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13]
    t_gouts = torch.autograd.grad(loss_t, t_acts, retain_graph=True)
    t_params = list(torch_model.parameters())
    t_pgrads = torch.autograd.grad(loss_t, t_params, retain_graph=True)
    torch_model.zero_grad()
    x_np = t10.detach().cpu().numpy().astype(np.float64)
    rank = int(np.linalg.matrix_rank(x_np))
    svals = np.linalg.svd(x_np, compute_uv=False)
    print(f"DIAG-STAGE rank-X: shape={list(x_np.shape)} rank={rank} s_max={svals[0]:.6g} s_min={svals[-1]:.6g}", flush=True)
    xt_lt, yt_lt = first_lt_batch
    lt_opt.zero_grad()
    p = lt_model.parameters()
    c1w, c1b, c2w, c2b, c3w, c3b, f1w, f1b, f2w, f2b = p
    b0 = int(xb0.shape[0])
    a1 = lt.Ops.conv2d(xt_lt, c1w, c1b, stride=1, padding=1)
    a2 = lt.Ops.relu(a1)
    a3 = lt.Ops.max_pool2d(a2, 2)
    a4 = lt.Ops.conv2d(a3, c2w, c2b, stride=1, padding=1)
    a5 = lt.Ops.relu(a4)
    a6 = lt.Ops.max_pool2d(a5, 2)
    a7 = lt.Ops.conv2d(a6, c3w, c3b, stride=1, padding=1)
    a8 = lt.Ops.relu(a7)
    a9 = lt.Ops.adaptive_avg_pool2d(a8, 4, 4)
    a10 = a9.view([b0, 2048])
    w1t = f1w.transpose(0, 1)
    a11 = lt.Ops.matmul_bias(a10, w1t, f1b)
    a12 = lt.Ops.relu(a11)
    w2t = f2w.transpose(0, 1)
    a13 = lt.Ops.matmul_bias(a12, w2t, f2b)
    a_acts = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13]
    stage_names = ["fwd-conv1", "fwd-relu1", "fwd-pool1", "fwd-conv2", "fwd-relu2", "fwd-pool2",
                   "fwd-conv3", "fwd-relu3", "fwd-adaptivepool", "fwd-flatten", "fwd-fc1",
                   "fwd-relu4", "fwd-fc2"]
    for name, la_t, tt in zip(stage_names, a_acts, t_acts):
        la = np.array(la_t.to(cpu).to_vector(), dtype=np.float64)
        ta = tt.detach().cpu().numpy().astype(np.float64).reshape(-1)
        diff = np.abs(la.reshape(ta.shape) - ta)
        print(f"DIAG-STAGE {name}: lt_norm={np.linalg.norm(la):.6g} torch_norm={np.linalg.norm(ta):.6g} max_abs_diff={diff.max():.6g}", flush=True)
    diag_mirrors("pre-backward", [(f"act-{n}", t) for n, t in zip(stage_names, a_acts)] + [(f"param-{n}", t) for n, t in zip(PARAM_NAMES, p)])
    diag_counters("pre-backward")
    loss_l = lt.Ops.cross_entropy_loss(a13, yt_lt)
    loss_l.backward()
    sync()
    gout_names = ["gout-conv1", "gout-relu1", "gout-pool1", "gout-conv2", "gout-relu2", "gout-pool2",
                  "gout-conv3", "gout-relu3", "gout-adaptivepool", "gout-flatten", "gout-fc1",
                  "gout-relu4", "gout-ce"]
    for name, la_t, tg in zip(gout_names, a_acts, t_gouts):
        lg = la_t.grad
        if lg is None:
            print(f"DIAG-STAGE {name}: lt_grad=None", flush=True)
            continue
        la = np.array(lg.to(cpu).to_vector(), dtype=np.float64)
        ta = tg.detach().cpu().numpy().astype(np.float64).reshape(-1)
        diff = np.abs(la.reshape(ta.shape) - ta)
        print(f"DIAG-STAGE {name}: lt_norm={np.linalg.norm(la):.6g} torch_norm={np.linalg.norm(ta):.6g} max_abs_diff={diff.max():.6g}", flush=True)
    for i, name in enumerate(PARAM_NAMES):
        lg = p[i].grad
        tg = t_pgrads[i].detach().cpu().numpy().astype(np.float64)
        if lg is None:
            print(f"DIAG-STAGE leaf-{name}: lt_grad=None", flush=True)
            continue
        la = np.array(lg.to(cpu).to_vector(), dtype=np.float64)
        diff = np.abs(la.reshape(tg.shape) - tg)
        print(f"DIAG-STAGE leaf-{name}: lt_norm={np.linalg.norm(la):.6g} torch_norm={np.linalg.norm(tg):.6g} max_abs_diff={diff.max():.6g}", flush=True)
    diag_mirrors("post-backward", [(f"act-{n}", t) for n, t in zip(stage_names, a_acts)] + [(f"param-{n}", t) for n, t in zip(PARAM_NAMES, p)] + [(f"grad-{n}", p[i].grad) for i, n in enumerate(PARAM_NAMES)])
    diag_counters("post-backward")
    print("DIAG-ALGO note: search this log for '[litetorch] cudnn bwd-data/bwd-filter algo chosen=' stderr lines (item 6, enabled via LITETORCH_CUDNN_ALGO_LOG=1)", flush=True)
    lt_opt.zero_grad()


def pred_histogram(model, x_test, y_test, dev):
    counts = [0] * NUM_CLASSES
    for xb, yb in batches(x_test, y_test, 512, shuffle=False):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 3, 32, 32], dev)
        out = model.forward(xt)
        pred = np.array(out.to(lt.Device("cpu")).to_vector()).reshape(b, 10).argmax(1)
        for c in pred:
            counts[int(c)] += 1
    sync()
    print(f"DIAG-HISTOGRAM {counts}", flush=True)
    return counts


def train_lt(x_train, y_train, x_test, y_test, torch_model=None):
    dev = diag_device()
    cpu = lt.Device("cpu")
    model = build_lt(dev)
    if torch_model is not None:
        copy_init_from_torch(torch_model, model, dev)
    opt = lt.optim.Adam(model.parameters(), lr=LR)
    params = model.parameters()
    n = len(x_train)
    print("[lt] preloading batches...", flush=True)
    t_pre = time.perf_counter()
    lt_batches = []
    xb0 = yb0 = None
    for k, (xb, yb) in enumerate(batches(x_train, y_train, BATCH)):
        if k == 0:
            xb0, yb0 = xb, yb
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 3, 32, 32], dev)
        yt = lt.Tensor.from_vector([float(v) for v in yb], [b], dev)
        lt_batches.append((xt, yt))
    sync()
    print(f"[lt] preload done in {time.perf_counter()-t_pre:.2f}s", flush=True)
    if torch_model is not None:
        acc0 = eval_lt(model, x_test, y_test, dev)
        print(f"DIAG-COPY-EVAL acc={acc0:.4f}", flush=True)
        gradcheck(torch_model, model, opt, xb0, yb0, lt_batches[0], dev)
        stagedump_gradcheck(torch_model, model, opt, xb0, yb0, lt_batches[0], dev)
    times = []
    step_idx = 0
    stop = False
    for ep in range(EPOCHS):
        if stop:
            break
        t0 = time.perf_counter()
        loss_sum = 0.0
        loss_cnt = 0
        for xt, yt in lt_batches:
            if DIAG_MAX_STEPS and step_idx >= DIAG_MAX_STEPS:
                stop = True
                break
            detailed = step_idx < 3
            opt.zero_grad()
            loss = lt.Ops.cross_entropy_loss(model.forward(xt), yt)
            loss.backward()
            lv = scalar_value(loss)
            loss_sum += lv
            loss_cnt += 1
            if detailed:
                for i, name in enumerate(PARAM_NAMES):
                    g = params[i].grad
                    if g is None:
                        print(f"DIAG-STEP step={step_idx} {name}: grad=None", flush=True)
                    else:
                        ga = np.array(g.to(cpu).to_vector(), dtype=np.float64)
                        print(f"DIAG-STEP step={step_idx} {name}: grad_norm={np.linalg.norm(ga):.6g} grad_nonzero={int((ga != 0).sum())}/{ga.size}", flush=True)
                before = snapshot_params(params, cpu)
                for i, name in enumerate(PARAM_NAMES):
                    norm, mn, mx, nan = tensor_stats(before[i])
                    print(f"DIAG-P-BEFORE step={step_idx} {name}: norm={norm:.6g} min={mn:.6g} max={mx:.6g} nan={nan}", flush=True)
                print(f"DIAG-STEP step={step_idx} lr={opt.get_lr():.6g} step_count={step_idx}", flush=True)
            else:
                before = None
            opt.step()
            if detailed:
                sync()
                after = snapshot_params(params, cpu)
                for i, name in enumerate(PARAM_NAMES):
                    delta = after[i].astype(np.float64) - before[i].astype(np.float64)
                    g = params[i].grad
                    if g is not None:
                        ga = np.array(g.to(cpu).to_vector(), dtype=np.float64)
                        expected = LR * np.sqrt(float((ga != 0).sum()))
                    else:
                        expected = 0.0
                    print(f"DIAG-DELTA step={step_idx} {name}: delta_norm={np.linalg.norm(delta):.6g} delta_nan={int(np.isnan(delta).sum())} expected_adam={expected:.6g}", flush=True)
            step_idx += 1
        sync()
        dt = time.perf_counter() - t0
        times.append(dt)
        mean_loss = loss_sum / max(loss_cnt, 1)
        print(f"[lt] epoch {ep+1}/{EPOCHS}: {dt:.2f}s ({n/dt:.0f} samples/s) loss_mean={mean_loss:.6g}", flush=True)
    acc = eval_lt(model, x_test, y_test, dev)
    print(f"[lt] test acc: {acc:.4f}", flush=True)
    pred_histogram(model, x_test, y_test, dev)
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
    print(f"DIAG-SCRIPT diag-nolearn-stagedump device={DIAG_DEVICE} max_steps={DIAG_MAX_STEPS} force_per_param={DIAG_FORCE_PER_PARAM!r}", flush=True)
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
