import os
import sys
import urllib.request


def pick_device():
    import litetorch as lt
    if lt.is_gpu_available():
        return lt.Device("gpu:0")
    print("WARNING: no GPU available, falling back to CPU (slow)", flush=True)
    return lt.Device("cpu")


def sync():
    import litetorch as lt
    f = getattr(lt, "cuda_synchronize", None)
    if f is not None:
        f()


def to_list(t):
    import litetorch as lt
    sync()
    return t.to(lt.Device("cpu")).to_vector()


def scalar(t):
    return float(to_list(t)[0])


def download(urls, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    last = None
    for u in urls:
        try:
            print(f"downloading {u} ...", flush=True)
            urllib.request.urlretrieve(u, dest)
            return dest
        except Exception as e:
            last = e
            print(f"mirror failed: {u} ({e}), trying next", flush=True)
    raise RuntimeError(f"all mirrors failed for {dest}: {last}")


def batches(xs, ys, batch, shuffle=True):
    import numpy as np
    n = len(xs)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch):
        bi = idx[i:i + batch]
        yield xs[bi], ys[bi]


def lt_batch(xb, yb, dev):
    import litetorch as lt
    import numpy as np
    xf = xb.astype(np.float32).reshape(-1).tolist()
    yf = yb.astype(np.float32).reshape(-1).tolist()
    xt = lt.Tensor.from_vector(xf, list(xb.shape), dev)
    yt = lt.Tensor.from_vector(yf, list(yb.shape), dev)
    return xt, yt


def accuracy_litetorch(model_forward, xs, ys, batch, dev, n_classes):
    import litetorch as lt
    import numpy as np
    correct = 0
    total = 0
    for xb, yb in batches(xs, ys, batch, shuffle=False):
        xt, _ = lt_batch(xb, yb, dev)
        logits = model_forward(xt)
        pred = np.array(to_list(lt.Ops.argmax(logits, 1)), dtype=np.int64)
        correct += int((pred == yb.reshape(-1)).sum())
        total += len(yb)
    return correct / total


def check(cond, msg):
    if not cond:
        print(f"ASSERT FAILED: {msg}", flush=True)
        sys.exit(1)


def torch_forward_parity(lt_out, torch_out, tol, tag):
    import numpy as np
    a = np.array(to_list(lt_out), dtype=np.float64)
    b = torch_out.detach().cpu().numpy().reshape(-1).astype(np.float64)
    d = float(np.max(np.abs(a - b)))
    print(f"[{tag}] torch parity max abs diff = {d:.3e} (tol {tol})", flush=True)
    check(d <= tol, f"{tag}: torch parity diff {d} exceeds tol {tol}")


def copy_linear(lt_lin, torch_lin):
    import numpy as np
    w = np.array(to_list(lt_lin.weight), dtype=np.float32)
    b = np.array(to_list(lt_lin.bias), dtype=np.float32)
    torch_lin.weight.data.copy_(__import__("torch").from_numpy(w.reshape(torch_lin.weight.shape)))
    torch_lin.bias.data.copy_(__import__("torch").from_numpy(b.reshape(torch_lin.bias.shape)))


def copy_conv2d(lt_conv, torch_conv):
    import numpy as np
    w = np.array(to_list(lt_conv.weight), dtype=np.float32)
    torch_conv.weight.data.copy_(__import__("torch").from_numpy(w.reshape(torch_conv.weight.shape)))
    if lt_conv.bias is not None and torch_conv.bias is not None:
        b = np.array(to_list(lt_conv.bias), dtype=np.float32)
        torch_conv.bias.data.copy_(__import__("torch").from_numpy(b.reshape(torch_conv.bias.shape)))


def copy_embedding(lt_emb, torch_emb):
    import numpy as np
    w = np.array(to_list(lt_emb.weight), dtype=np.float32)
    torch_emb.weight.data.copy_(__import__("torch").from_numpy(w.reshape(torch_emb.weight.shape)))
