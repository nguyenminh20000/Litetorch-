import math

import numpy as np

import litetorch as lt
from litetorch.nn import functional as F


DEV = lt.Device("cpu")
RESULTS = []


def tv(data, shape, requires_grad=False):
    return lt.Tensor.from_vector(list(data), list(shape), DEV, requires_grad)


def run(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:
        RESULTS.append((name, "FAIL", type(e).__name__ + ": " + str(e)))


def close(a, b, tol=1e-5, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{msg}: shape {a.shape} vs {b.shape}"
    d = np.max(np.abs(a - b))
    assert d <= tol, f"{msg}: max diff {d} > {tol}"


def test_dropout_p0_identity():
    x = tv([1.0, 2.0, 3.0, 4.0], [2, 2], True)
    y = F.dropout(x, p=0.0, training=True)
    close(y.to_vector(), [1.0, 2.0, 3.0, 4.0], msg="dropout p=0")


def test_dropout_eval_passthrough():
    x = tv([1.0, 2.0, 3.0, 4.0], [2, 2], True)
    y = F.dropout(x, p=0.9, training=False)
    close(y.to_vector(), [1.0, 2.0, 3.0, 4.0], msg="dropout eval")


def test_dropout_p1_zeros():
    x = tv([1.0, 2.0, 3.0, 4.0], [2, 2])
    y = F.dropout(x, p=1.0, training=True)
    close(y.to_vector(), [0.0, 0.0, 0.0, 0.0], msg="dropout p=1")


def test_dropout_stats():
    lt.manual_seed(0)
    n = 20000
    xv = np.random.RandomState(0).uniform(0.5, 1.5, n)
    x = tv(xv, [n])
    y = F.dropout(x, p=0.5, training=True)
    yv = np.array(y.to_vector())
    frac_zero = np.mean(yv == 0.0)
    assert abs(frac_zero - 0.5) < 0.02, f"dropout frac_zero={frac_zero}"
    kept = yv[yv != 0.0]
    close(kept, xv[yv != 0.0] * 2.0, tol=1e-5, msg="dropout scale")


def test_dropout_backward():
    lt.manual_seed(1)
    xv = np.ones(64, dtype=np.float32)
    x = tv(xv, [64], True)
    y = F.dropout(x, p=0.5, training=True)
    yv = np.array(y.to_vector())
    y.sum().backward()
    g = np.array(x.grad.to_vector())
    kept = yv != 0.0
    close(g[kept], np.full(kept.sum(), 2.0), msg="dropout grad kept")
    close(g[~kept], np.zeros((~kept).sum()), msg="dropout grad dropped")


def test_linear_2d():
    rng = np.random.RandomState(2)
    xv = rng.randn(4, 8).astype(np.float32)
    wv = rng.randn(5, 8).astype(np.float32)
    bv = rng.randn(5).astype(np.float32)
    x = tv(xv.ravel(), [4, 8])
    w = tv(wv.ravel(), [5, 8])
    b = tv(bv.ravel(), [5])
    y = F.linear(x, w, b)
    close(y.to_vector(), (xv @ wv.T + bv).ravel(), tol=1e-4, msg="linear 2d")


def test_linear_no_bias_batched():
    rng = np.random.RandomState(3)
    xv = rng.randn(2, 3, 8).astype(np.float32)
    wv = rng.randn(5, 8).astype(np.float32)
    x = tv(xv.ravel(), [2, 3, 8])
    w = tv(wv.ravel(), [5, 8])
    y = F.linear(x, w)
    close(y.to_vector(), (xv @ wv.T).ravel(), tol=1e-4, msg="linear batched")
    assert list(y.shape) == [2, 3, 5]


def test_linear_backward():
    rng = np.random.RandomState(4)
    xv = rng.randn(4, 8).astype(np.float32)
    wv = rng.randn(5, 8).astype(np.float32)
    bv = rng.randn(5).astype(np.float32)
    x = tv(xv.ravel(), [4, 8], True)
    w = tv(wv.ravel(), [5, 8], True)
    b = tv(bv.ravel(), [5], True)
    y = F.linear(x, w, b)
    y.sum().backward()
    close(np.array(x.grad.to_vector()), (np.ones((4, 5), dtype=np.float32) @ wv).ravel(),
          tol=1e-4, msg="linear x.grad")
    close(np.array(b.grad.to_vector()), np.full(5, 4.0), tol=1e-4, msg="linear b.grad")
    close(np.array(w.grad.to_vector()), (np.ones((4, 5), dtype=np.float32).T @ xv).ravel(),
          tol=1e-4, msg="linear w.grad")


def test_pad_2d():
    xv = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    x = tv(xv.ravel(), [2, 3])
    y = F.pad(x, [1, 2, 3, 0])
    ref = np.pad(xv, ((3, 0), (1, 2)), mode="constant")
    close(y.to_vector(), ref.ravel(), msg="pad 2d")
    assert list(y.shape) == [5, 6]


def test_pad_value_3d():
    rng = np.random.RandomState(5)
    xv = rng.randn(2, 2, 2).astype(np.float32)
    x = tv(xv.ravel(), [2, 2, 2])
    y = F.pad(x, [1, 1, 0, 0, 2, 0], value=2.5)
    ref = np.pad(xv, ((2, 0), (0, 0), (1, 1)), mode="constant", constant_values=2.5)
    close(y.to_vector(), ref.ravel(), msg="pad 3d value")
    assert list(y.shape) == [4, 2, 4]


def test_pad_backward():
    xv = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    x = tv(xv.ravel(), [2, 3], True)
    y = F.pad(x, [1, 1, 1, 1])
    y.sum().backward()
    close(np.array(x.grad.to_vector()), np.ones(6), msg="pad backward")


def test_pad_bad_mode():
    x = tv([1.0, 2.0], [1, 2])
    try:
        F.pad(x, [1, 1], mode="reflect")
    except Exception:
        return
    raise AssertionError("pad reflect should throw")


def np_nearest(x, out_spatial):
    ndim = x.ndim
    ns = ndim - 2
    out_shape = (x.shape[0], x.shape[1]) + tuple(out_spatial)
    out = np.zeros(out_shape, dtype=np.float32)
    for idx in np.ndindex(*out_shape):
        src_idx = list(idx[:2])
        for d in range(ns):
            s = idx[2 + d] * x.shape[2 + d] / out_spatial[d]
            src_idx.append(min(int(s), x.shape[2 + d] - 1))
        out[idx] = x[tuple(src_idx)]
    return out


def test_interpolate_scale_4d():
    rng = np.random.RandomState(6)
    xv = rng.randn(1, 2, 3, 4).astype(np.float32)
    x = tv(xv.ravel(), [1, 2, 3, 4])
    y = F.interpolate(x, scale_factor=2.0)
    ref = np_nearest(xv, [6, 8])
    close(y.to_vector(), ref.ravel(), tol=1e-5, msg="interp scale 4d")
    assert list(y.shape) == [1, 2, 6, 8]


def test_interpolate_size_3d():
    rng = np.random.RandomState(7)
    xv = rng.randn(2, 3, 8).astype(np.float32)
    x = tv(xv.ravel(), [2, 3, 8])
    y = F.interpolate(x, size=[5])
    ref = np_nearest(xv, [5])
    close(y.to_vector(), ref.ravel(), tol=1e-5, msg="interp size 3d")
    assert list(y.shape) == [2, 3, 5]


def test_interpolate_backward():
    xv = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2)
    x = tv(xv.ravel(), [1, 1, 2, 2], True)
    y = F.interpolate(x, scale_factor=2.0)
    y.sum().backward()
    ref = np.zeros((1, 1, 2, 2), dtype=np.float32)
    src = np_nearest(xv, [4, 4])
    for i in range(2):
        for j in range(2):
            si = min(int(i * 0.5), 1)
            sj = min(int(j * 0.5), 1)
            ref[0, 0, si, sj] += 1.0
    close(np.array(x.grad.to_vector()), ref.ravel(), tol=1e-5, msg="interp backward")


def test_interpolate_bad_mode():
    x = tv([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
    try:
        F.interpolate(x, scale_factor=2.0, mode="bilinear")
    except Exception:
        return
    raise AssertionError("interpolate bilinear should throw")


def test_nll_loss():
    rng = np.random.RandomState(8)
    logp = rng.randn(4, 5).astype(np.float32)
    tgt = np.array([0, 2, 4, 1])
    x = tv(logp.ravel(), [4, 5], True)
    t = tv(tgt.astype(np.float32), [4])
    for red in ["mean", "sum", "none"]:
        y = F.nll_loss(x, t, reduction=red)
        li = -logp[np.arange(4), tgt]
        ref = {"mean": li.mean(), "sum": li.sum(), "none": li}[red]
        close(y.to_vector(), np.atleast_1d(ref).astype(np.float32), tol=1e-5,
              msg=f"nll {red}")


def test_nll_loss_ignore_weight():
    logp = np.array([[-0.5, -1.5, -2.5], [-2.0, -0.3, -1.0]], dtype=np.float32)
    tgt = np.array([0, -100])
    w = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    x = tv(logp.ravel(), [2, 3])
    t = tv(tgt.astype(np.float32), [2])
    ww = tv(w, [3])
    y = F.nll_loss(x, t, weight=ww, reduction="mean", ignore_index=-100)
    ref = (0.5 * 1.0) / 1.0
    close(y.to_vector(), [ref], tol=1e-5, msg="nll ignore+weight")


def test_nll_backward():
    logp = np.array([[-0.5, -1.5], [-2.0, -0.3]], dtype=np.float32)
    tgt = np.array([1, 0])
    x = tv(logp.ravel(), [2, 2], True)
    t = tv(tgt.astype(np.float32), [2])
    F.nll_loss(x, t, reduction="mean").backward()
    ref = np.zeros((2, 2), dtype=np.float32)
    ref[0, 1] = -0.5
    ref[1, 0] = -0.5
    close(np.array(x.grad.to_vector()), ref.ravel(), tol=1e-5, msg="nll backward")


def np_bce_logits(x, z):
    return np.maximum(x, 0) - x * z + np.log1p(np.exp(-np.abs(x)))


def test_bce_with_logits():
    rng = np.random.RandomState(9)
    xv = rng.randn(3, 4).astype(np.float32)
    zv = (rng.rand(3, 4) > 0.5).astype(np.float32)
    x = tv(xv.ravel(), [3, 4], True)
    z = tv(zv.ravel(), [3, 4])
    for red in ["mean", "sum", "none"]:
        y = F.bce_with_logits(x, z, reduction=red)
        li = np_bce_logits(xv, zv)
        ref = {"mean": li.mean(), "sum": li.sum(), "none": li}[red]
        close(y.to_vector(), np.atleast_1d(ref).ravel().astype(np.float32), tol=1e-4,
              msg=f"bce {red}")


def test_bce_backward():
    xv = np.array([[0.5, -1.0], [2.0, -0.2]], dtype=np.float32)
    zv = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    x = tv(xv.ravel(), [2, 2], True)
    z = tv(zv.ravel(), [2, 2])
    F.bce_with_logits(x, z, reduction="mean").backward()
    sig = 1.0 / (1.0 + np.exp(-xv))
    ref = (sig - zv) / 4.0
    close(np.array(x.grad.to_vector()), ref.ravel(), tol=1e-5, msg="bce backward")


def test_normalize():
    rng = np.random.RandomState(10)
    xv = rng.randn(3, 4).astype(np.float32) + 0.5
    x = tv(xv.ravel(), [3, 4])
    y = F.normalize(x, p=2.0, dim=1)
    ref = xv / np.linalg.norm(xv, ord=2, axis=1, keepdims=True)
    close(y.to_vector(), ref.ravel(), tol=1e-4, msg="normalize p2")
    y1 = F.normalize(x, p=1.0, dim=-1)
    ref1 = xv / np.abs(xv).sum(axis=1, keepdims=True)
    close(y1.to_vector(), ref1.ravel(), tol=1e-4, msg="normalize p1 negdim")


def test_glu():
    rng = np.random.RandomState(11)
    xv = rng.randn(2, 6).astype(np.float32)
    x = tv(xv.ravel(), [2, 6], True)
    y = F.glu(x, dim=-1)
    a, b = xv[..., :3], xv[..., 3:]
    ref = a * (1.0 / (1.0 + np.exp(-b)))
    close(y.to_vector(), ref.ravel(), tol=1e-5, msg="glu")
    assert list(y.shape) == [2, 3]


def test_glu_odd_dim_throws():
    x = tv([1.0, 2.0, 3.0], [1, 3])
    try:
        F.glu(x)
    except Exception:
        return
    raise AssertionError("glu odd dim should throw")


def test_elu():
    rng = np.random.RandomState(12)
    xv = rng.randn(2, 5).astype(np.float32)
    x = tv(xv.ravel(), [2, 5], True)
    y = F.elu(x, alpha=1.5)
    ref = np.where(xv > 0, xv, 1.5 * (np.exp(xv) - 1.0))
    close(y.to_vector(), ref.ravel(), tol=1e-5, msg="elu")


def test_elu_backward():
    xv = np.array([[-1.0, 0.5, -0.2, 2.0]], dtype=np.float32)
    x = tv(xv.ravel(), [1, 4], True)
    F.elu(x, alpha=1.5).sum().backward()
    ref = np.where(xv > 0, 1.0, 1.5 * np.exp(xv))
    close(np.array(x.grad.to_vector()), ref.ravel(), tol=1e-5, msg="elu backward")


def main():
    tests = [
        ("dropout_p0_identity", test_dropout_p0_identity),
        ("dropout_eval_passthrough", test_dropout_eval_passthrough),
        ("dropout_p1_zeros", test_dropout_p1_zeros),
        ("dropout_stats", test_dropout_stats),
        ("dropout_backward", test_dropout_backward),
        ("linear_2d", test_linear_2d),
        ("linear_no_bias_batched", test_linear_no_bias_batched),
        ("linear_backward", test_linear_backward),
        ("pad_2d", test_pad_2d),
        ("pad_value_3d", test_pad_value_3d),
        ("pad_backward", test_pad_backward),
        ("pad_bad_mode", test_pad_bad_mode),
        ("interpolate_scale_4d", test_interpolate_scale_4d),
        ("interpolate_size_3d", test_interpolate_size_3d),
        ("interpolate_backward", test_interpolate_backward),
        ("interpolate_bad_mode", test_interpolate_bad_mode),
        ("nll_loss", test_nll_loss),
        ("nll_loss_ignore_weight", test_nll_loss_ignore_weight),
        ("nll_backward", test_nll_backward),
        ("bce_with_logits", test_bce_with_logits),
        ("bce_backward", test_bce_backward),
        ("normalize", test_normalize),
        ("glu", test_glu),
        ("glu_odd_dim_throws", test_glu_odd_dim_throws),
        ("elu", test_elu),
        ("elu_backward", test_elu_backward),
    ]
    for name, fn in tests:
        run(name, fn)
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    for name, status, msg in RESULTS:
        extra = f" [{msg}]" if msg else ""
        print(f"{status} {name}{extra}", flush=True)
    print(f"TOTAL: {npass}/{len(RESULTS)} passed", flush=True)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
