import math

import litetorch as lt


DEV = lt.Device("cuda:0") if lt.is_gpu_available() else lt.Device("cpu")
RESULTS = []


def tv(data, shape, requires_grad=False):
    return lt.Tensor.from_vector(data, shape, DEV, requires_grad)


def vals(n, lo=0.1, hi=1.0):
    out = []
    s = 987654321
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(lo + (hi - lo) * (s / 0x7FFFFFFF))
    return out


def grad_ok(t):
    g = t.grad
    if g is None:
        return False, "grad is None"
    for x in g.to_vector():
        if not math.isfinite(x):
            return False, "grad not finite"
    return True, ""


def check(t, label):
    ok, msg = grad_ok(t)
    assert ok, label + ": " + msg


def check_params(params, label):
    assert len(params) > 0, label + ": no params"
    for i, p in enumerate(params):
        ok, msg = grad_ok(p)
        assert ok, label + f" param{i}: " + msg


def run(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:
        RESULTS.append((name, "FAIL", type(e).__name__ + ": " + str(e)))


def test_embedding():
    m = lt.nn.Embedding(8, 4)
    m.to(DEV)
    idx = tv([0.0, 1.0, 3.0, 7.0], [4], False)
    out = m.forward(idx)
    assert out.shape == [4, 4], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(m.parameters(), "embedding")


def test_dropout():
    m = lt.nn.Dropout(0.5)
    m.to(DEV)
    x = tv(vals(32), [8, 4], True)
    m.eval()
    y_eval = m.forward(x)
    assert y_eval.to_vector() == x.to_vector(), "eval mode must be identity"
    m.train()
    y_train = m.forward(x)
    lt.Ops.sum(y_train).backward()
    check(x, "dropout")


def test_mha_self():
    m = lt.nn.MultiHeadAttention(16, 4)
    m.to(DEV)
    x = tv(vals(2 * 4 * 16), [2, 4, 16], True)
    out = m.forward(x)
    assert out.shape == [2, 4, 16], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(m.parameters(), "mha_self")


def test_mha_cross():
    m = lt.nn.MultiHeadAttention(16, 4)
    m.to(DEV)
    q = tv(vals(2 * 3 * 16), [2, 3, 16], True)
    kv = tv(vals(2 * 5 * 16), [2, 5, 16], False)
    out = m.forward(q, kv, kv)
    assert out.shape == [2, 3, 16], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(m.parameters(), "mha_cross")


def test_decoder_layer():
    m = lt.nn.TransformerDecoderLayer(16, 4, 32)
    m.to(DEV)
    x = tv(vals(2 * 4 * 16), [2, 4, 16], True)
    out = m.forward(x, x)
    assert out.shape == [2, 4, 16], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(m.parameters(), "decoder")


def test_decoder_layer_nomem():
    m = lt.nn.TransformerDecoderLayer(16, 4, 32)
    m.to(DEV)
    x = tv(vals(2 * 4 * 16), [2, 4, 16], True)
    out = m.forward(x)
    lt.Ops.sum(out).backward()
    params = m.parameters()
    used = params[:4] + params[8:14] + params[16:]
    check_params(used, "decoder_nomem")


def test_conv3d():
    m = lt.nn.Conv3d(2, 4, 3, padding=1)
    m.to(DEV)
    x = tv(vals(2 * 2 * 4 * 4 * 4), [2, 2, 4, 4, 4], True)
    out = m.forward(x)
    assert out.shape == [2, 4, 4, 4, 4], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(m.parameters(), "conv3d")
    check(x, "conv3d_input")


def test_maxpool2d():
    m = lt.nn.MaxPool2d(2, 2)
    m.to(DEV)
    x = tv(vals(2 * 2 * 4 * 4), [2, 2, 4, 4], True)
    out = m.forward(x)
    assert out.shape == [2, 2, 2, 2], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check(x, "maxpool2d")


def test_maxpool3d():
    m = lt.nn.MaxPool3d(2, 2)
    m.to(DEV)
    x = tv(vals(2 * 2 * 4 * 4 * 4), [2, 2, 4, 4, 4], True)
    out = m.forward(x)
    assert out.shape == [2, 2, 2, 2, 2], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check(x, "maxpool3d")


def test_adaptive_avgpool2d():
    m = lt.nn.AdaptiveAvgPool2d(2, 2)
    m.to(DEV)
    x = tv(vals(2 * 3 * 4 * 4), [2, 3, 4, 4], True)
    out = m.forward(x)
    assert out.shape == [2, 3, 2, 2], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check(x, "adaptive_avgpool2d")


def test_sequential():
    m = lt.nn.Sequential([lt.nn.Linear(8, 16), lt.nn.ReLU(), lt.nn.Linear(16, 4)])
    m.to(DEV)
    params = m.parameters()
    assert len(params) == 4, f"expected 4 params, got {len(params)}"
    x = tv(vals(3 * 8), [3, 8], True)
    out = m.forward(x)
    assert out.shape == [3, 4], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(params, "sequential")


def test_sequential_to_device():
    m = lt.nn.Sequential([lt.nn.Linear(4, 4)])
    m.to(DEV)
    for p in m.parameters():
        assert "cuda" in p.device.to_string() or "gpu" in p.device.to_string() or "cpu" in DEV.to_string(), \
            f"param not on device: {p.device.to_string()}"


def test_moe():
    m = lt.nn.MoELinear(8, 8, 4, 2)
    m.to(DEV)
    x = tv(vals(3 * 8), [3, 8], True)
    out = m.forward(x)
    assert out.shape == [3, 8], f"shape {out.shape}"
    lt.Ops.sum(out).backward()
    check_params(m.parameters(), "moe")


def main():
    print(f"device: {DEV.to_string()}", flush=True)
    tests = [
        ("embedding", test_embedding),
        ("dropout", test_dropout),
        ("mha_self", test_mha_self),
        ("mha_cross", test_mha_cross),
        ("decoder", test_decoder_layer),
        ("decoder_nomem", test_decoder_layer_nomem),
        ("conv3d", test_conv3d),
        ("maxpool2d", test_maxpool2d),
        ("maxpool3d", test_maxpool3d),
        ("adaptive_avgpool2d", test_adaptive_avgpool2d),
        ("sequential", test_sequential),
        ("sequential_to_device", test_sequential_to_device),
        ("moe", test_moe),
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
