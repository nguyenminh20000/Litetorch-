import math

import numpy as np

import litetorch as lt

CPU = lt.Device("cpu")

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name} {detail}")


def vec(t):
    return np.array(t.to(CPU).to_vector(), dtype=np.float64)


def t2np(t):
    return np.array(t.to(CPU).to_vector(), dtype=np.float64).reshape(t.shape)


def set_tensor(t, arr):
    flat = np.asarray(arr, dtype=np.float32).ravel()
    src = lt.Tensor.from_vector([float(v) for v in flat], list(t.shape), CPU)
    t.copy_(src)


def alltclose(a, b, tol=1e-4):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.allclose(a, b, atol=tol, rtol=tol)


def mkt(shape, vals=None):
    n = 1
    for s in shape:
        n *= s
    if vals is None:
        vals = [(i % 7 - 3) * 0.25 for i in range(n)]
    return lt.Tensor.from_vector([float(v) for v in vals], list(shape), CPU)


# ---------------- RNNCell ----------------
def test_rnncell():
    cell = lt.nn.RNNCell(3, 4)
    x = np.array([[0.5, -0.25, 0.75], [0.1, 0.2, -0.3]])
    h = np.array([[0.1, 0.2, 0.3, -0.1], [-0.2, 0.4, 0.0, 0.1]])
    wih = np.arange(12, dtype=np.float64).reshape(4, 3) * 0.1 - 0.3
    whh = np.arange(16, dtype=np.float64).reshape(4, 4) * 0.05 - 0.2
    bih = np.array([0.1, -0.1, 0.2, 0.0])
    bhh = np.array([0.0, 0.1, -0.2, 0.15])
    set_tensor(cell.weight_ih, wih)
    set_tensor(cell.weight_hh, whh)
    set_tensor(cell.bias_ih, bih)
    set_tensor(cell.bias_hh, bhh)
    ref = np.tanh(x @ wih.T + bih + h @ whh.T + bhh)
    out = cell.forward(mkt((2, 3), x.ravel()), mkt((2, 4), h.ravel()))
    check("rnncell_tanh", alltclose(t2np(out), ref), f"maxerr={np.abs(t2np(out)-ref).max()}")
    check("rnncell_shape", list(out.shape) == [2, 4], str(out.shape))


def test_rnncell_relu_nobias():
    cell = lt.nn.RNNCell(2, 3, bias=False, nonlinearity="relu")
    check("rnncell_nobias", cell.bias_ih is None and cell.bias_hh is None)
    x = np.array([[0.5, -0.5]])
    h = np.array([[0.2, -0.3, 0.1]])
    wih = np.eye(3, 2, dtype=np.float64) * 0.5
    whh = np.eye(3, dtype=np.float64) * 0.25
    set_tensor(cell.weight_ih, wih)
    set_tensor(cell.weight_hh, whh)
    ref = np.maximum(x @ wih.T + h @ whh.T, 0.0)
    out = cell.forward(mkt((1, 2), x.ravel()), mkt((1, 3), h.ravel()))
    check("rnncell_relu", alltclose(t2np(out), ref))


# ---------------- LSTMCell ----------------
def test_lstmcell():
    cell = lt.nn.LSTMCell(3, 2)
    x = np.array([[0.5, -0.25, 0.75]])
    h = np.array([[0.1, -0.2]])
    c = np.array([[0.3, 0.05]])
    rng = np.random.RandomState(0)
    wih = rng.randn(8, 3) * 0.3
    whh = rng.randn(8, 2) * 0.3
    bih = rng.randn(8) * 0.1
    bhh = rng.randn(8) * 0.1
    set_tensor(cell.weight_ih, wih)
    set_tensor(cell.weight_hh, whh)
    set_tensor(cell.bias_ih, bih)
    set_tensor(cell.bias_hh, bhh)
    gates = x @ wih.T + bih + h @ whh.T + bhh
    i = 1 / (1 + np.exp(-gates[:, 0:2]))
    f = 1 / (1 + np.exp(-gates[:, 2:4]))
    g = np.tanh(gates[:, 4:6])
    o = 1 / (1 + np.exp(-gates[:, 6:8]))
    cn = f * c + i * g
    hn = o * np.tanh(cn)
    hy, cy = cell.forward(mkt((1, 3), x.ravel()), mkt((1, 2), h.ravel()), mkt((1, 2), c.ravel()))
    check("lstmcell_h", alltclose(t2np(hy), hn), f"maxerr={np.abs(t2np(hy)-hn).max()}")
    check("lstmcell_c", alltclose(t2np(cy), cn))


# ---------------- GRUCell ----------------
def test_grucell():
    cell = lt.nn.GRUCell(3, 2)
    x = np.array([[0.5, -0.25, 0.75]])
    h = np.array([[0.1, -0.2]])
    rng = np.random.RandomState(1)
    wih = rng.randn(6, 3) * 0.3
    whh = rng.randn(6, 2) * 0.3
    bih = rng.randn(6) * 0.1
    bhh = rng.randn(6) * 0.1
    set_tensor(cell.weight_ih, wih)
    set_tensor(cell.weight_hh, whh)
    set_tensor(cell.bias_ih, bih)
    set_tensor(cell.bias_hh, bhh)
    gi = x @ wih.T + bih
    gh = h @ whh.T + bhh
    r = 1 / (1 + np.exp(-(gi[:, 0:2] + gh[:, 0:2])))
    z = 1 / (1 + np.exp(-(gi[:, 2:4] + gh[:, 2:4])))
    n = np.tanh(gi[:, 4:6] + r * gh[:, 4:6])
    ref = (1 - z) * n + z * h
    out = cell.forward(mkt((1, 3), x.ravel()), mkt((1, 2), h.ravel()))
    check("grucell", alltclose(t2np(out), ref), f"maxerr={np.abs(t2np(out)-ref).max()}")


# ---------------- RNN (multi-layer) ----------------
def ref_rnn(x, cells_w, nonlinearity, bidirectional=False):
    S, B, _ = x.shape
    dirs = 2 if bidirectional else 1
    hns = []
    for d in range(dirs):
        wih, whh, bih, bhh = cells_w[d]
        h = np.zeros((B, wih.shape[0]))
        outs = []
        seq = range(S) if d == 0 else range(S - 1, -1, -1)
        for t in seq:
            pre = x[t] @ wih.T + bih + h @ whh.T + bhh
            h = np.tanh(pre) if nonlinearity == "tanh" else np.maximum(pre, 0)
            outs.append(h)
        if d == 1:
            outs = outs[::-1]
        hns.append(h)
        if d == 0:
            out = np.stack(outs)
        else:
            out = np.concatenate([out, np.stack(outs)], axis=2)
    return out, np.stack(hns)


def get_cell_w(cell):
    return (t2np(cell.weight_ih), t2np(cell.weight_hh), t2np(cell.bias_ih), t2np(cell.bias_hh))


def test_rnn():
    m = lt.nn.RNN(3, 4, num_layers=1, batch_first=False)
    x = np.arange(24, dtype=np.float64).reshape(4, 2, 3) * 0.05 - 0.3
    ref_out, ref_hn = ref_rnn(x, [get_cell_w(m.cells[0])], "tanh")
    out, hn = m.forward_full(mkt((4, 2, 3), x.ravel()))
    check("rnn_out", alltclose(t2np(out), ref_out), f"maxerr={np.abs(t2np(out)-ref_out).max()}")
    check("rnn_hn", alltclose(t2np(hn), ref_hn))
    check("rnn_hn_shape", list(hn.shape) == [1, 2, 4], str(hn.shape))


def test_rnn_bidirectional_batchfirst():
    m = lt.nn.RNN(3, 4, num_layers=2, batch_first=True, bidirectional=True)
    m.eval()
    x = np.arange(24, dtype=np.float64).reshape(2, 4, 3) * 0.05 - 0.3
    out, hn = m.forward_full(mkt((2, 4, 3), x.ravel()))
    check("rnn_bi_out_shape", list(out.shape) == [2, 4, 8], str(out.shape))
    check("rnn_bi_hn_shape", list(hn.shape) == [4, 2, 4], str(hn.shape))
    check("rnn_bi_finite", np.all(np.isfinite(t2np(out))))


def test_rnn_grad():
    m = lt.nn.RNN(3, 4, num_layers=2)
    x = mkt((5, 2, 3))
    x.requires_grad = True
    out, hn = m.forward_full(x)
    loss = lt.Ops.sum(out)
    loss.backward()
    grads = [p.grad for p in m.parameters()]
    check("rnn_grad_all", all(g is not None for g in grads), f"{sum(g is None for g in grads)} missing")
    check("rnn_grad_finite", all(np.all(np.isfinite(t2np(g))) for g in grads if g is not None))
    check("rnn_input_grad", x.grad is not None and np.all(np.isfinite(t2np(x.grad))))


# ---------------- LSTM ----------------
def ref_lstm(x, cells_w, bidirectional=False):
    S, B, _ = x.shape
    dirs = 2 if bidirectional else 1
    hns, cns = [], []
    for d in range(dirs):
        wih, whh, bih, bhh = cells_w[d]
        H = wih.shape[0] // 4
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        outs = []
        seq = range(S) if d == 0 else range(S - 1, -1, -1)
        for t in seq:
            gates = x[t] @ wih.T + bih + h @ whh.T + bhh
            i = 1 / (1 + np.exp(-gates[:, 0:H]))
            f = 1 / (1 + np.exp(-gates[:, H:2 * H]))
            gg = np.tanh(gates[:, 2 * H:3 * H])
            o = 1 / (1 + np.exp(-gates[:, 3 * H:4 * H]))
            c = f * c + i * gg
            h = o * np.tanh(c)
            outs.append(h)
        if d == 1:
            outs = outs[::-1]
        hns.append(h)
        cns.append(c)
        out_d = np.stack(outs)
        out = out_d if d == 0 else np.concatenate([out, out_d], axis=2)
    return out, np.stack(hns), np.stack(cns)


def test_lstm():
    m = lt.nn.LSTM(3, 4, num_layers=1, batch_first=False)
    x = np.arange(24, dtype=np.float64).reshape(4, 2, 3) * 0.05 - 0.3
    ref_out, ref_hn, ref_cn = ref_lstm(x, [get_cell_w(m.cells[0])], False)
    out, hn, cn = m.forward_full(mkt((4, 2, 3), x.ravel()))
    check("lstm_out", alltclose(t2np(out), ref_out), f"maxerr={np.abs(t2np(out)-ref_out).max()}")
    check("lstm_hn", alltclose(t2np(hn), ref_hn))
    check("lstm_cn", alltclose(t2np(cn), ref_cn))


def test_lstm_bi_2layer():
    m = lt.nn.LSTM(4, 5, num_layers=2, batch_first=True, bidirectional=True)
    m.eval()
    x = mkt((3, 6, 4))
    out, hn, cn = m.forward_full(x)
    check("lstm_bi_shapes", list(out.shape) == [3, 6, 10] and list(hn.shape) == [4, 3, 5]
          and list(cn.shape) == [4, 3, 5], f"{out.shape} {hn.shape} {cn.shape}")
    check("lstm_bi_finite", np.all(np.isfinite(t2np(out))))


def test_lstm_grad():
    m = lt.nn.LSTM(3, 4)
    x = mkt((5, 2, 3))
    x.requires_grad = True
    out, hn, cn = m.forward_full(x)
    lt.Ops.sum(out).backward()
    grads = [p.grad for p in m.parameters()]
    check("lstm_grad_all", all(g is not None for g in grads))
    check("lstm_grad_finite", all(np.all(np.isfinite(t2np(g))) for g in grads if g is not None))


# ---------------- GRU ----------------
def test_gru():
    m = lt.nn.GRU(3, 4, num_layers=1)
    x = np.arange(24, dtype=np.float64).reshape(4, 2, 3) * 0.05 - 0.3

    def ref_gru_cell(xt, h, w):
        wih, whh, bih, bhh = w
        H = wih.shape[0] // 3
        gi = xt @ wih.T + bih
        gh = h @ whh.T + bhh
        r = 1 / (1 + np.exp(-(gi[:, 0:H] + gh[:, 0:H])))
        z = 1 / (1 + np.exp(-(gi[:, H:2 * H] + gh[:, H:2 * H])))
        n = np.tanh(gi[:, 2 * H:3 * H] + r * gh[:, 2 * H:3 * H])
        return (1 - z) * n + z * h

    w = get_cell_w(m.cells[0])
    h = np.zeros((2, 4))
    outs = []
    for t in range(4):
        h = ref_gru_cell(x[t], h, w)
        outs.append(h)
    ref_out = np.stack(outs)
    out, hn = m.forward_full(mkt((4, 2, 3), x.ravel()))
    check("gru_out", alltclose(t2np(out), ref_out), f"maxerr={np.abs(t2np(out)-ref_out).max()}")
    check("gru_hn", alltclose(t2np(hn), ref_out[-1:]))


def test_gru_grad():
    m = lt.nn.GRU(3, 4, num_layers=2, bidirectional=True)
    m.eval()
    x = mkt((5, 2, 3))
    x.requires_grad = True
    out, hn = m.forward_full(x)
    lt.Ops.sum(out).backward()
    grads = [p.grad for p in m.parameters()]
    check("gru_grad_all", all(g is not None for g in grads))
    check("gru_grad_finite", all(np.all(np.isfinite(t2np(g))) for g in grads if g is not None))


# ---------------- TransformerEncoder ----------------
def test_transformer_encoder_layer():
    layer = lt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
    layer.eval()
    x = mkt((5, 2, 8))
    out = layer.forward(x)
    check("tel_shape", list(out.shape) == [5, 2, 8], str(out.shape))
    check("tel_finite", np.all(np.isfinite(t2np(out))))
    out2 = layer.forward(x)
    check("tel_deterministic", alltclose(t2np(out), t2np(out2), tol=1e-6))


def test_transformer_encoder():
    layer = lt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
    enc = lt.nn.TransformerEncoder(layer, 2)
    enc.eval()
    x = mkt((5, 2, 8))
    out = enc.forward(x)
    check("te_shape", list(out.shape) == [5, 2, 8], str(out.shape))
    check("te_finite", np.all(np.isfinite(t2np(out))))
    check("te_params", len(enc.parameters()) > 0, str(len(enc.parameters())))
    x2 = mkt((5, 2, 8))
    x2.requires_grad = True
    lt.Ops.sum(enc.forward(x2)).backward()
    check("te_grad", all(p.grad is not None for p in enc.parameters()))


# ---------------- EmbeddingBag ----------------
def test_embedding_bag():
    for mode in ["sum", "mean", "max"]:
        eb = lt.nn.EmbeddingBag(6, 4, mode=mode)
        w = np.arange(24, dtype=np.float64).reshape(6, 4) * 0.1
        set_tensor(eb.weight, w)
        idx = np.array([0, 2, 2, 5, 1], dtype=np.int64)
        offsets = np.array([0, 2, 5], dtype=np.int64)
        bags = []
        for b in range(3):
            s, e = offsets[b], offsets[b + 1] if b + 1 < 3 else 5
            rows = w[idx[s:e]]
            if len(rows) == 0:
                bags.append(np.zeros(4))
            else:
                bags.append({"sum": rows.sum(0), "mean": rows.mean(0), "max": rows.max(0)}[mode])
        ref = np.stack(bags)
        it = lt.Tensor.from_vector([float(v) for v in idx], [5], CPU)
        off = lt.Tensor.from_vector([float(v) for v in offsets], [3], CPU)
        out = eb.forward(it, off)
        check(f"ebag_{mode}", alltclose(t2np(out), ref), f"maxerr={np.abs(t2np(out)-ref).max()}")
        check(f"ebag_{mode}_shape", list(out.shape) == [3, 4], str(out.shape))


# ---------------- ConvTranspose ----------------
def ref_ct2d(x, w, b, stride, padding, groups=1, dilation=1):
    N, Cin, H, Wd = x.shape
    _, CoG, kH, kW = w.shape
    Cout, CinG = CoG * groups, Cin // groups
    Ho = (H - 1) * stride - 2 * padding + dilation * (kH - 1) + 1
    Wo = (Wd - 1) * stride - 2 * padding + dilation * (kW - 1) + 1
    y = np.zeros((N, Cout, Ho, Wo))
    for n in range(N):
        for g in range(groups):
            for oc in range(CoG):
                for oh in range(Ho):
                    for ow in range(Wo):
                        acc = b[g * CoG + oc] if b is not None else 0.0
                        for ic in range(CinG):
                            for kh in range(kH):
                                for kw in range(kW):
                                    ih = oh + padding - kh * dilation
                                    iw = ow + padding - kw * dilation
                                    if ih % stride == 0 and iw % stride == 0:
                                        ih //= stride
                                        iw //= stride
                                        if 0 <= ih < H and 0 <= iw < Wd:
                                            acc += x[n, g * CinG + ic, ih, iw] * w[g * CinG + ic, oc, kh, kw]
                        y[n, g * CoG + oc, oh, ow] = acc
    return y


def test_convtranspose2d():
    m = lt.nn.ConvTranspose2d(2, 3, 2, stride=2, padding=0, bias=True)
    x = np.arange(8, dtype=np.float64).reshape(1, 2, 2, 2) * 0.25 - 0.5
    rng = np.random.RandomState(2)
    w = rng.randn(2, 3, 2, 2) * 0.3
    b = rng.randn(3) * 0.2
    set_tensor(m.weight, w)
    set_tensor(m.bias, b)
    ref = ref_ct2d(x, w, b, 2, 0)
    out = m.forward(mkt((1, 2, 2, 2), x.ravel()))
    check("ct2d", alltclose(t2np(out), ref, tol=2e-4), f"maxerr={np.abs(t2np(out)-ref).max()}")
    check("ct2d_shape", list(out.shape) == [1, 3, 4, 4], str(out.shape))


def test_convtranspose2d_stride1_pad():
    m = lt.nn.ConvTranspose2d(2, 2, 3, stride=1, padding=1, bias=False)
    x = np.arange(18, dtype=np.float64).reshape(1, 2, 3, 3) * 0.1
    rng = np.random.RandomState(3)
    w = rng.randn(2, 2, 3, 3) * 0.2
    set_tensor(m.weight, w)
    ref = ref_ct2d(x, w, None, 1, 1)
    out = m.forward(mkt((1, 2, 3, 3), x.ravel()))
    check("ct2d_s1p1", alltclose(t2np(out), ref, tol=2e-4), f"maxerr={np.abs(t2np(out)-ref).max()}")


def test_convtranspose2d_groups():
    m = lt.nn.ConvTranspose2d(4, 4, 2, stride=1, padding=0, groups=2, bias=False)
    x = np.arange(16, dtype=np.float64).reshape(1, 4, 2, 2) * 0.1
    rng = np.random.RandomState(4)
    w = rng.randn(4, 2, 2, 2) * 0.2
    set_tensor(m.weight, w)
    ref = ref_ct2d(x, w, None, 1, 0, groups=2)
    out = m.forward(mkt((1, 4, 2, 2), x.ravel()))
    check("ct2d_groups", alltclose(t2np(out), ref, tol=2e-4), f"maxerr={np.abs(t2np(out)-ref).max()}")


def test_convtranspose_backward():
    m = lt.nn.ConvTranspose2d(2, 3, 2, stride=2, padding=1)
    x = mkt((1, 2, 3, 3))
    x.requires_grad = True
    out = m.forward(x)
    lt.Ops.sum(out).backward()
    check("ct2d_bwd_x", x.grad is not None and np.all(np.isfinite(t2np(x.grad))))
    check("ct2d_bwd_w", m.weight.grad is not None and np.all(np.isfinite(t2np(m.weight.grad))))
    check("ct2d_bwd_b", m.bias.grad is not None and np.all(np.isfinite(t2np(m.bias.grad))))
    gw = t2np(m.weight.grad)
    check("ct2d_bwd_w_nonzero", np.abs(gw).max() > 1e-8, f"max={np.abs(gw).max()}")


def test_convtranspose1d_3d_shapes():
    m1 = lt.nn.ConvTranspose1d(2, 3, 2, stride=2)
    o1 = m1.forward(mkt((1, 2, 4)))
    check("ct1d_shape", list(o1.shape) == [1, 3, 8], str(o1.shape))
    m3 = lt.nn.ConvTranspose3d(2, 2, 2, stride=2, padding=0)
    o3 = m3.forward(mkt((1, 2, 2, 2, 2)))
    check("ct3d_shape", list(o3.shape) == [1, 2, 4, 4, 4], str(o3.shape))
    check("ct3d_finite", np.all(np.isfinite(t2np(o3))))


# ---------------- InstanceNorm ----------------
def ref_instancenorm(x, w, b, eps):
    N, C = x.shape[:2]
    y = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            v = x[n, c].ravel()
            mu, var = v.mean(), v.var()
            xn = (v - mu) / math.sqrt(var + eps)
            if w is not None:
                xn = xn * w[c] + b[c]
            y[n, c] = xn.reshape(x[n, c].shape)
    return y


def test_instancenorm():
    for cls, shape in [(lt.nn.InstanceNorm1d, (2, 3, 5)), (lt.nn.InstanceNorm2d, (2, 3, 4, 4)),
                       (lt.nn.InstanceNorm3d, (2, 3, 2, 2, 2))]:
        for affine in [False, True]:
            m = cls(3, affine=affine)
            m.eval()
            x = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape) * 0.07 - 1.0
            w = np.array([1.5, 0.5, 2.0]) if affine else None
            b = np.array([0.1, -0.2, 0.3]) if affine else None
            if affine:
                set_tensor(m.weight, w)
                set_tensor(m.bias, b)
            ref = ref_instancenorm(x, w, b, 1e-5)
            out = m.forward(mkt(shape, x.ravel()))
            check(f"in{len(shape)-2}d_aff{int(affine)}", alltclose(t2np(out), ref, tol=2e-4),
                  f"maxerr={np.abs(t2np(out)-ref).max()}")


# ---------------- BatchNorm3d / SyncBatchNorm ----------------
def ref_batchnorm(x, w, b, eps):
    N, C = x.shape[:2]
    y = np.zeros_like(x)
    for c in range(C):
        v = x[:, c].ravel()
        mu, var = v.mean(), v.var()
        xn = (v - mu) / math.sqrt(var + eps)
        xn = xn * w[c] + b[c]
        y[:, c] = xn.reshape(x[:, c].shape)
    return y


def test_batchnorm3d():
    m = lt.nn.BatchNorm3d(3)
    x = np.arange(2 * 3 * 2 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2, 2) * 0.05 - 0.5
    w = np.array([1.0, 2.0, 0.5])
    b = np.array([0.1, -0.1, 0.2])
    set_tensor(m.weight, w)
    set_tensor(m.bias, b)
    ref = ref_batchnorm(x, w, b, 1e-5)
    out = m.forward(mkt((2, 3, 2, 2, 2), x.ravel()))
    check("bn3d", alltclose(t2np(out), ref, tol=2e-4), f"maxerr={np.abs(t2np(out)-ref).max()}")
    m.eval()
    out_e = m.forward(mkt((2, 3, 2, 2, 2), x.ravel()))
    check("bn3d_eval_finite", np.all(np.isfinite(t2np(out_e))))
    check("bn3d_running_updated", np.abs(t2np(m.running_mean)).max() > 1e-9)


def test_syncbn():
    m = lt.nn.SyncBatchNorm(3)
    x = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4) * 0.1 - 0.3
    w = np.array([1.0, 0.5, 2.0])
    b = np.array([0.0, 0.1, -0.1])
    set_tensor(m.weight, w)
    set_tensor(m.bias, b)
    ref = ref_batchnorm(x, w, b, 1e-5)
    out = m.forward(mkt((2, 3, 4), x.ravel()))
    check("syncbn", alltclose(t2np(out), ref, tol=2e-4), f"maxerr={np.abs(t2np(out)-ref).max()}")


# ---------------- activations ----------------
def test_elu():
    m = lt.nn.ELU(1.0)
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    ref = np.where(x >= 0, x, 1.0 * (np.exp(x) - 1))
    out = m.forward(mkt((5,), x))
    check("elu", alltclose(t2np(out), ref, tol=1e-6))


def test_prelu():
    m = lt.nn.PReLU(1, init=0.25)
    x = np.array([[-2.0, 0.5], [1.0, -1.0]])
    ref = np.where(x >= 0, x, 0.25 * x)
    out = m.forward(mkt((2, 2), x.ravel()))
    check("prelu_scalar", alltclose(t2np(out), ref, tol=1e-6))
    m2 = lt.nn.PReLU(2, init=0.5)
    set_tensor(m2.weight, np.array([0.5, 0.1]))
    ref2 = np.stack([np.where(x[:, 0] >= 0, x[:, 0], 0.5 * x[:, 0]),
                     np.where(x[:, 1] >= 0, x[:, 1], 0.1 * x[:, 1])], axis=1)
    out2 = m2.forward(mkt((2, 2), x.ravel()))
    check("prelu_channel", alltclose(t2np(out2), ref2, tol=1e-6))
    x3 = mkt((2, 2), x.ravel())
    x3.requires_grad = True
    lt.Ops.sum(m2.forward(x3)).backward()
    check("prelu_grad", m2.weight.grad is not None and np.all(np.isfinite(t2np(m2.weight.grad))))


def test_glu():
    m = lt.nn.GLU(-1)
    x = np.array([[0.5, -0.5, 1.0, -1.0]])
    a, b = x[:, :2], x[:, 2:]
    ref = a * (1 / (1 + np.exp(-b)))
    out = m.forward(mkt((1, 4), x.ravel()))
    check("glu", alltclose(t2np(out), ref, tol=1e-6))
    check("glu_shape", list(out.shape) == [1, 2], str(out.shape))


# ---------------- PixelShuffle ----------------
def test_pixelshuffle():
    m = lt.nn.PixelShuffle(2)
    x = np.arange(16, dtype=np.float64).reshape(1, 4, 2, 2)
    ref = x.reshape(1, 1, 2, 2, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(1, 1, 4, 4)
    out = m.forward(mkt((1, 4, 2, 2), x.ravel()))
    check("pixelshuffle", alltclose(t2np(out), ref, tol=1e-6))
    u = lt.nn.PixelUnshuffle(2)
    back = u.forward(out)
    check("pixelunshuffle_roundtrip", alltclose(t2np(back), x, tol=1e-6))
    check("pixelunshuffle_shape", list(back.shape) == [1, 4, 2, 2], str(back.shape))


# ---------------- Identity / Unflatten ----------------
def test_identity_unflatten():
    m = lt.nn.Identity()
    x = mkt((2, 3))
    out = m.forward(x)
    check("identity", alltclose(t2np(out), t2np(x), tol=0))
    u = lt.nn.Unflatten(1, [2, 3])
    x2 = mkt((4, 6))
    o2 = u.forward(x2)
    check("unflatten", list(o2.shape) == [4, 2, 3], str(o2.shape))
    check("unflatten_vals", alltclose(t2np(o2).ravel(), t2np(x2).ravel(), tol=0))


# ---------------- misc ----------------
def test_state_dict_keys():
    m = lt.nn.LSTM(3, 4, num_layers=2)
    keys = [k for k, _ in m.named_parameters()]
    check("lstm_named_params", any("weight_ih" in k for k in keys), str(keys[:4]))
    check("lstm_param_count", len(m.parameters()) == 8, str(len(m.parameters())))
    m2 = lt.nn.GRU(3, 4, num_layers=2, bidirectional=True)
    check("gru_bi_param_count", len(m2.parameters()) == 16, str(len(m2.parameters())))


def test_transformer_encoder_batchfirst():
    layer = lt.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True)
    layer.eval()
    x = mkt((2, 5, 8))
    out = layer.forward(x)
    check("tel_bf_shape", list(out.shape) == [2, 5, 8], str(out.shape))
    check("tel_bf_finite", np.all(np.isfinite(t2np(out))))


test_rnncell()
test_rnncell_relu_nobias()
test_lstmcell()
test_grucell()
test_rnn()
test_rnn_bidirectional_batchfirst()
test_rnn_grad()
test_lstm()
test_lstm_bi_2layer()
test_lstm_grad()
test_gru()
test_gru_grad()
test_transformer_encoder_layer()
test_transformer_encoder()
test_embedding_bag()
test_convtranspose2d()
test_convtranspose2d_stride1_pad()
test_convtranspose2d_groups()
test_convtranspose_backward()
test_convtranspose1d_3d_shapes()
test_instancenorm()
test_batchnorm3d()
test_syncbn()
test_elu()
test_prelu()
test_glu()
test_pixelshuffle()
test_identity_unflatten()
test_state_dict_keys()
test_transformer_encoder_batchfirst()

print(f"\nTOTAL: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
