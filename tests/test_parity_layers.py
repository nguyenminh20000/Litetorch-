import math

import numpy as np
import pytest

import litetorch
from litetorch import Ops, nn


def tv(data, requires_grad=False):
    return litetorch.tensor(data, requires_grad=requires_grad)


def to_np(t):
    return np.array(t.to_vector(), dtype=np.float64).reshape(tuple(t.shape))


def grad_np(t):
    g = t.grad
    assert g is not None, "grad is None"
    return np.array(g.to_vector(), dtype=np.float64).reshape(tuple(t.shape))


rng = np.random.default_rng(42)


def finite_diff_grad(fn, x_np, eps=1e-4):
    grad = np.zeros_like(x_np)
    it = np.nditer(x_np, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x_np.copy()
        xp[idx] += eps
        xm = x_np.copy()
        xm[idx] -= eps
        grad[idx] = (fn(xp) - fn(xm)) / (2 * eps)
        it.iternext()
    return grad


class TestGroupNorm:
    def gn_ref(self, x, G, w, b, eps):
        N, C = x.shape[:2]
        xr = x.reshape(N, G, -1)
        mu = xr.mean(axis=2, keepdims=True)
        var = xr.var(axis=2, keepdims=True)
        xh = (xr - mu) / np.sqrt(var + eps)
        xh = xh.reshape(x.shape)
        if w is not None:
            s = [1, C] + [1] * (x.ndim - 2)
            xh = xh * w.reshape(s) + b.reshape(s)
        return xh

    def test_forward_affine(self):
        N, C, H, Wd, G = 2, 6, 4, 5, 3
        x = rng.normal(size=(N, C, H, Wd))
        m = nn.GroupNorm(G, C)
        w = rng.normal(size=(C,))
        b = rng.normal(size=(C,))
        m.weight = tv(w.tolist())
        m.bias = tv(b.tolist())
        out = to_np(m.forward(tv(x.tolist())))
        ref = self.gn_ref(x, G, w, b, 1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_forward_no_affine_3d(self):
        N, C, L, G = 2, 4, 8, 2
        x = rng.normal(size=(N, C, L))
        m = nn.GroupNorm(G, C, affine=False)
        out = to_np(m.forward(tv(x.tolist())))
        ref = self.gn_ref(x, G, None, None, 1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_forward_single_group(self):
        N, C, H, G = 1, 4, 3, 1
        x = rng.normal(size=(N, C, H))
        m = nn.GroupNorm(G, C)
        out = to_np(m.forward(tv(x.tolist())))
        ref = self.gn_ref(x, G, np.ones(C), np.zeros(C), 1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_bad_groups_raises(self):
        with pytest.raises(Exception):
            nn.GroupNorm(3, 4)

    def test_backward(self):
        N, C, H, G = 2, 4, 3, 2
        x_np = rng.normal(size=(N, C, H))
        eps = 1e-5
        x = tv(x_np.tolist(), requires_grad=True)
        m = nn.GroupNorm(G, C)
        out = m.forward(x)
        Ops.sum(out).backward()
        got = grad_np(x)
        ref = finite_diff_grad(
            lambda v: self.gn_ref(v.reshape(N, C, H), G, np.ones(C), np.zeros(C), eps).sum(), x_np
        )
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


class TestBatchNorm1d:
    def test_train_2d(self):
        N, C = 4, 3
        x = rng.normal(size=(N, C))
        m = nn.BatchNorm1d(C)
        m.train()
        out = to_np(m.forward(tv(x.tolist())))
        mu = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_train_3d(self):
        N, C, L = 4, 3, 5
        x = rng.normal(size=(N, C, L))
        m = nn.BatchNorm1d(C)
        m.train()
        out = to_np(m.forward(tv(x.tolist())))
        mu = x.mean(axis=(0, 2), keepdims=True)
        var = x.var(axis=(0, 2), keepdims=True)
        ref = (x - mu) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_eval_uses_running_stats(self):
        N, C = 4, 3
        x = rng.normal(size=(N, C))
        m = nn.BatchNorm1d(C)
        rm = rng.normal(size=(C,))
        rv = np.abs(rng.normal(size=(C,))) + 0.5
        m.running_mean = tv(rm.tolist())
        m.running_var = tv(rv.tolist())
        m.eval()
        out = to_np(m.forward(tv(x.tolist())))
        ref = (x - rm.reshape(1, C)) / np.sqrt(rv.reshape(1, C) + 1e-5)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_bad_shape_raises(self):
        m = nn.BatchNorm1d(3)
        with pytest.raises(Exception):
            m.forward(tv(rng.normal(size=(2, 3, 4, 5)).tolist()))


class TestConv1d:
    def conv_ref(self, x, w, b, stride, padding):
        N, Cin, L = x.shape
        Cout, _, K = w.shape
        xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        Lout = (L + 2 * padding - K) // stride + 1
        out = np.zeros((N, Cout, Lout))
        for n in range(N):
            for co in range(Cout):
                for lo in range(Lout):
                    s = 0.0
                    for ci in range(Cin):
                        for k in range(K):
                            s += xp[n, ci, lo * stride + k] * w[co, ci, k]
                    out[n, co, lo] = s + (b[co] if b is not None else 0.0)
        return out

    def test_forward(self):
        N, Cin, Cout, L, K = 2, 3, 4, 9, 3
        x = rng.normal(size=(N, Cin, L))
        w = rng.normal(size=(Cout, Cin, K))
        b = rng.normal(size=(Cout,))
        m = nn.Conv1d(Cin, Cout, K, stride=2, padding=1)
        m.weight = tv(w.tolist())
        m.bias = tv(b.tolist())
        out = to_np(m.forward(tv(x.tolist())))
        ref = self.conv_ref(x, w, b, 2, 1)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_forward_no_bias(self):
        N, Cin, Cout, L, K = 1, 2, 2, 7, 3
        x = rng.normal(size=(N, Cin, L))
        w = rng.normal(size=(Cout, Cin, K))
        m = nn.Conv1d(Cin, Cout, K, has_bias=False)
        m.weight = tv(w.tolist())
        out = to_np(m.forward(tv(x.tolist())))
        ref = self.conv_ref(x, w, None, 1, 0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_ops_conv1d(self):
        N, Cin, Cout, L, K = 2, 2, 3, 8, 3
        x = rng.normal(size=(N, Cin, L))
        w = rng.normal(size=(Cout, Cin, K))
        out = to_np(Ops.conv1d(tv(x.tolist()), tv(w.tolist())))
        ref = self.conv_ref(x, w, None, 1, 0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_backward(self):
        N, Cin, Cout, L, K = 1, 2, 2, 6, 3
        x_np = rng.normal(size=(N, Cin, L))
        w_np = rng.normal(size=(Cout, Cin, K))
        x = tv(x_np.tolist(), requires_grad=True)
        w = tv(w_np.tolist(), requires_grad=True)
        out = Ops.conv1d(x, w)
        Ops.sum(out).backward()
        got_x = grad_np(x)
        got_w = grad_np(w)
        ref_x = finite_diff_grad(lambda v: self.conv_ref(v.reshape(N, Cin, L), w_np, None, 1, 0).sum(), x_np)
        ref_w = finite_diff_grad(lambda v: self.conv_ref(x_np, v.reshape(Cout, Cin, K), None, 1, 0).sum(), w_np)
        np.testing.assert_allclose(got_x, ref_x, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(got_w, ref_w, rtol=1e-3, atol=1e-4)

    def test_bad_shapes_raise(self):
        with pytest.raises(Exception):
            Ops.conv1d(tv([1.0, 2.0]), tv([1.0]))


class TestAvgPool2d:
    def pool_ref(self, x, K, stride, padding):
        N, C, H, Wd = x.shape
        Ho = (H - K + 2 * padding) // stride + 1
        Wo = (Wd - K + 2 * padding) // stride + 1
        out = np.zeros((N, C, Ho, Wo))
        for n in range(N):
            for c in range(C):
                for ho in range(Ho):
                    for wo in range(Wo):
                        s = 0.0
                        for ky in range(K):
                            y = ho * stride - padding + ky
                            if y < 0 or y >= H:
                                continue
                            for kx in range(K):
                                xx = wo * stride - padding + kx
                                if xx < 0 or xx >= Wd:
                                    continue
                                s += x[n, c, y, xx]
                        out[n, c, ho, wo] = s / (K * K)
        return out

    def test_forward(self):
        N, C, H, Wd = 2, 3, 7, 6
        x = rng.normal(size=(N, C, H, Wd))
        m = nn.AvgPool2d(3, stride=2, padding=1)
        out = to_np(m.forward(tv(x.tolist())))
        ref = self.pool_ref(x, 3, 2, 1)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_ops_default_stride(self):
        x = rng.normal(size=(1, 1, 4, 4))
        out = to_np(Ops.avg_pool2d(tv(x.tolist()), 2))
        ref = self.pool_ref(x, 2, 2, 0)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_backward(self):
        N, C, H, Wd, K = 1, 2, 5, 5, 3
        x_np = rng.normal(size=(N, C, H, Wd))
        x = tv(x_np.tolist(), requires_grad=True)
        out = Ops.avg_pool2d(x, K, stride=1, padding=1)
        Ops.sum(out).backward()
        got = grad_np(x)
        ref = finite_diff_grad(lambda v: self.pool_ref(v.reshape(N, C, H, Wd), K, 1, 1).sum(), x_np)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


class TestNLLLoss:
    def nll_ref(self, logp, tgt, weight, ignore_index, reduction):
        N, C = logp.shape
        losses = np.zeros(N)
        wsum = 0.0
        for i in range(N):
            t = int(tgt[i])
            if t == ignore_index:
                continue
            w = weight[t] if weight is not None else 1.0
            losses[i] = -w * logp[i, t]
            wsum += w
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        return losses.sum() / wsum

    def test_mean(self):
        N, C = 4, 5
        logp = np.log(rng.dirichlet(np.ones(C), size=N))
        tgt = np.array([0, 2, 4, 1])
        m = nn.NLLLoss()
        out = to_np(m.forward(tv(logp.tolist()), tv(tgt.tolist())))
        ref = self.nll_ref(logp, tgt, None, -100, "mean")
        np.testing.assert_allclose(out.ravel()[0], ref, rtol=1e-5)

    def test_sum_none_weight_ignore(self):
        N, C = 4, 5
        logp = np.log(rng.dirichlet(np.ones(C), size=N))
        tgt = np.array([0, -100, 4, 1])
        w = np.abs(rng.normal(size=(C,))) + 0.5
        for reduction in ("sum", "none", "mean"):
            m = nn.NLLLoss(weight=tv(w.tolist()), ignore_index=-100, reduction=reduction)
            out = to_np(m.forward(tv(logp.tolist()), tv(tgt.tolist())))
            ref = self.nll_ref(logp, tgt, w, -100, reduction)
            np.testing.assert_allclose(out.ravel(), np.atleast_1d(ref), rtol=1e-5)

    def test_backward(self):
        N, C = 3, 4
        logp_np = np.log(rng.dirichlet(np.ones(C), size=N))
        tgt = np.array([0, 2, 1])
        x = tv(logp_np.tolist(), requires_grad=True)
        t = tv(tgt.tolist())
        nn.NLLLoss()(x, t).backward()
        got = grad_np(x)
        ref = finite_diff_grad(
            lambda v: self.nll_ref(v.reshape(N, C), tgt, None, -100, "mean"), logp_np
        )
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


class TestBCEWithLogitsLoss:
    def bce_ref(self, x, y, w, reduction, pos_weight):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = np.ones_like(x) if pos_weight is None else np.asarray(pos_weight, dtype=np.float64)
        if p.ndim == 1 and p.shape[0] == x.shape[-1] and x.ndim > 1:
            p = np.broadcast_to(p.reshape((1,) * (x.ndim - 1) + p.shape), x.shape)
        p = np.broadcast_to(p, x.shape).copy()
        ax = np.abs(x)
        max_neg = np.maximum(-x, 0.0)
        log_term = np.log1p(np.exp(-ax))
        pw = 1.0 + (p - 1.0) * y
        li = (1.0 - y) * x + pw * (max_neg + log_term)
        if w is not None:
            li = li * np.asarray(w, dtype=np.float64)
        if reduction == "none":
            return li
        if reduction == "sum":
            return li.sum()
        return li.mean()

    def test_forward_stable(self):
        N, C = 3, 4
        x = rng.normal(size=(N, C)) * 3.0
        y = (rng.random(size=(N, C)) > 0.5).astype(float)
        m = nn.BCEWithLogitsLoss()
        out = to_np(m.forward(tv(x.tolist()), tv(y.tolist())))
        ref = self.bce_ref(x, y, None, "mean", None)
        np.testing.assert_allclose(out.ravel()[0], ref, rtol=1e-5)

    def test_extreme_logits(self):
        x = np.array([[-50.0, 50.0, 0.0], [30.0, -30.0, 1.0]])
        y = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
        m = nn.BCEWithLogitsLoss()
        out = to_np(m.forward(tv(x.tolist()), tv(y.tolist())))
        ref = self.bce_ref(x, y, None, "mean", None)
        assert math.isfinite(float(out.ravel()[0]))
        np.testing.assert_allclose(out.ravel()[0], ref, rtol=1e-4)

    def test_pos_weight_and_weight(self):
        N, C = 3, 4
        x = rng.normal(size=(N, C))
        y = (rng.random(size=(N, C)) > 0.5).astype(float)
        pw = np.abs(rng.normal(size=(C,))) + 0.5
        w = np.abs(rng.normal(size=(N, C))) + 0.5
        m = nn.BCEWithLogitsLoss(weight=tv(w.tolist()), pos_weight=tv(pw.tolist()))
        out = to_np(m.forward(tv(x.tolist()), tv(y.tolist())))
        ref = self.bce_ref(x, y, w, "mean", pw)
        np.testing.assert_allclose(out.ravel()[0], ref, rtol=1e-5)

    def test_reductions(self):
        x = rng.normal(size=(2, 3))
        y = (rng.random(size=(2, 3)) > 0.5).astype(float)
        for reduction in ("mean", "sum", "none"):
            m = nn.BCEWithLogitsLoss(reduction=reduction)
            out = to_np(m.forward(tv(x.tolist()), tv(y.tolist())))
            ref = self.bce_ref(x, y, None, reduction, None)
            np.testing.assert_allclose(out.ravel(), np.atleast_1d(ref), rtol=1e-5)

    def test_backward(self):
        N, C = 2, 3
        x_np = rng.normal(size=(N, C))
        y_np = (rng.random(size=(N, C)) > 0.5).astype(float)
        x = tv(x_np.tolist(), requires_grad=True)
        y = tv(y_np.tolist())
        nn.BCEWithLogitsLoss()(x, y).backward()
        got = grad_np(x)
        ref = finite_diff_grad(lambda v: self.bce_ref(v.reshape(N, C), y_np, None, "mean", None), x_np)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


class TestCrossEntropyLabelSmoothing:
    def ce_ref(self, logits, tgt, eps, reduction, ignore_index=-100):
        N, C = logits.shape
        lse = np.logaddexp.reduce(logits, axis=1)
        nll = lse - logits[np.arange(N), tgt]
        smooth = lse - logits.mean(axis=1)
        losses = (1 - eps) * nll + eps * smooth
        mask = tgt != ignore_index
        losses = np.where(mask, losses, 0.0)
        if reduction == "none":
            return losses
        if reduction == "sum":
            return losses.sum()
        return losses.sum() / mask.sum()

    def test_no_smoothing_matches_plain(self):
        N, C = 4, 5
        logits = rng.normal(size=(N, C))
        tgt = np.array([0, 2, 4, 1])
        a = to_np(nn.CrossEntropyLoss()(tv(logits.tolist()), tv(tgt.tolist())))
        b = to_np(nn.CrossEntropyLoss(label_smoothing=0.0)(tv(logits.tolist()), tv(tgt.tolist())))
        np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_smoothing_forward(self):
        N, C = 4, 5
        logits = rng.normal(size=(N, C))
        tgt = np.array([0, 2, 4, 1])
        for reduction in ("mean", "sum", "none"):
            m = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            out = to_np(m.forward(tv(logits.tolist()), tv(tgt.tolist())))
            ref = self.ce_ref(logits, tgt, 0.1, reduction)
            np.testing.assert_allclose(out.ravel(), np.atleast_1d(ref), rtol=1e-5)

    def test_ops_level(self):
        N, C = 3, 4
        logits = rng.normal(size=(N, C))
        tgt = np.array([1, 0, 3])
        out = to_np(Ops.cross_entropy_loss(tv(logits.tolist()), tv(tgt.tolist()), label_smoothing=0.2))
        ref = self.ce_ref(logits, tgt, 0.2, "mean")
        np.testing.assert_allclose(out.ravel()[0], ref, rtol=1e-5)

    def test_invalid_smoothing_raises(self):
        with pytest.raises(Exception):
            nn.CrossEntropyLoss(label_smoothing=1.5)
        with pytest.raises(Exception):
            nn.CrossEntropyLoss(label_smoothing=-0.1)

    def test_backward(self):
        N, C = 3, 4
        logits_np = rng.normal(size=(N, C))
        tgt = np.array([1, 0, 3])
        x = tv(logits_np.tolist(), requires_grad=True)
        t = tv(tgt.tolist())
        nn.CrossEntropyLoss(label_smoothing=0.1)(x, t).backward()
        got = grad_np(x)
        ref = finite_diff_grad(lambda v: self.ce_ref(v.reshape(N, C), tgt, 0.1, "mean"), logits_np)
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


class TestSiLU:
    def test_forward(self):
        x = rng.normal(size=(2, 5))
        out = to_np(nn.SiLU().forward(tv(x.tolist())))
        ref = x / (1.0 + np.exp(-x))
        np.testing.assert_allclose(out, ref, rtol=1e-5)

    def test_backward(self):
        x_np = rng.normal(size=(2, 4))
        x = tv(x_np.tolist(), requires_grad=True)
        out = nn.SiLU().forward(x)
        Ops.sum(out).backward()
        got = grad_np(x)
        s = 1.0 / (1.0 + np.exp(-x_np))
        ref = s * (1.0 + x_np * (1.0 - s))
        np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-4)


class TestLogSoftmax:
    def test_forward(self):
        x = rng.normal(size=(3, 4))
        for dim in (-1, 0, 1):
            out = to_np(nn.LogSoftmax(dim=dim).forward(tv(x.tolist())))
            lse = np.logaddexp.reduce(x, axis=dim % 2, keepdims=True)
            ref = x - lse
            np.testing.assert_allclose(out, ref, rtol=1e-5)


class TestMHAMasks:
    def make_mha(self):
        return nn.MultiHeadAttention(8, 2, dropout=0.0, batch_first=True)

    def test_backward_compat_single_tensor(self):
        m = self.make_mha()
        m.eval()
        B, T, E = 2, 5, 8
        x = rng.normal(size=(B, T, E))
        out = m.forward(tv(x.tolist()))
        assert list(out.shape) == [B, T, E]

    def test_masked_positions_zero_weight(self):
        m = self.make_mha()
        m.eval()
        B, T, E = 2, 6, 8
        q = rng.normal(size=(B, T, E))
        k = rng.normal(size=(B, T, E))
        v = rng.normal(size=(B, T, E))
        attn_mask = np.zeros((T, T))
        attn_mask[0, 4] = 1.0
        attn_mask[1, 5] = 1.0
        attn_mask[2, 3] = 1.0
        kpm = np.zeros((B, T))
        kpm[0, 5] = 1.0
        kpm[1, 4] = 1.0
        kpm[1, 5] = 1.0
        out, weights = m.forward(
            tv(q.tolist()), tv(k.tolist()), tv(v.tolist()),
            attn_mask=tv(attn_mask.tolist()),
            key_padding_mask=tv(kpm.tolist()),
            need_weights=True,
        )
        w = to_np(weights)
        assert w.shape == (B, 2, T, T)
        np.testing.assert_allclose(w[0, :, :, 5], 0.0, atol=1e-6)
        np.testing.assert_allclose(w[1, :, :, 4:], 0.0, atol=1e-6)
        assert w[0, 0, 0, 4] < 1e-6
        assert w[1, 0, 2, 3] < 1e-6
        for b in range(B):
            for h in range(2):
                for i in range(T):
                    np.testing.assert_allclose(w[b, h, i].sum(), 1.0, rtol=1e-5)
        out_plain, _ = m.forward(
            tv(q.tolist()), tv(k.tolist()), tv(v.tolist()), need_weights=True
        )
        assert not np.allclose(to_np(out), to_np(out_plain))

    def test_mask_renormalizes_unmasked(self):
        m = self.make_mha()
        m.eval()
        B, T, E = 1, 5, 8
        q = rng.normal(size=(B, T, E))
        k = rng.normal(size=(B, T, E))
        v = rng.normal(size=(B, T, E))
        attn_mask = np.zeros((T, T))
        attn_mask[:, 3] = 1.0
        _, w_plain = m.forward(
            tv(q.tolist()), tv(k.tolist()), tv(v.tolist()), need_weights=True
        )
        _, w_masked = m.forward(
            tv(q.tolist()), tv(k.tolist()), tv(v.tolist()),
            attn_mask=tv(attn_mask.tolist()), need_weights=True,
        )
        wp = to_np(w_plain)[0, 0]
        wm = to_np(w_masked)[0, 0]
        assert np.all(wm[:, 3] < 1e-6)
        keep = [j for j in range(T) if j != 3]
        ratio = wm[:, keep] / wp[:, keep]
        for i in range(T):
            np.testing.assert_allclose(ratio[i], ratio[i, 0], rtol=1e-4)

    def test_3d_and_float_masks(self):
        m = self.make_mha()
        m.eval()
        B, T, E, H = 2, 4, 8, 2
        q = rng.normal(size=(B, T, E))
        k = rng.normal(size=(B, T, E))
        v = rng.normal(size=(B, T, E))
        bool_mask = np.zeros((T, T))
        bool_mask[0, 2] = 1.0
        float_mask = np.zeros((T, T))
        float_mask[0, 2] = -1e9
        mask3d = np.broadcast_to(bool_mask, (B * H, T, T)).copy()
        r1 = m.forward(tv(q.tolist()), tv(k.tolist()), tv(v.tolist()),
                       attn_mask=tv(bool_mask.tolist()), need_weights=True)
        r2 = m.forward(tv(q.tolist()), tv(k.tolist()), tv(v.tolist()),
                       attn_mask=tv(float_mask.tolist()), need_weights=True)
        r3 = m.forward(tv(q.tolist()), tv(k.tolist()), tv(v.tolist()),
                       attn_mask=tv(mask3d.tolist()), need_weights=True)
        np.testing.assert_allclose(to_np(r1[0]), to_np(r2[0]), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(to_np(r1[0]), to_np(r3[0]), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(to_np(r1[1]), to_np(r3[1]), rtol=1e-4, atol=1e-5)

    def test_bad_mask_shape_raises(self):
        m = self.make_mha()
        m.eval()
        B, T, E = 1, 4, 8
        q = tv(rng.normal(size=(B, T, E)).tolist())
        with pytest.raises(Exception):
            m.forward(q, q, q, attn_mask=tv([1.0, 2.0]), need_weights=True)
        with pytest.raises(Exception):
            m.forward(q, q, q, key_padding_mask=tv([1.0]), need_weights=True)

    def test_need_weights_false_returns_none(self):
        m = self.make_mha()
        m.eval()
        B, T, E = 1, 4, 8
        q = tv(rng.normal(size=(B, T, E)).tolist())
        out, weights = m.forward(q, q, q, need_weights=False)
        assert list(out.shape) == [B, T, E]
        assert weights is None
