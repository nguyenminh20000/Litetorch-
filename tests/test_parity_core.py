import math
import numpy as np
import pytest

import litetorch
from litetorch import Ops


def tv(data, requires_grad=False):
    return litetorch.tensor(data, requires_grad=requires_grad)


def to_np(t):
    return np.array(t.to_vector(), dtype=np.float64).reshape(tuple(t.shape))


class TestFactories:
    def test_tensor_nested(self):
        t = litetorch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert list(t.shape) == [2, 2]
        np.testing.assert_allclose(to_np(t), [[1, 2], [3, 4]])

    def test_tensor_flat_and_scalar(self):
        t = litetorch.tensor([1.0, 2.0, 3.0])
        assert list(t.shape) == [3]
        t2 = litetorch.tensor(5.0)
        assert list(t2.shape) == [1]
        assert to_np(t2)[0] == 5.0

    def test_tensor_3d(self):
        t = litetorch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        assert list(t.shape) == [2, 2, 2]
        np.testing.assert_allclose(to_np(t).ravel(), np.arange(1, 9))

    def test_tensor_ragged_raises(self):
        with pytest.raises(Exception):
            litetorch.tensor([[1.0, 2.0], [3.0]])

    def test_empty(self):
        t = litetorch.empty((2, 3))
        assert list(t.shape) == [2, 3]
        t2 = litetorch.empty(4)
        assert list(t2.shape) == [4]

    def test_full(self):
        t = litetorch.full((2, 3), 7.5)
        np.testing.assert_allclose(to_np(t), np.full((2, 3), 7.5))

    def test_linspace(self):
        t = litetorch.linspace(0.0, 1.0, 5)
        np.testing.assert_allclose(to_np(t), np.linspace(0, 1, 5), rtol=1e-6)
        t1 = litetorch.linspace(3.0, 9.0, 1)
        assert to_np(t1)[0] == pytest.approx(3.0)

    def test_logspace(self):
        t = litetorch.logspace(0.0, 2.0, 3)
        np.testing.assert_allclose(to_np(t), np.logspace(0, 2, 3), rtol=1e-6)
        t2 = litetorch.logspace(0.0, 3.0, 4, base=2.0)
        np.testing.assert_allclose(to_np(t2), 2.0 ** np.linspace(0, 3, 4), rtol=1e-6)

    def test_eye(self):
        np.testing.assert_allclose(to_np(litetorch.eye(3)), np.eye(3))
        np.testing.assert_allclose(to_np(litetorch.eye(2, 4)), np.eye(2, 4))

    def test_rand(self):
        t = litetorch.rand((200, 200))
        v = to_np(t)
        assert v.min() >= 0.0 and v.max() < 1.0
        assert abs(v.mean() - 0.5) < 0.02

    def test_randint(self):
        t = litetorch.randint(3, 10, (50, 50))
        v = to_np(t)
        assert v.min() >= 3 and v.max() <= 9
        assert set(np.unique(v)).issubset(set(range(3, 10)))
        t2 = litetorch.randint(5, (20,))
        v2 = to_np(t2)
        assert v2.min() >= 0 and v2.max() <= 4

    def test_randperm(self):
        p = to_np(litetorch.randperm(10)).astype(int)
        assert sorted(p.tolist()) == list(range(10))

    def test_like_variants(self):
        a = litetorch.tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(to_np(litetorch.zeros_like(a)), np.zeros((2, 2)))
        np.testing.assert_allclose(to_np(litetorch.ones_like(a)), np.ones((2, 2)))
        e = litetorch.empty_like(a)
        assert list(e.shape) == [2, 2]
        np.testing.assert_allclose(to_np(litetorch.full_like(a, 2.5)), np.full((2, 2), 2.5))
        r = litetorch.rand_like(a)
        assert list(r.shape) == [2, 2]
        assert to_np(r).min() >= 0.0 and to_np(r).max() < 1.0
        rn = litetorch.randn_like(a)
        assert list(rn.shape) == [2, 2]
        assert abs(to_np(rn).mean()) < 0.3


class TestPointwise:
    def test_log2(self):
        x = np.array([1.0, 2.0, 4.0, 0.5])
        np.testing.assert_allclose(to_np(Ops.log2(tv(x.tolist()))), np.log2(x), rtol=1e-6)

    def test_log10(self):
        x = np.array([1.0, 10.0, 100.0, 0.1])
        np.testing.assert_allclose(to_np(Ops.log10(tv(x.tolist()))), np.log10(x), rtol=1e-6)

    def test_rsqrt(self):
        x = np.array([1.0, 4.0, 9.0, 0.25])
        np.testing.assert_allclose(to_np(Ops.rsqrt(tv(x.tolist()))), 1.0 / np.sqrt(x), rtol=1e-6)

    def test_tan(self):
        x = np.array([0.0, 0.5, -0.3, 1.0])
        np.testing.assert_allclose(to_np(Ops.tan(tv(x.tolist()))), np.tan(x), rtol=1e-6)

    def test_reciprocal(self):
        x = np.array([1.0, 2.0, -4.0, 0.5])
        np.testing.assert_allclose(to_np(Ops.reciprocal(tv(x.tolist()))), 1.0 / x, rtol=1e-6)

    def test_sign(self):
        x = [-3.0, 0.0, 2.5]
        np.testing.assert_allclose(to_np(Ops.sign(tv(x))), np.sign(x))

    def test_floor_ceil_trunc(self):
        x = [-2.7, -2.2, 2.2, 2.7]
        np.testing.assert_allclose(to_np(Ops.floor(tv(x))), np.floor(x))
        np.testing.assert_allclose(to_np(Ops.ceil(tv(x))), np.ceil(x))
        np.testing.assert_allclose(to_np(Ops.trunc(tv(x))), np.trunc(x))

    def test_round(self):
        x = [-2.7, -2.2, 2.2, 2.7, 1.4]
        np.testing.assert_allclose(to_np(Ops.round(tv(x))), np.round(x))

    def test_grad_log2(self):
        x = tv([1.0, 2.0, 4.0], requires_grad=True)
        Ops.log2(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), 1.0 / (np.array([1.0, 2.0, 4.0]) * math.log(2)), rtol=1e-5)

    def test_grad_log10(self):
        x = tv([1.0, 2.0, 10.0], requires_grad=True)
        Ops.log10(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), 1.0 / (np.array([1.0, 2.0, 10.0]) * math.log(10)), rtol=1e-5)

    def test_grad_rsqrt(self):
        xv = np.array([1.0, 4.0, 9.0])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.rsqrt(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), -0.5 * xv ** -1.5, rtol=1e-5)

    def test_grad_tan(self):
        xv = np.array([0.1, 0.5, -0.3])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.tan(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), 1.0 + np.tan(xv) ** 2, rtol=1e-5)

    def test_grad_reciprocal(self):
        xv = np.array([1.0, 2.0, -4.0])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.reciprocal(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), -1.0 / xv ** 2, rtol=1e-5)


class TestClamp:
    def test_clamp_both(self):
        x = [-2.0, 0.5, 3.0]
        np.testing.assert_allclose(to_np(Ops.clamp(tv(x), 0.0, 1.0)), np.clip(x, 0.0, 1.0))

    def test_clamp_min_only(self):
        x = [-2.0, 0.5, 3.0]
        np.testing.assert_allclose(to_np(Ops.clamp(tv(x), min=0.0)), np.clip(x, 0.0, None))
        np.testing.assert_allclose(to_np(Ops.clamp_min(tv(x), 0.0)), np.clip(x, 0.0, None))

    def test_clamp_max_only(self):
        x = [-2.0, 0.5, 3.0]
        np.testing.assert_allclose(to_np(Ops.clamp(tv(x), max=1.0)), np.clip(x, None, 1.0))
        np.testing.assert_allclose(to_np(Ops.clamp_max(tv(x), 1.0)), np.clip(x, None, 1.0))

    def test_clamp_min_grad(self):
        xv = np.array([-1.0, 0.5, 2.0])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.clamp_min(x, 0.0).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), [0.0, 1.0, 1.0])

    def test_clamp_max_grad(self):
        xv = np.array([-1.0, 0.5, 2.0])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.clamp_max(x, 1.0).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), [1.0, 1.0, 0.0])


class TestReductions:
    def test_min_full(self):
        x = [[3.0, 1.0], [4.0, 2.0]]
        assert to_np(Ops.min(tv(x)))[0] == pytest.approx(1.0)

    def test_amax_int_dim(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.amax(tv(x.tolist()), 1)), x.max(axis=1), rtol=1e-6)

    def test_amax_tuple_dims(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.amax(tv(x.tolist()), (0, 2))), x.max(axis=(0, 2)), rtol=1e-6)

    def test_amax_keepdim(self):
        x = np.arange(12.0).reshape(3, 4)
        r = Ops.amax(tv(x.tolist()), (0, 1), keepdim=True)
        assert list(r.shape) == [1, 1]
        assert to_np(r)[0, 0] == pytest.approx(11.0)

    def test_amin(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.amin(tv(x.tolist()), 1)), x.min(axis=1), rtol=1e-6)
        np.testing.assert_allclose(to_np(Ops.amin(tv(x.tolist()), (0, 2))), x.min(axis=(0, 2)), rtol=1e-6)

    def test_sum_tuple_dims(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.sum(tv(x.tolist()), (0, 2))), x.sum(axis=(0, 2)), rtol=1e-5)
        r = Ops.sum(tv(x.tolist()), (1, 2), keepdim=True)
        assert list(r.shape) == [2, 1, 1]

    def test_sum_int_dim_still_works(self):
        x = np.arange(12.0).reshape(3, 4)
        np.testing.assert_allclose(to_np(Ops.sum(tv(x.tolist()), 1)), x.sum(axis=1), rtol=1e-5)

    def test_mean_tuple_dims(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.mean(tv(x.tolist()), (0, 2))), x.mean(axis=(0, 2)), rtol=1e-5)

    def test_max_tuple_dims(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.max(tv(x.tolist()), (0, 2))), x.max(axis=(0, 2)), rtol=1e-6)

    def test_amax_grad(self):
        xv = np.array([[1.0, 5.0], [3.0, 2.0]])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.amax(x, 1).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), [[0, 1], [1, 0]])

    def test_sum_tupledim_grad(self):
        xv = np.arange(6.0).reshape(2, 3)
        x = tv(xv.tolist(), requires_grad=True)
        Ops.sum(x, (0, 1)).backward()
        np.testing.assert_allclose(to_np(x.grad), np.ones((2, 3)))


class TestTriuTril:
    def test_triu(self):
        x = np.arange(1, 10.0).reshape(3, 3)
        np.testing.assert_allclose(to_np(Ops.triu(tv(x.tolist()))), np.triu(x))

    def test_tril(self):
        x = np.arange(1, 10.0).reshape(3, 3)
        np.testing.assert_allclose(to_np(Ops.tril(tv(x.tolist()))), np.tril(x))

    def test_triu_diagonal(self):
        x = np.arange(1, 10.0).reshape(3, 3)
        np.testing.assert_allclose(to_np(Ops.triu(tv(x.tolist()), 1)), np.triu(x, 1))
        np.testing.assert_allclose(to_np(Ops.tril(tv(x.tolist()), -1)), np.tril(x, -1))

    def test_triu_batched(self):
        x = np.arange(1, 19.0).reshape(2, 3, 3)
        np.testing.assert_allclose(to_np(Ops.triu(tv(x.tolist()))), np.triu(x))

    def test_triu_grad(self):
        xv = np.arange(1, 10.0).reshape(3, 3)
        x = tv(xv.tolist(), requires_grad=True)
        Ops.triu(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), np.triu(np.ones((3, 3))))


class TestSort:
    def test_sort_asc(self):
        x = [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
        vals, idx = Ops.sort(tv(x))
        np.testing.assert_allclose(to_np(vals), np.sort(x, axis=-1))
        np.testing.assert_allclose(to_np(idx), np.argsort(x, axis=-1))

    def test_sort_desc(self):
        x = [[3.0, 1.0, 2.0]]
        vals, idx = Ops.sort(tv(x), dim=-1, descending=True)
        np.testing.assert_allclose(to_np(vals), np.sort(x, axis=-1)[..., ::-1])

    def test_sort_dim0(self):
        x = [[3.0, 1.0], [1.0, 4.0], [2.0, 2.0]]
        vals, idx = Ops.sort(tv(x), dim=0)
        np.testing.assert_allclose(to_np(vals), np.sort(x, axis=0))
        np.testing.assert_allclose(to_np(idx), np.argsort(x, axis=0))

    def test_argsort(self):
        x = [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
        np.testing.assert_allclose(to_np(Ops.argsort(tv(x))), np.argsort(x, axis=-1))

    def test_sort_grad(self):
        xv = np.array([[3.0, 1.0, 2.0]])
        x = tv(xv.tolist(), requires_grad=True)
        vals, _ = Ops.sort(x)
        vals.sum().backward()
        np.testing.assert_allclose(to_np(x.grad), np.ones((1, 3)))


class TestClose:
    def test_allclose_true(self):
        a = tv([1.0, 2.0, 3.0])
        b = tv([1.0 + 1e-7, 2.0 - 1e-7, 3.0])
        assert Ops.allclose(a, b) is True

    def test_allclose_false(self):
        assert Ops.allclose(tv([1.0]), tv([1.1])) is False

    def test_allclose_equal_nan(self):
        a = tv([float("nan"), 1.0])
        b = tv([float("nan"), 1.0])
        assert Ops.allclose(a, b, equal_nan=True) is True
        assert Ops.allclose(a, b) is False

    def test_allclose_broadcast(self):
        a = tv([[1.0, 2.0, 3.0]])
        b = tv([1.0, 2.0, 3.0])
        assert Ops.allclose(a, b) is True

    def test_isclose(self):
        a = tv([1.0, 2.0, 3.0])
        b = tv([1.0, 2.5, 3.0 + 1e-9])
        np.testing.assert_allclose(to_np(Ops.isclose(a, b)), [1.0, 0.0, 1.0])


class TestCumsum:
    def test_cumsum_dim1(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(to_np(Ops.cumsum(tv(x.tolist()), 1)), np.cumsum(x, axis=1))

    def test_cumsum_dim0_neg(self):
        x = np.array([[1.0, 2.0], [4.0, 5.0]])
        np.testing.assert_allclose(to_np(Ops.cumsum(tv(x.tolist()), -2)), np.cumsum(x, axis=0))

    def test_cumsum_grad(self):
        xv = np.array([[1.0, 2.0, 3.0]])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.cumsum(x, 1).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), [[3.0, 2.0, 1.0]])


class TestScatter:
    def test_scatter(self):
        inp = np.zeros((2, 4))
        index = np.array([[0, 2], [1, 3]])
        src = np.array([[10.0, 20.0], [30.0, 40.0]])
        out = Ops.scatter(tv(inp.tolist()), 1, tv(index.tolist()), tv(src.tolist()))
        expected = np.zeros((2, 4))
        expected[0, 0] = 10.0
        expected[0, 2] = 20.0
        expected[1, 1] = 30.0
        expected[1, 3] = 40.0
        np.testing.assert_allclose(to_np(out), expected)

    def test_scatter_add(self):
        inp = np.ones((2, 4))
        index = np.array([[0, 0], [1, 3]])
        src = np.array([[10.0, 20.0], [30.0, 40.0]])
        out = Ops.scatter_add(tv(inp.tolist()), 1, tv(index.tolist()), tv(src.tolist()))
        expected = np.ones((2, 4))
        expected[0, 0] += 30.0
        expected[1, 1] += 30.0
        expected[1, 3] += 40.0
        np.testing.assert_allclose(to_np(out), expected)

    def test_scatter_add_grad(self):
        inp = tv([[1.0, 2.0, 3.0]], requires_grad=True)
        src = tv([[10.0, 20.0]], requires_grad=True)
        idx = tv([[2, 0]])
        out = Ops.scatter_add(inp, 1, idx, src)
        out.sum().backward()
        np.testing.assert_allclose(to_np(inp.grad), [[1.0, 1.0, 1.0]])
        np.testing.assert_allclose(to_np(src.grad), [[1.0, 1.0]])

    def test_scatter_grad(self):
        inp = tv([[1.0, 2.0, 3.0]], requires_grad=True)
        src = tv([[10.0, 20.0]], requires_grad=True)
        idx = tv([[2, 0]])
        out = Ops.scatter(inp, 1, idx, src)
        out.sum().backward()
        np.testing.assert_allclose(to_np(inp.grad), [[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(to_np(src.grad), [[1.0, 1.0]])


class TestShapeOps:
    def test_flatten(self):
        x = np.arange(24.0).reshape(2, 3, 4)
        np.testing.assert_allclose(to_np(Ops.flatten(tv(x.tolist()))), x.reshape(-1))
        r = Ops.flatten(tv(x.tolist()), 1, 2)
        assert list(r.shape) == [2, 12]
        r2 = Ops.flatten(tv(x.tolist()), 0, 1)
        assert list(r2.shape) == [6, 4]

    def test_flatten_grad(self):
        xv = np.arange(6.0).reshape(2, 3)
        x = tv(xv.tolist(), requires_grad=True)
        Ops.flatten(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), np.ones((2, 3)))

    def test_repeat(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(to_np(Ops.repeat(tv(x.tolist()), [2, 3])), np.tile(x, (2, 3)))

    def test_repeat_grad(self):
        xv = np.array([[1.0, 2.0]])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.repeat(x, [3, 2]).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), [[6.0, 6.0]])

    def test_narrow(self):
        x = np.arange(12.0).reshape(3, 4)
        r = Ops.narrow(tv(x.tolist()), 1, 1, 2)
        assert list(r.shape) == [3, 2]
        np.testing.assert_allclose(to_np(r), x[:, 1:3])

    def test_narrow_grad(self):
        xv = np.arange(12.0).reshape(3, 4)
        x = tv(xv.tolist(), requires_grad=True)
        Ops.narrow(x, 0, 1, 2).sum().backward()
        expected = np.zeros((3, 4))
        expected[1:3, :] = 1.0
        np.testing.assert_allclose(to_np(x.grad), expected)

    def test_t(self):
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = Ops.t(tv(x.tolist()))
        assert list(r.shape) == [3, 2]
        np.testing.assert_allclose(to_np(r), x.T)
        v = tv([1.0, 2.0])
        assert list(Ops.t(v).shape) == [2]

    def test_t_grad(self):
        xv = np.array([[1.0, 2.0], [3.0, 4.0]])
        x = tv(xv.tolist(), requires_grad=True)
        Ops.t(x).sum().backward()
        np.testing.assert_allclose(to_np(x.grad), np.ones((2, 2)))
