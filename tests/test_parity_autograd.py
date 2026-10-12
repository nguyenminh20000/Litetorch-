import math

import numpy as np
import pytest

import litetorch as lt

DEV = lt.Device("cpu")
F64 = lt.DataType.FP64


def tv(data, shape, requires_grad=False, dtype=None):
    kw = {} if dtype is None else {"dtype": dtype}
    return lt.Tensor.from_vector(list(data), list(shape), DEV, requires_grad, **kw)


def close(a, b, tol=1e-5, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{msg}: shape {a.shape} vs {b.shape}"
    d = np.max(np.abs(a - b))
    assert d <= tol, f"{msg}: max diff {d} > {tol}"


def test_autograd_grad_basic():
    x = tv([1.0, 2.0, 3.0], [3], True)
    y = lt.Ops.sum(x * x + 2.0 * x + 1.0)
    (g,) = lt.autograd.grad(y, x)
    close(g.tolist(), [2 * 1.0 + 2, 2 * 2.0 + 2, 2 * 3.0 + 2], msg="d(x^2+2x+1)/dx")
    assert x.grad is None, "autograd.grad must not accumulate into .grad"


def test_autograd_grad_identity():
    x = tv([1.0, 2.0], [2], True)
    (g,) = lt.autograd.grad(x, x)
    close(g.tolist(), [1.0, 1.0], msg="grad(x, x) = ones")


def test_autograd_grad_multi_input():
    x1 = tv([2.0, 3.0], [2], True)
    x2 = tv([4.0, 5.0], [2], True)
    y = lt.Ops.sum(x1 * x2 + x1)
    g1, g2 = lt.autograd.grad(y, [x1, x2])
    close(g1.tolist(), [5.0, 6.0], msg="dy/dx1 = x2+1")
    close(g2.tolist(), [2.0, 3.0], msg="dy/dx2 = x1")


def test_autograd_grad_multi_output_shared():
    x = tv([2.0], [1], True)
    y1 = x * x
    y2 = 3.0 * x
    (g,) = lt.autograd.grad([y1, y2], [x])
    close(g.tolist(), [2 * 2.0 + 3.0], msg="summed contributions")


def test_autograd_grad_explicit_grad_outputs():
    x = tv([1.0, 2.0], [2], True)
    y = x * x
    go = tv([10.0, 100.0], [2])
    (g,) = lt.autograd.grad(y, x, grad_outputs=go)
    close(g.tolist(), [2 * 1.0 * 10.0, 2 * 2.0 * 100.0], msg="vjp with grad_outputs")


def test_autograd_grad_nonscalar_no_grad_outputs_raises():
    x = tv([1.0, 2.0], [2], True)
    y = x * x
    with pytest.raises(RuntimeError):
        lt.autograd.grad(y, x)


def test_autograd_grad_allow_unused():
    x = tv([2.0], [1], True)
    z = tv([5.0], [1], True)
    y = x * x
    (gx, gz) = lt.autograd.grad(y, [x, z], allow_unused=True, retain_graph=True)
    close(gx.tolist(), [4.0])
    assert gz is None
    with pytest.raises(RuntimeError):
        lt.autograd.grad(y, [x, z])


def test_autograd_grad_retain_graph_twice():
    x = tv([2.0, 3.0], [2], True)
    y = lt.Ops.sum(x * x)
    (g1,) = lt.autograd.grad(y, x, retain_graph=True)
    (g2,) = lt.autograd.grad(y, x)
    close(g1.tolist(), [4.0, 6.0])
    close(g2.tolist(), [4.0, 6.0])


def test_autograd_grad_create_graph_second_order():
    x = tv([1.5, 2.0], [2], True)
    y = lt.Ops.sum(x * x * x)
    (g1,) = lt.autograd.grad(y, x, create_graph=True)
    close(g1.tolist(), [3 * 1.5**2, 3 * 2.0**2], tol=1e-4, msg="first order")
    (g2,) = lt.autograd.grad(lt.Ops.sum(g1), x)
    close(g2.tolist(), [6 * 1.5, 6 * 2.0], tol=1e-4, msg="second order")


def test_retain_grad_api():
    x = tv([2.0], [1], True)
    h = x * 3.0
    h.retain_grad()
    y = h * h
    y.backward()
    assert h.grad is not None
    close(h.grad.tolist(), [2 * 6.0], msg="d(h^2)/dh at h=6")


def test_function_basic_cube():
    class Cube(lt.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x * x

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return 3.0 * x * x * grad_output

    x = tv([2.0, -1.5], [2], True)
    y = Cube.apply(x)
    close(y.tolist(), [8.0, -3.375], msg="forward x^3")
    lt.Ops.sum(y).backward()
    close(x.grad.tolist(), [3 * 4.0, 3 * 2.25], msg="backward 3x^2")


def test_function_nontensor_arg():
    class Scale(lt.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale):
            ctx.scale = scale
            return x * scale

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * ctx.scale, None

    x = tv([1.0, 2.0], [2], True)
    y = Scale.apply(x, 5.0)
    close(y.tolist(), [5.0, 10.0])
    lt.Ops.sum(y).backward()
    close(x.grad.tolist(), [5.0, 5.0])


def test_function_multi_output():
    class Split2(lt.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2.0, x * 3.0

        @staticmethod
        def backward(ctx, g1, g2):
            return g1 * 2.0 + g2 * 3.0

    x = tv([4.0], [1], True)
    a, b = Split2.apply(x)
    close(a.tolist(), [8.0])
    close(b.tolist(), [12.0])
    (lt.Ops.sum(a) + lt.Ops.sum(b)).backward()
    close(x.grad.tolist(), [2.0 + 3.0])


def test_function_needs_input_grad():
    seen = {}

    class Probe(lt.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            seen["needs"] = list(ctx.needs_input_grad)
            return x * 1.0

        @staticmethod
        def backward(ctx, g):
            return g

    x = tv([1.0], [1], True)
    Probe.apply(x)
    assert seen["needs"] == [True]


def test_function_missing_forward_raises():
    class Bad(lt.autograd.Function):
        @staticmethod
        def backward(ctx, g):
            return g

    x = tv([1.0], [1], True)
    with pytest.raises(RuntimeError):
        Bad.apply(x)


def test_gradcheck_pass():
    x = tv([1.5, -2.0, 0.5], [3], True, F64)

    def f(x):
        return lt.Ops.sum(x * x * 3.0 + lt.Ops.sin(x))

    assert lt.autograd.gradcheck(f, (x,), eps=1e-2, atol=1e-3, rtol=1e-2) is True


def test_gradcheck_multi_input():
    x = tv([1.5, 0.5], [2], True, F64)
    w = tv([2.0, -1.0], [2], True, F64)

    def f(x, w):
        return lt.Ops.sum(x * w + lt.Ops.exp(x))

    assert lt.autograd.gradcheck(f, (x, w), eps=1e-2, atol=1e-3, rtol=1e-2) is True


def test_gradcheck_detects_wrong_grad():
    class WrongSquare(lt.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * x

        @staticmethod
        def backward(ctx, g):
            return g * 3.0

    x = tv([2.0, -0.5], [2], True, F64)

    def f(x):
        return lt.Ops.sum(WrongSquare.apply(x))

    with pytest.raises(RuntimeError):
        lt.autograd.gradcheck(f, (x,), eps=1e-2, atol=1e-3, rtol=1e-2)


def test_gradcheck_raise_false_returns_false():
    class WrongSquare(lt.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * x

        @staticmethod
        def backward(ctx, g):
            return g * 3.0

    x = tv([2.0], [1], True, F64)

    def f(x):
        return lt.Ops.sum(WrongSquare.apply(x))

    assert lt.autograd.gradcheck(f, (x,), eps=1e-2, atol=1e-3, rtol=1e-2, raise_exception=False) is False


def test_gradcheck_rejects_non_float():
    x = lt.Tensor.from_vector([1.0, 2.0], [2], DEV, True, lt.DataType.INT32)
    with pytest.raises(RuntimeError):
        lt.autograd.gradcheck(lambda x: lt.Ops.sum(x), (x,))


def test_gradgradcheck_pass():
    x = tv([1.5, -2.0], [2], True, F64)

    def f(x):
        return lt.Ops.sum(x * x * x)

    assert lt.autograd.gradgradcheck(f, (x,), eps=1e-2, atol=1e-3, rtol=1e-2) is True
