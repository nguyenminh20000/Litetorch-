import math

import litetorch as lt


DEV = lt.Device("cuda:0") if lt.is_gpu_available() else lt.Device("cpu")
RESULTS = []


def tv(data, shape, requires_grad=False):
    return lt.Tensor.from_vector(data, shape, DEV, requires_grad)


def vals(n, lo=0.1, hi=1.0):
    out = []
    s = 12345
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


def run(name, fn):
    try:
        fn()
        RESULTS.append((name, "PASS", ""))
    except AssertionError as e:
        RESULTS.append((name, "FAIL", str(e)))
    except Exception as e:
        RESULTS.append((name, "FAIL", type(e).__name__ + ": " + str(e)))


def unary_op(name, fn, lo=-1.0, hi=1.0):
    def t():
        x = tv(vals(8, lo, hi), [2, 4], True)
        lt.Ops.sum(fn(x)).backward()
        check(x, name)
    run("op_" + name, t)


def binary_op(name, fn, alo=-1.0, ahi=1.0, blo=-1.0, bhi=1.0):
    def t():
        a = tv(vals(8, alo, ahi), [2, 4], True)
        b = tv(vals(8, blo, bhi), [2, 4], True)
        lt.Ops.sum(fn(a, b)).backward()
        check(a, name + "_a")
        check(b, name + "_b")
    run("op_" + name, t)


def nn_activation(name, mod):
    def t():
        m = mod()
        m.to(DEV)
        x = tv(vals(8, -1.0, 1.0), [2, 4], True)
        lt.Ops.sum(m.forward(x)).backward()
        check(x, name)
    run("nn_" + name, t)


def test_activations():
    unary_op("relu", lt.Ops.relu)
    unary_op("sigmoid", lt.Ops.sigmoid)
    unary_op("tanh", lt.Ops.tanh)
    unary_op("gelu", lt.Ops.gelu)
    unary_op("leaky_relu", lt.Ops.leaky_relu)
    unary_op("softmax", lambda x: lt.Ops.softmax(x, -1))
    unary_op("silu_compose", lambda x: lt.Ops.mul(x, lt.Ops.sigmoid(x)))
    nn_activation("relu", lt.nn.ReLU)
    nn_activation("leaky_relu", lt.nn.LeakyReLU)
    nn_activation("sigmoid", lt.nn.Sigmoid)
    nn_activation("tanh", lt.nn.Tanh)
    nn_activation("gelu", lt.nn.GELU)
    nn_activation("softmax", lt.nn.Softmax)


def test_elementwise():
    binary_op("add", lt.Ops.add)
    binary_op("sub", lt.Ops.sub)
    binary_op("mul", lt.Ops.mul)
    binary_op("div", lt.Ops.div, blo=0.5, bhi=1.5)
    unary_op("pow", lambda x: lt.Ops.pow(x, 2.0), lo=0.1, hi=2.0)
    unary_op("sqrt", lt.Ops.sqrt, lo=0.5, hi=2.0)
    unary_op("exp", lt.Ops.exp, lo=-1.0, hi=1.0)
    unary_op("log", lt.Ops.log, lo=0.5, hi=2.0)


def test_matmul():
    def t_mm():
        a = tv(vals(12), [3, 4], True)
        b = tv(vals(20), [4, 5], True)
        lt.Ops.sum(lt.Ops.matmul(a, b)).backward()
        check(a, "matmul_a")
        check(b, "matmul_b")
    run("op_matmul", t_mm)

    def t_bmm():
        a = tv(vals(24), [2, 3, 4], True)
        b = tv(vals(40), [2, 4, 5], True)
        lt.Ops.sum(lt.Ops.bmm(a, b)).backward()
        check(a, "bmm_a")
        check(b, "bmm_b")
    run("op_bmm", t_bmm)


def test_linear():
    def t(bias):
        m = lt.nn.Linear(4, 3, bias)
        m.to(DEV)
        x = tv(vals(16), [4, 4], True)
        lt.Ops.sum(m.forward(x)).backward()
        for p in m.parameters():
            check(p, "linear")
        check(x, "linear_input")
    run("nn_linear_bias", lambda: t(True))
    run("nn_linear_nobias", lambda: t(False))


def test_conv2d():
    def t(has_bias):
        m = lt.nn.Conv2d(2, 3, 3, 1, 1, has_bias)
        m.to(DEV)
        x = tv(vals(50), [1, 2, 5, 5], True)
        lt.Ops.sum(m.forward(x)).backward()
        for p in m.parameters():
            check(p, "conv2d")
        check(x, "conv2d_input")
    run("nn_conv2d_bias", lambda: t(True))
    run("nn_conv2d_nobias", lambda: t(False))


def test_norm():
    def t_ln_ops():
        x = tv(vals(8), [2, 4], True)
        w = tv(vals(4), [4], True)
        b = tv(vals(4), [4], True)
        lt.Ops.sum(lt.Ops.layer_norm(x, [4], w, b)).backward()
        check(x, "layernorm_input")
        check(w, "layernorm_weight")
        check(b, "layernorm_bias")
    run("op_layernorm", t_ln_ops)

    def t_ln_nn():
        m = lt.nn.LayerNorm([4])
        m.to(DEV)
        x = tv(vals(8), [2, 4], True)
        lt.Ops.sum(m.forward(x)).backward()
        for p in m.parameters():
            check(p, "layernorm_nn")
        check(x, "layernorm_nn_input")
    run("nn_layernorm", t_ln_nn)

    def t_bn():
        m = lt.nn.BatchNorm2d(3)
        m.to(DEV)
        m.train()
        x = tv(vals(96), [2, 3, 4, 4], True)
        lt.Ops.sum(m.forward(x)).backward()
        for p in m.parameters():
            check(p, "batchnorm")
        check(x, "batchnorm_input")
    run("nn_batchnorm2d", t_bn)


def test_losses():
    def t_mse():
        a = tv(vals(8), [2, 4], True)
        b = tv(vals(8), [2, 4])
        lt.Ops.mse_loss(a, b).backward()
        check(a, "mse")
    run("op_mse_loss", t_mse)

    def t_ce():
        a = tv(vals(12), [4, 3], True)
        b = tv([0.0, 1.0, 2.0, 1.0], [4])
        lt.Ops.cross_entropy_loss(a, b).backward()
        check(a, "cross_entropy")
    run("op_cross_entropy", t_ce)

    def t_l1():
        a = tv(vals(8), [2, 4], True)
        b = tv(vals(8), [2, 4])
        lt.Ops.l1_loss(a, b).backward()
        check(a, "l1")
    run("op_l1_loss", t_l1)

    def t_bce():
        a = tv(vals(8, -1.0, 1.0), [2, 4], True)
        p = lt.Ops.sigmoid(a)
        b = tv([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [2, 4])
        lt.Ops.bce_loss(p, b).backward()
        check(a, "bce")
    run("op_bce_loss", t_bce)

    def t_nn_mse():
        m = lt.nn.MSELoss()
        a = tv(vals(8), [2, 4], True)
        b = tv(vals(8), [2, 4])
        m.forward(a, b).backward()
        check(a, "nn_mse")
    run("nn_mse_loss", t_nn_mse)

    def t_nn_ce():
        m = lt.nn.CrossEntropyLoss()
        a = tv(vals(12), [4, 3], True)
        b = tv([0.0, 1.0, 2.0, 1.0], [4])
        m.forward(a, b).backward()
        check(a, "nn_cross_entropy")
    run("nn_cross_entropy", t_nn_ce)

    def t_nn_l1():
        m = lt.nn.L1Loss()
        a = tv(vals(8), [2, 4], True)
        b = tv(vals(8), [2, 4])
        m.forward(a, b).backward()
        check(a, "nn_l1")
    run("nn_l1_loss", t_nn_l1)

    def t_nn_bce():
        m = lt.nn.BCELoss()
        a = tv(vals(8, -1.0, 1.0), [2, 4], True)
        p = lt.Ops.sigmoid(a)
        b = tv([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0], [2, 4])
        m.forward(p, b).backward()
        check(a, "nn_bce")
    run("nn_bce_loss", t_nn_bce)


def main():
    print("device: " + DEV.to_string(), flush=True)
    test_activations()
    test_elementwise()
    test_matmul()
    test_linear()
    test_conv2d()
    test_norm()
    test_losses()
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    for name, status, msg in RESULTS:
        line = status + " " + name
        if msg:
            line += " | " + msg
        print(line, flush=True)
    print("TOTAL: %d/%d passed" % (npass, len(RESULTS)), flush=True)


if __name__ == "__main__":
    main()
