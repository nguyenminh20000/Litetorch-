import sys

import litetorch as lt

DEV = lt.Device("cpu")

CHECKS = 0
FAILS = []


def check(name, cond, got=None, want=None):
    global CHECKS
    CHECKS += 1
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s: got %r want %r" % (name, got, want))
        FAILS.append(name)


def tv(data, shape, requires_grad=False):
    return lt.Tensor.from_vector(list(data), list(shape), DEV, requires_grad)


def expect_throw(name, fn, substr):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except RuntimeError as e:
        if substr in str(e):
            print("PASS %s" % name)
            return
        print("FAIL %s: wrong message %r" % (name, str(e)))
        FAILS.append(name)
        return
    except TypeError as e:
        print("FAIL %s: got TypeError %r" % (name, str(e)))
        FAILS.append(name)
        return
    print("FAIL %s: no throw" % name)
    FAILS.append(name)


def main():
    doc = lt.Tensor.backward.__doc__ or ""
    check("signature has retain_graph", "retain_graph" in doc, doc, "retain_graph in doc")
    check("signature default retain_graph=False", "retain_graph: bool = False" in doc, doc, "retain_graph: bool = False")
    check("signature default create_graph=False", "create_graph: bool = False" in doc, doc, "create_graph: bool = False")

    x = tv([2.0, 3.0], [2], True)
    y = lt.Ops.sum(x * x)
    y.backward()
    check("first backward grad", x.grad.tolist() == [4.0, 6.0], x.grad.tolist(), [4.0, 6.0])
    expect_throw("second backward without retain_graph throws",
                 lambda: y.backward(), "retain_graph=true")

    try:
        x = tv([2.0, 3.0], [2], True)
        y = lt.Ops.sum(x * x)
        y.backward(retain_graph=True)
        check("retain_graph first grad", x.grad.tolist() == [4.0, 6.0], x.grad.tolist(), [4.0, 6.0])
        y.backward()
        check("retain_graph second backward accumulates", x.grad.tolist() == [8.0, 12.0], x.grad.tolist(), [8.0, 12.0])
    except TypeError as e:
        check("retain_graph accepted", False, str(e)[:80], "no TypeError")

    try:
        x = tv([2.0, 3.0], [2], True)
        y = lt.Ops.sum(x * x)
        y.backward(retain_graph=False)
        expect_throw("retain_graph=False second backward throws",
                     lambda: y.backward(), "retain_graph=true")
    except TypeError as e:
        check("retain_graph=False accepted", False, str(e)[:80], "no TypeError")

    try:
        x = tv([2.0, 3.0], [2], True)
        y = lt.Ops.sum(x * x)
        y.backward(create_graph=True, retain_graph=True)
        check("create_graph+retain_graph first grad", x.grad.tolist() == [4.0, 6.0], x.grad.tolist(), [4.0, 6.0])
        y.backward()
        check("create_graph+retain_graph second accumulates", x.grad.tolist() == [8.0, 12.0], x.grad.tolist(), [8.0, 12.0])
    except TypeError as e:
        check("create_graph+retain_graph accepted", False, str(e)[:80], "no TypeError")

    import torch
    xt = torch.tensor([2.0, 3.0], requires_grad=True)
    yt = torch.sum(xt * xt)
    yt.backward(retain_graph=True)
    yt.backward()
    check("torch parity grad", xt.grad.tolist() == [8.0, 12.0], xt.grad.tolist(), [8.0, 12.0])

    print("%d FAILURES (%d checks)" % (len(FAILS), CHECKS) if FAILS else "ALL PASS (%d checks)" % CHECKS)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
