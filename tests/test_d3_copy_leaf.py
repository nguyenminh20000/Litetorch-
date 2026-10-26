import sys

import litetorch as lt

CPU = lt.Device("cpu")


def build_model():
    m = lt.nn.Sequential([
        lt.nn.Conv2d(3, 8, 3, padding=1), lt.nn.ReLU(),
        lt.nn.Flatten(),
        lt.nn.Linear(8 * 32 * 32, 10),
    ])
    m.to(CPU)
    return m


def build_linear():
    m = lt.nn.Sequential([lt.nn.Linear(4, 3), lt.nn.ReLU(), lt.nn.Linear(3, 2)])
    m.to(CPU)
    return m


def ones_source(shape):
    n = 1
    for d in shape:
        n *= d
    return lt.Tensor.from_vector([1.0] * n, list(shape), CPU)


def values_match(t, expected):
    got = t.to(CPU).to_vector()
    if len(got) != len(expected):
        return False
    return all(abs(a - b) < 1e-6 for a, b in zip(got, expected))


def test_leaf_copy_throws():
    m = build_model()
    p = m.parameters()[0]
    if not p.requires_grad:
        return False, "param requires_grad is False, expected True"
    try:
        p.copy_(ones_source(list(p.shape)))
    except RuntimeError as e:
        if "leaf Tensor that requires grad" in str(e):
            return True, ""
        return False, "wrong error: " + str(e)
    return False, "leaf copy_ did not throw"


def test_to_cpu_keeps_leaf():
    m = build_model()
    m.to(CPU)
    p = m.parameters()[0]
    try:
        p.copy_(ones_source(list(p.shape)))
    except RuntimeError as e:
        if "leaf Tensor that requires grad" in str(e):
            return True, ""
        return False, "wrong error: " + str(e)
    return False, "param copy_ after .to(cpu) did not throw, expected leaf"


def test_copy_init_no_grad():
    m = build_model()
    srcs = [ones_source(list(p.shape)) for p in m.parameters()]
    with lt.no_grad():
        for lp, src in zip(m.parameters(), srcs):
            lp.copy_(src)
    for p in m.parameters():
        n = 1
        for d in p.shape:
            n *= d
        if not values_match(p, [1.0] * n):
            return False, "copied values do not match source"
    lin = build_linear()
    with lt.no_grad():
        for lp in lin.parameters():
            lp.copy_(ones_source(list(lp.shape)))
    x = lt.Tensor.from_vector([0.5, -1.0, 2.0, 0.25], [1, 4], CPU, True)
    y = lin(x)
    y.sum().backward()
    for i, p in enumerate(lin.parameters()):
        if p.grad is None:
            return False, "param%d grad is None after backward" % i
    return True, ""


def test_load_state_dict_cpu():
    m1 = build_linear()
    m2 = build_linear()
    with lt.no_grad():
        for p in m1.parameters():
            p.copy_(lt.Tensor.from_vector([2.0] * p.numel(), list(p.shape), CPU))
    missing, unexpected = m2.load_state_dict(m1.state_dict())
    if missing or unexpected:
        return False, "missing=%s unexpected=%s" % (missing, unexpected)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if not values_match(p2, p1.to(CPU).to_vector()):
            return False, "load_state_dict values do not match source"
    return True, ""


TESTS = [
    ("leaf_copy_throws", test_leaf_copy_throws),
    ("to_cpu_keeps_leaf", test_to_cpu_keeps_leaf),
    ("copy_init_no_grad", test_copy_init_no_grad),
    ("load_state_dict_cpu", test_load_state_dict_cpu),
]


def main():
    failures = 0
    for name, fn in TESTS:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, "unexpected exception: %s: %s" % (type(e).__name__, e)
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else " :: " + msg))
        if not ok:
            failures += 1
    print("%d/%d passed" % (len(TESTS) - failures, len(TESTS)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
