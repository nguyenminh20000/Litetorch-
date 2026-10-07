import math
import sys

sys.path.insert(0, "/home/hatch/workspace/Litetorch-")

import numpy as np

import litetorch as lt

try:
    import torch
    import torch.nn as tnn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

cpu = lt.Device("cpu")


def as_np(t):
    return np.array(t.to(cpu).to_vector(), dtype=np.float64).reshape(tuple(t.shape))


def kaiming_bound(fan_in, a=math.sqrt(5.0)):
    gain = math.sqrt(2.0 / (1.0 + a * a))
    return math.sqrt(3.0) * gain / math.sqrt(fan_in)


def check_uniform(name, w, fan_in):
    bound = kaiming_bound(fan_in)
    assert w.min() >= -bound - 1e-6, f"{name}: min {w.min()} < -bound {-bound}"
    assert w.max() <= bound + 1e-6, f"{name}: max {w.max()} > bound {bound}"
    assert abs(w.min() + bound) < 0.25 * bound, f"{name}: min {w.min()} far from -bound"
    assert abs(w.max() - bound) < 0.25 * bound, f"{name}: max {w.max()} far from +bound"
    exp_std = bound / math.sqrt(3.0)
    assert abs(w.std() - exp_std) < 0.15 * exp_std, f"{name}: std {w.std()} != {exp_std}"
    print(f"OK {name}: uniform(±{bound:.5f}) mean={w.mean():.5f} std={w.std():.5f}")


def check_bias_uniform(name, b, fan_in):
    bound = 1.0 / math.sqrt(fan_in)
    assert b.min() >= -bound - 1e-6 and b.max() <= bound + 1e-6, f"{name} out of ±{bound}"
    print(f"OK {name}: uniform(±{bound:.5f})")


def test_linear():
    for in_f, out_f in [(2048, 64), (64, 10), (128, 2)]:
        m = lt.nn.Linear(in_f, out_f)
        check_uniform(f"Linear({in_f},{out_f}).weight", as_np(m.weight), in_f)
        check_bias_uniform(f"Linear({in_f},{out_f}).bias", as_np(m.bias), in_f)


def test_conv2d():
    for ic, oc in [(3, 32), (32, 64), (1, 32)]:
        m = lt.nn.Conv2d(ic, oc, 3, padding=1)
        fan_in = ic * 3 * 3
        check_uniform(f"Conv2d({ic},{oc}).weight", as_np(m.weight), fan_in)
        check_bias_uniform(f"Conv2d({ic},{oc}).bias", as_np(m.bias), fan_in)


def test_vs_torch():
    if not HAS_TORCH:
        print("SKIP test_vs_torch: torch not installed")
        return
    torch.manual_seed(0)
    for shape, mk_lt, mk_t in [
        ((64, 2048), lambda: lt.nn.Linear(2048, 64), lambda: tnn.Linear(2048, 64)),
        ((32, 64, 3, 3), lambda: lt.nn.Conv2d(64, 32, 3, padding=1), lambda: tnn.Conv2d(64, 32, 3, padding=1)),
    ]:
        a, b = mk_lt(), mk_t()
        wa, wb = as_np(a.weight), b.weight.detach().numpy().reshape(shape)
        fa = 2048 if len(shape) == 2 else 64 * 9
        ba = 1.0 / math.sqrt(fa)
        for tag, w in [("lt", wa), ("torch", wb)]:
            assert w.min() >= -ba and w.max() <= ba, f"{tag} weight out of ±{ba}"
        print(f"OK vs torch {shape}: both uniform(±{ba:.5f}) "
              f"lt std={wa.std():.5f} torch std={wb.std():.5f}")


if __name__ == "__main__":
    test_linear()
    test_conv2d()
    test_vs_torch()
    print("INIT PARITY: ALL OK")
