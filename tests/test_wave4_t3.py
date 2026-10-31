import math
import random

import litetorch as lt
from litetorch import nn, Ops

CPU = lt.Device("cpu")

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"[FAIL] {name}")

def vec(t):
    return t.to(CPU).to_vector()

def t(data, shape, grad=False):
    flat = []
    def rec(d):
        if isinstance(d, (list, tuple)):
            for v in d: rec(v)
        else:
            flat.append(d)
    rec(data)
    return lt.Tensor.from_vector(flat, shape, CPU, grad)

def randt(shape, seed):
    random.seed(seed)
    n = 1
    for d in shape:
        n *= d
    vals = [random.uniform(-2.0, 2.0) for _ in range(n)]
    return lt.Tensor.from_vector(vals, list(shape), CPU, False)

def approx(a, b, tol=1e-4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))

# ============ Activations ============

def test_relu6():
    m = nn.ReLU6()
    x = t([-1.0, 0.5, 3.0, 7.0, 6.0], [5], False)
    y = vec(m.forward(x))
    check("relu6", approx(y, [0.0, 0.5, 3.0, 6.0, 6.0]))

def test_hardtanh():
    m = nn.Hardtanh(-1.0, 1.0)
    x = t([-2.0, -0.5, 0.5, 2.0], [4], False)
    y = vec(m.forward(x))
    check("hardtanh", approx(y, [-1.0, -0.5, 0.5, 1.0]))

def test_threshold():
    m = nn.Threshold(0.0, 0.0)
    x = t([-1.0, 0.5, -0.2, 1.0], [4], False)
    y = vec(m.forward(x))
    check("threshold", approx(y, [0.0, 0.5, 0.0, 1.0]))

def test_softshrink():
    m = nn.Softshrink(0.5)
    x = t([-1.0, -0.3, 0.0, 0.3, 1.0], [5], False)
    y = vec(m.forward(x))
    check("softshrink", approx(y, [-0.5, 0.0, 0.0, 0.0, 0.5]))

def test_hardshrink():
    m = nn.Hardshrink(0.5)
    x = t([-1.0, -0.3, 0.0, 0.3, 1.0], [5], False)
    y = vec(m.forward(x))
    check("hardshrink", approx(y, [-1.0, 0.0, 0.0, 0.0, 1.0]))

def test_tanhshrink():
    m = nn.Tanhshrink()
    x = t([0.0, 1.0, -1.0], [3], False)
    y = vec(m.forward(x))
    ref = [v - math.tanh(v) for v in [0.0, 1.0, -1.0]]
    check("tanhshrink", approx(y, ref))

def test_softsign():
    m = nn.Softsign()
    x = t([0.0, 1.0, -2.0], [3], False)
    y = vec(m.forward(x))
    ref = [v / (1 + abs(v)) for v in [0.0, 1.0, -2.0]]
    check("softsign", approx(y, ref))

def test_softplus():
    m = nn.Softplus(1.0, 20.0)
    x = t([0.0, 1.0, -1.0], [3], False)
    y = vec(m.forward(x))
    ref = [math.log(1 + math.exp(v)) for v in [0.0, 1.0, -1.0]]
    check("softplus", approx(y, ref))
    x2 = t([30.0], [1], False)
    y2 = vec(m.forward(x2))
    check("softplus_threshold", approx(y2, [30.0], tol=1e-3))

def test_mish():
    m = nn.Mish()
    x = t([0.0, 1.0, -1.0], [3], False)
    y = vec(m.forward(x))
    ref = [v * math.tanh(math.log(1 + math.exp(v))) for v in [0.0, 1.0, -1.0]]
    check("mish", approx(y, ref))

def test_celu():
    m = nn.CELU(1.0)
    x = t([1.0, -1.0, 0.0], [3], False)
    y = vec(m.forward(x))
    ref = [1.0, math.exp(-1.0) - 1.0, 0.0]
    check("celu", approx(y, ref))

def test_selu():
    m = nn.SELU()
    x = t([1.0, -1.0, 0.0], [3], False)
    y = vec(m.forward(x))
    scale, alpha = 1.0507009873554805, 1.6732632423543772
    ref = [scale * 1.0, scale * alpha * (math.exp(-1.0) - 1.0), 0.0]
    check("selu", approx(y, ref))

def test_rrelu_eval():
    m = nn.RReLU(0.125, 1.0/3.0)
    m.eval()
    x = t([1.0, -2.0], [2], False)
    y = vec(m.forward(x))
    slope = (0.125 + 1.0/3.0) / 2.0
    check("rrelu_eval", approx(y, [1.0, -2.0 * slope]))

# ============ Losses ============

def test_hinge_embedding():
    m = nn.HingeEmbeddingLoss(1.0, "none")
    x = t([0.5, 2.0, -0.5, -2.0], [4], False)
    y = t([1.0, 1.0, -1.0, -1.0], [4], False)
    out = vec(m.forward(x, y))
    check("hinge_embedding", approx(out, [0.5, 2.0, 1.5, 3.0]))

def test_cosine_embedding():
    m = nn.CosineEmbeddingLoss(0.0, "none")
    x1 = t([1.0, 0.0], [1, 2], False)
    x2 = t([1.0, 0.0], [1, 2], False)
    y = t([1.0], [1], False)
    out = vec(m.forward(x1, x2, y))
    check("cosine_embedding_pos", approx(out, [0.0]))
    y2 = t([-1.0], [1], False)
    out2 = vec(m.forward(x1, x2, y2))
    check("cosine_embedding_neg", approx(out2, [1.0]))

def test_margin_ranking():
    m = nn.MarginRankingLoss(0.0, "none")
    x1 = t([2.0], [1], False)
    x2 = t([1.0], [1], False)
    y = t([1.0], [1], False)
    out = vec(m.forward(x1, x2, y))
    check("margin_ranking", approx(out, [0.0]))

def test_soft_margin():
    m = nn.SoftMarginLoss("none")
    x = t([0.0], [1], False)
    y = t([1.0], [1], False)
    out = vec(m.forward(x, y))
    check("soft_margin", approx(out, [math.log(2.0)]))

def test_triplet():
    m = nn.TripletMarginLoss(1.0, 2, 1e-6, False, "none")
    a = t([0.0, 0.0], [1, 2], False)
    p = t([1.0, 0.0], [1, 2], False)
    n = t([0.0, 0.5], [1, 2], False)
    out = vec(m.forward(a, p, n))
    check("triplet", approx(out, [1.5]))

def test_triplet_dist():
    m = nn.TripletMarginWithDistanceLoss(1.0, "none")
    a = t([0.0, 0.0], [1, 2], False)
    p = t([3.0, 4.0], [1, 2], False)
    n = t([0.0, 1.0], [1, 2], False)
    out = vec(m.forward(a, p, n))
    check("triplet_dist", approx(out, [5.0]))

def test_poisson_nll():
    m = nn.PoissonNLLLoss(True, False, 1e-8, "none")
    x = t([0.0], [1], False)
    y = t([1.0], [1], False)
    out = vec(m.forward(x, y))
    check("poisson_nll", approx(out, [1.0]))

def test_gaussian_nll():
    m = nn.GaussianNLLLoss(False, 1e-6, "none")
    x = t([0.0], [1], False)
    y = t([0.0], [1], False)
    v = t([1.0], [1], False)
    out = vec(m.forward(x, y, v))
    check("gaussian_nll", approx(out, [0.0]))

def test_multi_margin():
    m = nn.MultiMarginLoss(1, 1.0, "none")
    x = t([3.0, 0.0, 0.0], [1, 3], False)
    y = t([0.0], [1], False)
    out = vec(m.forward(x, y))
    check("multi_margin", approx(out, [0.0], tol=1e-3))

def test_multilabel_softmargin():
    m = nn.MultiLabelSoftMarginLoss("none")
    x = t([[0.0, 0.0]], [1, 2], False)
    y = t([[1.0, 0.0]], [1, 2], False)
    out = vec(m.forward(x, y))
    check("multilabel_softmargin", approx(out, [math.log(2.0)], tol=1e-3))

# ============ Bilinear / Fold / Unfold ============

def test_bilinear():
    m = nn.Bilinear(4, 4, 2)
    x1 = randt([3, 4], 0)
    x2 = randt([3, 4], 1)
    out = m.forward(x1, x2)
    check("bilinear_shape", out.shape == [3, 2])

def test_unfold():
    m = nn.Unfold([2, 2])
    x = randt([1, 1, 3, 3], 2)
    out = m.forward(x)
    check("unfold_shape", out.shape == [1, 4, 4])

def test_fold():
    m = nn.Fold([3, 3], [2, 2])
    x = randt([1, 4, 4], 3)
    out = m.forward(x)
    check("fold_shape", out.shape == [1, 1, 3, 3])

def test_ops_bilinear():
    x1 = randt([2, 4], 4)
    x2 = randt([2, 4], 5)
    w = randt([3, 4, 4], 6)
    out = Ops.bilinear(x1, x2, w)
    check("ops_bilinear_shape", out.shape == [2, 3])

def test_fold_unfold_roundtrip():
    uf = nn.Unfold([2, 2], [1, 1], [0, 0], [2, 2])
    f = nn.Fold([4, 4], [2, 2], [1, 1], [0, 0], [2, 2])
    x = randt([1, 1, 4, 4], 7)
    patches = uf.forward(x)
    back = f.forward(patches)
    check("fold_unfold_roundtrip", approx(vec(back), vec(x), tol=1e-4))

# ============ run ============

test_relu6()
test_hardtanh()
test_threshold()
test_softshrink()
test_hardshrink()
test_tanhshrink()
test_softsign()
test_softplus()
test_mish()
test_celu()
test_selu()
test_rrelu_eval()
test_hinge_embedding()
test_cosine_embedding()
test_margin_ranking()
test_soft_margin()
test_triplet()
test_triplet_dist()
test_poisson_nll()
test_gaussian_nll()
test_multi_margin()
test_multilabel_softmargin()
test_bilinear()
test_unfold()
test_fold()
test_ops_bilinear()
test_fold_unfold_roundtrip()

print(f"\nW4-T3: {PASS} passed, {FAIL} failed")
if FAIL:
    raise SystemExit(1)
