import sys

import litetorch as lt
import torch

passed = 0
failed = 0

def check(name, ok):
    global passed, failed
    if ok:
        passed += 1
        print("  PASS " + name)
    else:
        failed += 1
        print("  FAIL " + name)

def vecclose(a, b, tol=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))

def flat(t):
    return [float(v) for v in t.to_vector()]

def forward_throws(bag, idx, off):
    try:
        bag.forward(idx, off)
        return False
    except Exception:
        return True

def torch_bag(mode, w):
    tbag = torch.nn.EmbeddingBag(5, 4, mode=mode)
    tbag.weight.data = torch.tensor(w).reshape(5, 4)
    return tbag

def set_weight(bag, w):
    src = lt.Tensor.from_vector([float(v) for v in w], list(bag.weight.shape), lt.Device(lt.DeviceType.CPU, 0))
    with lt.no_grad():
        bag.weight.copy_(src)

print("embeddingbag max-branch oob, CPU:0")

w = [1.0, 0.0, 0.0, 0.0,
     0.0, 2.0, 0.0, 0.0,
     0.0, 0.0, 3.0, 0.0,
     0.0, 0.0, 0.0, 4.0,
     0.5, 0.5, 0.5, 0.5]

bagx = lt.nn.EmbeddingBag(5, 4, "max")
set_weight(bagx, w)
idx = lt.tensor([0.0, 1.0, 2.0])
off0 = lt.tensor([0.0])
outx = bagx.forward(idx, off0)
check("max valid forward values exact", vecclose(flat(outx), [1.0, 2.0, 3.0, 0.0]))
tbagx = torch_bag("max", w)
toutx = tbagx(torch.tensor([0, 1, 2]), torch.tensor([0]))
check("torch parity max valid forward", vecclose(flat(outx), [float(v) for v in toutx.detach().flatten().tolist()]))
off2 = lt.tensor([0.0, 2.0])
out2 = bagx.forward(idx, off2)
check("max two-bag forward values", vecclose(flat(out2), [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]))
offe = lt.tensor([0.0, 2.0, 2.0])
idxe = lt.tensor([0.0, 1.0])
oute = bagx.forward(idxe, offe)
check("max empty-bag forward zeros", vecclose(flat(oute), [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

bags = lt.nn.EmbeddingBag(5, 4, "sum")
set_weight(bags, w)
outs = bags.forward(idx, off0)
check("sum valid forward values exact", vecclose(flat(outs), [1.0, 2.0, 3.0, 0.0]))
idxoob = lt.tensor([0.0, 5.0])
idxoob6 = lt.tensor([0.0, 6.0])
idxneg = lt.tensor([0.0, -1.0])
check("sum oob index 5 throws", forward_throws(bags, idxoob, off0))
check("sum negative index throws", forward_throws(bags, idxneg, off0))

bagm = lt.nn.EmbeddingBag(5, 4, "mean")
set_weight(bagm, w)
check("mean oob index 5 throws", forward_throws(bagm, idxoob, off0))
check("mean negative index throws", forward_throws(bagm, idxneg, off0))

check("max oob index 5 throws", forward_throws(bagx, idxoob, off0))
check("max oob index 6 throws", forward_throws(bagx, idxoob6, off0))
check("max negative index throws", forward_throws(bagx, idxneg, off0))

try:
    tbagx(torch.tensor([0, 5]), torch.tensor([0]))
    torch_max_oob_raises = False
except RuntimeError:
    torch_max_oob_raises = True
check("torch max oob raises RuntimeError", torch_max_oob_raises)
try:
    tbagx(torch.tensor([0, -1]), torch.tensor([0]))
    torch_max_neg_raises = False
except RuntimeError:
    torch_max_neg_raises = True
check("torch max negative raises RuntimeError", torch_max_neg_raises)

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
