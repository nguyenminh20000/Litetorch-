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

def try_backward(loss):
    try:
        loss.backward()
        return True
    except Exception:
        return False

def torch_bag(mode, w):
    tbag = torch.nn.EmbeddingBag(5, 4, mode=mode)
    tbag.weight.data = torch.tensor(w).reshape(5, 4)
    tbag.weight.requires_grad_(True)
    return tbag

def torch_grads(tbag, idx, off):
    tout = tbag(torch.tensor(idx), torch.tensor(off))
    tout.sum().backward()
    return flat_torch(tout), [float(v) for v in tbag.weight.grad.flatten().tolist()]

def flat_torch(t):
    return [float(v) for v in t.detach().flatten().tolist()]

print("embeddingbag autograd, CPU:0")

bag = lt.nn.EmbeddingBag(5, 4, "sum")
pre = bag.parameters()
check("embbag parameters 1", len(pre) == 1)
check("embbag baseline weight is member", len(pre) == 1 and pre[0] is bag.weight)
check("embbag baseline grad null", pre[0].grad is None)
check("embbag baseline device CPU:0", pre[0].device.type == lt.DeviceType.CPU and pre[0].device.index == 0)
w = flat(pre[0])

idx = lt.tensor([0.0, 1.0, 2.0, 0.0])
out = bag.forward(idx)
want = [w[0 * 4 + j] + w[1 * 4 + j] + w[2 * 4 + j] + w[0 * 4 + j] for j in range(4)]
check("embbag forward values exact row sums", vecclose(flat(out), want))
tbag = torch_bag("sum", w)
tout, _ = torch_grads(tbag, [0, 1, 2, 0], [0])
check("embbag torch parity forward values", vecclose(flat(out), tout))
loss = out.sum()
check("embbag sum backward succeeds", try_backward(loss))
check("embbag sum grad populated", pre[0].grad is not None)
gwant = [0.0] * 20
for j in range(4):
    gwant[0 * 4 + j] = 2.0
    gwant[1 * 4 + j] = 1.0
    gwant[2 * 4 + j] = 1.0
check("embbag sum grad values are per-row counts", pre[0].grad is not None and vecclose(flat(pre[0].grad), gwant))
_, tgrad = torch_grads(torch_bag("sum", w), [0, 1, 2, 0], [0])
check("embbag torch parity sum grad", pre[0].grad is not None and vecclose(flat(pre[0].grad), tgrad))
check("embbag sum grad device", pre[0].grad is not None and pre[0].grad.device.type == lt.DeviceType.CPU and pre[0].grad.device.index == 0)

bagm = lt.nn.EmbeddingBag(5, 4, "mean")
pm = bagm.parameters()
wm = flat(pm[0])
outm = bagm.forward(idx)
lossm = outm.sum()
check("embbag mean backward succeeds", try_backward(lossm))
gmwant = [0.0] * 20
for j in range(4):
    gmwant[0 * 4 + j] = 0.5
    gmwant[1 * 4 + j] = 0.25
    gmwant[2 * 4 + j] = 0.25
check("embbag mean grad values counts over bag size", pm[0].grad is not None and vecclose(flat(pm[0].grad), gmwant))
_, tgradm = torch_grads(torch_bag("mean", wm), [0, 1, 2, 0], [0])
check("embbag torch parity mean grad", pm[0].grad is not None and vecclose(flat(pm[0].grad), tgradm))

bagx = lt.nn.EmbeddingBag(5, 4, "max")
px = bagx.parameters()
wx = flat(px[0])
outx = bagx.forward(idx)
lossx = outx.sum()
check("embbag max backward succeeds", try_backward(lossx))
gxwant = [0.0] * 20
for j in range(4):
    best = 0
    if wx[1 * 4 + j] > wx[best * 4 + j]:
        best = 1
    if wx[2 * 4 + j] > wx[best * 4 + j]:
        best = 2
    gxwant[best * 4 + j] = 1.0
check("embbag max grad only to argmax rows", px[0].grad is not None and vecclose(flat(px[0].grad), gxwant))
_, tgradx = torch_grads(torch_bag("max", wx), [0, 1, 2, 0], [0])
check("embbag torch parity max grad", px[0].grad is not None and vecclose(flat(px[0].grad), tgradx))

bag2 = lt.nn.EmbeddingBag(5, 4, "mean")
p2 = bag2.parameters()
w2 = flat(p2[0])
off2 = lt.tensor([0.0, 2.0])
out2 = bag2.forward(idx, off2)
want2 = [(w2[0 * 4 + j] + w2[1 * 4 + j]) / 2.0 for j in range(4)] + [(w2[2 * 4 + j] + w2[0 * 4 + j]) / 2.0 for j in range(4)]
check("embbag two-bag mean forward values", vecclose(flat(out2), want2))
tbag2 = torch_bag("mean", w2)
tout2, _ = torch_grads(tbag2, [0, 1, 2, 0], [0, 2])
check("embbag torch parity two-bag forward", vecclose(flat(out2), tout2))
loss2 = out2.sum()
check("embbag two-bag backward succeeds", try_backward(loss2))
g2want = [0.0] * 20
for j in range(4):
    g2want[0 * 4 + j] = 1.0
    g2want[1 * 4 + j] = 0.5
    g2want[2 * 4 + j] = 0.5
check("embbag two-bag mean grad values", p2[0].grad is not None and vecclose(flat(p2[0].grad), g2want))
_, tgrad2 = torch_grads(torch_bag("mean", w2), [0, 1, 2, 0], [0, 2])
check("embbag torch parity two-bag grad", p2[0].grad is not None and vecclose(flat(p2[0].grad), tgrad2))

bage = lt.nn.EmbeddingBag(5, 4, "sum")
pe = bage.parameters()
idxe = lt.tensor([0.0, 1.0])
offe = lt.tensor([0.0, 2.0, 2.0])
oute = bage.forward(idxe, offe)
we = flat(pe[0])
wante = [we[0 * 4 + j] + we[1 * 4 + j] for j in range(4)] + [0.0] * 8
check("embbag empty-bag forward values zero", vecclose(flat(oute), wante))
losse = oute.sum()
check("embbag empty-bag backward succeeds", try_backward(losse))
gewant = [0.0] * 20
for j in range(4):
    gewant[0 * 4 + j] = 1.0
    gewant[1 * 4 + j] = 1.0
check("embbag empty-bag grad values", pe[0].grad is not None and vecclose(flat(pe[0].grad), gewant))

bago = lt.nn.EmbeddingBag(5, 4, "sum")
opto = lt.optim.SGD(bago.parameters(), lr=0.01)
out_before = flat(bago.forward(idx))
loss_o = bago.forward(idx).sum()
ok_o = try_backward(loss_o)
if ok_o:
    opto.step()
out_after = flat(bago.forward(idx))
check("embbag optimizer steps forward output by -0.06", ok_o and all(abs((a - b) - (-0.06)) < 1e-6 for a, b in zip(out_after, out_before)))

dev1 = lt.Device(lt.DeviceType.CPU, 1)
bag1 = lt.nn.EmbeddingBag(5, 4, "sum")
bag1.to(dev1)
idx1 = lt.tensor([0.0, 1.0, 2.0, 0.0]).to(dev1)
out1 = bag1.forward(idx1)
loss1 = out1.sum()
check("embbag cpu1 backward succeeds", try_backward(loss1))
p1 = bag1.parameters()
check("embbag cpu1 leaf grad populated", p1[0].grad is not None)
check("embbag cpu1 grad device follows weight", p1[0].grad is not None and p1[0].grad.device.type == lt.DeviceType.CPU and p1[0].grad.device.index == 1)

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
