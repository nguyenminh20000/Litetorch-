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

def allclose(vals, want, tol=1e-6):
    return all(abs(v - want) <= tol for v in vals)

def flat(t):
    return [float(v) for v in t.to_vector()]

dev1 = lt.Device(lt.DeviceType.CPU, 1)
print("module-to grad-contract, CPU:0 -> CPU:1")

model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
pre = model.parameters()
check("sequential parameters 2", len(pre) == 2)
w_before = flat(pre[0])
b_before = flat(pre[1])
model.to(dev1)
post = model.parameters()
check("identity preserved", len(post) == 2 and post[0] is pre[0] and post[1] is pre[1])
check("device moved", post[0].device.type == lt.DeviceType.CPU and post[0].device.index == 1 and post[1].device.type == lt.DeviceType.CPU and post[1].device.index == 1)
check("values preserved", flat(post[0]) == w_before and flat(post[1]) == b_before)

x = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
y = model.forward(x)
loss = y.sum()
loss.backward()
check("grad populated on weight", post[0].grad is not None)
check("grad populated on bias", post[1].grad is not None)
check("weight grad all 0.5", post[0].grad is not None and allclose(flat(post[0].grad), 0.5))
check("bias grad all 1.0", post[1].grad is not None and allclose(flat(post[1].grad), 1.0))

tm = torch.nn.Linear(4, 4)
tx = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
ty = tm(tx)
tloss = ty.sum()
tloss.backward()
check("torch parity weight grad", post[0].grad is not None and max(abs(a - b) for a, b in zip(flat(post[0].grad), [float(v) for v in tm.weight.grad.flatten().tolist()])) < 1e-6)
check("torch parity bias grad", post[1].grad is not None and max(abs(a - b) for a, b in zip(flat(post[1].grad), [float(v) for v in tm.bias.grad.tolist()])) < 1e-6)

lin = lt.nn.Linear(4, 4, True)
lpre = lin.parameters()
lin.to(dev1)
lpost = lin.parameters()
check("plain Linear identity preserved", len(lpost) == 2 and lpost[0] is lpre[0] and lpost[1] is lpre[1])
lx = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
lloss = lin.forward(lx).sum()
lloss.backward()
check("plain Linear grad populated", lpost[0].grad is not None and lpost[1].grad is not None)
check("plain Linear weight grad all 0.5", lpost[0].grad is not None and allclose(flat(lpost[0].grad), 0.5))
check("plain Linear bias grad all 1.0", lpost[1].grad is not None and allclose(flat(lpost[1].grad), 1.0))

m2 = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
opt = lt.optim.SGD(m2.parameters(), lr=0.01)
m2.to(dev1)
x2 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
loss2 = m2.forward(x2).sum()
loss2.backward()
p2 = m2.parameters()
w2_before = flat(p2[0])
b2_before = flat(p2[1])
opt.step()
w2_after = flat(p2[0])
b2_after = flat(p2[1])
check("optimizer-before-to steps weight", all(abs((a - b) - (-0.005)) < 1e-6 for a, b in zip(w2_after, w2_before)))
check("optimizer-before-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b2_after, b2_before)))

m3 = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
m3.to(dev1)
opt3 = lt.optim.SGD(m3.parameters(), lr=0.01)
x3 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
loss3 = m3.forward(x3).sum()
loss3.backward()
p3 = m3.parameters()
w3_before = flat(p3[0])
b3_before = flat(p3[1])
opt3.step()
w3_after = flat(p3[0])
b3_after = flat(p3[1])
check("optimizer-after-to steps weight", all(abs((a - b) - (-0.005)) < 1e-6 for a, b in zip(w3_after, w3_before)))
check("optimizer-after-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b3_after, b3_before)))

bag = lt.nn.EmbeddingBag(5, 4, "sum")
epre = bag.parameters()
check("embbag parameters 1", len(epre) == 1)
check("embbag baseline weight is member", len(epre) == 1 and epre[0] is bag.weight)
check("embbag baseline grad null", epre[0].grad is None)
check("embbag baseline device CPU:0", epre[0].device.type == lt.DeviceType.CPU and epre[0].device.index == 0)
ew_before = flat(epre[0])
bag.to(dev1)
epost = bag.parameters()
check("embbag parameters()[0] is member weight", len(epost) == 1 and epost[0] is bag.weight)
check("embbag device moved", epost[0].device.type == lt.DeviceType.CPU and epost[0].device.index == 1)
check("embbag values preserved", flat(epost[0]) == ew_before)

eidx = lt.tensor([0.0, 1.0, 2.0]).to(dev1)
eout = bag.forward(eidx)
ewant = [ew_before[0 * 4 + j] + ew_before[1 * 4 + j] + ew_before[2 * 4 + j] for j in range(4)]
check("embbag forward values exact row sums", all(abs(a - b) < 1e-6 for a, b in zip(flat(eout), ewant)))
check("embbag forward output device", eout.device.type == lt.DeviceType.CPU and eout.device.index == 1)

tbag = torch.nn.EmbeddingBag(5, 4, mode="sum")
tbag.weight.data = torch.tensor(ew_before).reshape(5, 4)
tout = tbag(torch.tensor([0, 1, 2]), torch.tensor([0]))
check("embbag torch parity forward values", max(abs(a - b) for a, b in zip(flat(eout), [float(v) for v in tout.flatten().tolist()])) < 1e-6)

eloss = epost[0].sum()
eloss.backward()
check("embbag grad populated", epost[0].grad is not None)
check("embbag grad all 1.0", epost[0].grad is not None and allclose(flat(epost[0].grad), 1.0))
check("embbag grad device moved", epost[0].grad is not None and epost[0].grad.device.type == lt.DeviceType.CPU and epost[0].grad.device.index == 1)

bag2 = lt.nn.EmbeddingBag(5, 4, "sum")
optb = lt.optim.SGD(bag2.parameters(), lr=0.01)
bag2.to(dev1)
out_before = flat(bag2.forward(eidx))
loss_b = bag2.parameters()[0].sum()
loss_b.backward()
optb.step()
out_after = flat(bag2.forward(eidx))
check("embbag optimizer-before-to steps forward output", all(abs((a - b) - (-0.03)) < 1e-6 for a, b in zip(out_after, out_before)))

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
