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
print("medusa-head to grad-contract, CPU:0 -> CPU:1")

head = lt.nn.MedusaHead(4, 4)
pre = head.parameters()
check("medusa parameters 2", len(pre) == 2)
check("baseline weight is member", len(pre) == 2 and pre[0] is head.weight and pre[1] is head.bias)
w_before = flat(pre[0])
b_before = flat(pre[1])
head.to(dev1)
post = head.parameters()
check("identity preserved", len(post) == 2 and post[0] is pre[0] and post[1] is pre[1])
check("post parameters are members", len(post) == 2 and post[0] is head.weight and post[1] is head.bias)
check("device moved", post[0].device.type == lt.DeviceType.CPU and post[0].device.index == 1 and post[1].device.type == lt.DeviceType.CPU and post[1].device.index == 1)
check("values preserved", flat(post[0]) == w_before and flat(post[1]) == b_before)

x = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
y = head.forward(x)
loss = y.sum()
loss.backward()
check("grad populated on weight", post[0].grad is not None)
check("grad populated on bias", post[1].grad is not None)
check("weight grad all 0.5", post[0].grad is not None and allclose(flat(post[0].grad), 0.5))
check("bias grad all 1.0", post[1].grad is not None and allclose(flat(post[1].grad), 1.0))

tm = torch.nn.Linear(4, 4)
tx = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
tloss = tm(tx).sum()
tloss.backward()
check("torch parity weight grad", post[0].grad is not None and max(abs(a - b) for a, b in zip(flat(post[0].grad), [float(v) for v in tm.weight.grad.flatten().tolist()])) < 1e-6)
check("torch parity bias grad", post[1].grad is not None and max(abs(a - b) for a, b in zip(flat(post[1].grad), [float(v) for v in tm.bias.grad.tolist()])) < 1e-6)

h2 = lt.nn.MedusaHead(4, 4)
opt = lt.optim.SGD(h2.parameters(), lr=0.01)
h2.to(dev1)
x2 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
loss2 = h2.forward(x2).sum()
loss2.backward()
p2 = h2.parameters()
w2_before = flat(p2[0])
b2_before = flat(p2[1])
opt.step()
w2_after = flat(p2[0])
b2_after = flat(p2[1])
check("optimizer-before-to steps weight", all(abs((a - b) - (-0.005)) < 1e-6 for a, b in zip(w2_after, w2_before)))
check("optimizer-before-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b2_after, b2_before)))

h3 = lt.nn.MedusaHead(4, 4)
h3.to(dev1)
opt3 = lt.optim.SGD(h3.parameters(), lr=0.01)
x3 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
loss3 = h3.forward(x3).sum()
loss3.backward()
p3 = h3.parameters()
w3_before = flat(p3[0])
b3_before = flat(p3[1])
opt3.step()
w3_after = flat(p3[0])
b3_after = flat(p3[1])
check("optimizer-after-to steps weight", all(abs((a - b) - (-0.005)) < 1e-6 for a, b in zip(w3_after, w3_before)))
check("optimizer-after-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b3_after, b3_before)))

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
