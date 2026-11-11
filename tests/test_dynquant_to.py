import sys

import litetorch as lt
import torch
from litetorch.quantization import quantize_dynamic_linear

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
print("dynamic-quant-linear to() grad-contract, CPU:0 -> CPU:1")

lin = lt.nn.Linear(4, 4, True)
dlin = quantize_dynamic_linear(lin)
pre = dlin.parameters()
check("baseline parameters 1", len(pre) == 1)
check("baseline bias grad null", pre[0].grad is None)
check("baseline bias device CPU:0", pre[0].device.type == lt.DeviceType.CPU and pre[0].device.index == 0)
b_before = flat(pre[0])

x0 = lt.tensor([[0.5, 0.5, 0.5, 0.5]])
out_before = flat(dlin(x0))

dlin.to(dev1)
post = dlin.parameters()
check("bias identity preserved", len(post) == 1 and post[0] is pre[0])
check("bias device moved", post[0].device.type == lt.DeviceType.CPU and post[0].device.index == 1)
check("bias values preserved", flat(post[0]) == b_before)

x1 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
out1 = dlin(x1)
check("forward values unchanged by move", all(abs(a - b) < 1e-6 for a, b in zip(flat(out1), out_before)))
check("forward output device moved", out1.device.type == lt.DeviceType.CPU and out1.device.index == 1)

W = torch.tensor(flat(lin.weight)).reshape(4, 4)
scale = W.abs().max(dim=1, keepdim=True).values / 127.0
scale = torch.where(scale > 0, scale, torch.ones_like(scale))
Wdq = torch.clamp(torch.round(W / scale), -128, 127) * scale
tout = torch.tensor([[0.5, 0.5, 0.5, 0.5]]) @ Wdq.T + torch.tensor(flat(lin.bias))
check("forward torch parity replicated-quant", max(abs(a - b) for a, b in zip(flat(out1), [float(v) for v in tout.flatten().tolist()])) < 1e-6)

loss = out1.sum()
loss.backward()
check("bias grad populated", post[0].grad is not None)
check("bias grad all 1.0", post[0].grad is not None and allclose(flat(post[0].grad), 1.0))
check("bias grad device moved", post[0].grad is not None and post[0].grad.device.type == lt.DeviceType.CPU and post[0].grad.device.index == 1)

tm = torch.nn.Linear(4, 4)
tx = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
tloss = tm(tx).sum()
tloss.backward()
check("torch parity bias grad", post[0].grad is not None and max(abs(a - b) for a, b in zip(flat(post[0].grad), [float(v) for v in tm.bias.grad.tolist()])) < 1e-6)

dlin2 = quantize_dynamic_linear(lt.nn.Linear(4, 4, True))
opt = lt.optim.SGD(dlin2.parameters(), lr=0.01)
dlin2.to(dev1)
loss2 = dlin2(x1).sum()
loss2.backward()
p2 = dlin2.parameters()
b2_before = flat(p2[0])
opt.step()
b2_after = flat(p2[0])
check("optimizer-before-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b2_after, b2_before)))

dlin3 = quantize_dynamic_linear(lt.nn.Linear(4, 4, True))
dlin3.to(dev1)
opt3 = lt.optim.SGD(dlin3.parameters(), lr=0.01)
loss3 = dlin3(x1).sum()
loss3.backward()
p3 = dlin3.parameters()
b3_before = flat(p3[0])
opt3.step()
b3_after = flat(p3[0])
check("optimizer-after-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b3_after, b3_before)))

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
