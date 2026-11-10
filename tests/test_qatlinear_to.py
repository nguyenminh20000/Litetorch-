import sys

import litetorch as lt
import torch
import torch.nn.functional as F

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
print("qatlinear-to grad-contract, CPU:0 -> CPU:1")

lin = lt.nn.Linear(4, 4, True)
qat = lt.quantization.QATLinear(lin)
pre = qat.parameters()
check("baseline parameters 2", len(pre) == 2)
check("baseline weight is member", len(pre) == 2 and pre[0] is qat.weight and pre[1] is qat.bias)
check("baseline weight shared with source Linear", pre[0] is lin.weight and pre[1] is lin.bias)
check("baseline grad null", pre[0].grad is None and pre[1].grad is None)
check("baseline device CPU:0", pre[0].device.type == lt.DeviceType.CPU and pre[0].device.index == 0)
w_before = flat(pre[0])
b_before = flat(pre[1])
qat.to(dev1)
post = qat.parameters()
check("identity preserved", len(post) == 2 and post[0] is pre[0] and post[1] is pre[1])
check("device moved", post[0].device.type == lt.DeviceType.CPU and post[0].device.index == 1 and post[1].device.type == lt.DeviceType.CPU and post[1].device.index == 1)
check("values preserved", flat(post[0]) == w_before and flat(post[1]) == b_before)
check("source Linear still shares weight", lin.weight is qat.weight and lin.bias is qat.bias)
check("source Linear device moved", lin.weight.device.type == lt.DeviceType.CPU and lin.weight.device.index == 1)

qat.qat_enabled = False
x = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
y = qat.forward(x)
ywant = [sum(0.5 * w_before[j * 4 + k] for k in range(4)) + b_before[j] for j in range(4)]
check("plain forward values exact", all(abs(a - b) < 1e-6 for a, b in zip(flat(y), ywant)))
check("plain forward output device", y.device.type == lt.DeviceType.CPU and y.device.index == 1)
tm = torch.nn.Linear(4, 4)
tm.weight.data = torch.tensor(w_before).reshape(4, 4)
tm.bias.data = torch.tensor(b_before)
tx = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
ty = tm(tx)
check("plain torch parity forward values", max(abs(a - b) for a, b in zip(flat(y), [float(v) for v in ty.flatten().tolist()])) < 1e-6)
loss = y.sum()
loss.backward()
check("plain grad populated on weight", post[0].grad is not None)
check("plain grad populated on bias", post[1].grad is not None)
check("plain weight grad all 0.5", post[0].grad is not None and allclose(flat(post[0].grad), 0.5))
check("plain bias grad all 1.0", post[1].grad is not None and allclose(flat(post[1].grad), 1.0))
check("plain grad device moved", post[0].grad is not None and post[0].grad.device.type == lt.DeviceType.CPU and post[0].grad.device.index == 1)
tloss = ty.sum()
tloss.backward()
check("plain torch parity weight grad", post[0].grad is not None and max(abs(a - b) for a, b in zip(flat(post[0].grad), [float(v) for v in tm.weight.grad.flatten().tolist()])) < 1e-6)
check("plain torch parity bias grad", post[1].grad is not None and max(abs(a - b) for a, b in zip(flat(post[1].grad), [float(v) for v in tm.bias.grad.tolist()])) < 1e-6)

lin2 = lt.nn.Linear(4, 4, True)
qat2 = lt.quantization.QATLinear(lin2)
qat2.to(dev1)
p2 = qat2.parameters()
w2 = flat(p2[0])
b2 = flat(p2[1])
x2 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
y2 = qat2.forward(x2)
check("qat forward output device", y2.device.type == lt.DeviceType.CPU and y2.device.index == 1)
tw = torch.tensor([w2[i * 4:(i + 1) * 4] for i in range(4)], requires_grad=True)
tb = torch.tensor(b2, requires_grad=True)
tqw = torch.fake_quantize_per_tensor_affine(tw, 0.01, 0, -127, 127)
tqx = torch.fake_quantize_per_tensor_affine(tx, 0.01, 0, -127, 127)
tqout = F.linear(tqx, tqw, tb)
check("qat torch parity forward values", max(abs(a - b) for a, b in zip(flat(y2), [float(v) for v in tqout.flatten().tolist()])) < 1e-6)
loss2 = y2.sum()
loss2.backward()
check("qat grad populated on weight", p2[0].grad is not None)
check("qat grad populated on bias", p2[1].grad is not None)
check("qat weight grad all 0.5 (STE)", p2[0].grad is not None and allclose(flat(p2[0].grad), 0.5))
check("qat bias grad all 1.0", p2[1].grad is not None and allclose(flat(p2[1].grad), 1.0))
tqloss = tqout.sum()
tqloss.backward()
check("qat torch parity weight grad", p2[0].grad is not None and max(abs(a - b) for a, b in zip(flat(p2[0].grad), [float(v) for v in tw.grad.flatten().tolist()])) < 1e-6)
check("qat torch parity bias grad", p2[1].grad is not None and max(abs(a - b) for a, b in zip(flat(p2[1].grad), [float(v) for v in tb.grad.tolist()])) < 1e-6)

lin3 = lt.nn.Linear(4, 4, True)
qat3 = lt.quantization.QATLinear(lin3)
qat3.qat_enabled = False
opt = lt.optim.SGD(qat3.parameters(), lr=0.01)
qat3.to(dev1)
x3 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
loss3 = qat3.forward(x3).sum()
loss3.backward()
p3 = qat3.parameters()
w3_before = flat(p3[0])
b3_before = flat(p3[1])
opt.step()
w3_after = flat(p3[0])
b3_after = flat(p3[1])
check("optimizer-before-to steps weight", all(abs((a - b) - (-0.005)) < 1e-6 for a, b in zip(w3_after, w3_before)))
check("optimizer-before-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b3_after, b3_before)))

lin4 = lt.nn.Linear(4, 4, True)
qat4 = lt.quantization.QATLinear(lin4)
qat4.qat_enabled = False
qat4.to(dev1)
opt4 = lt.optim.SGD(qat4.parameters(), lr=0.01)
x4 = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
loss4 = qat4.forward(x4).sum()
loss4.backward()
p4 = qat4.parameters()
w4_before = flat(p4[0])
b4_before = flat(p4[1])
opt4.step()
w4_after = flat(p4[0])
b4_after = flat(p4[1])
check("optimizer-after-to steps weight", all(abs((a - b) - (-0.005)) < 1e-6 for a, b in zip(w4_after, w4_before)))
check("optimizer-after-to steps bias", all(abs((a - b) - (-0.01)) < 1e-6 for a, b in zip(b4_after, b4_before)))

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
