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

def flat(t):
    return [float(v) for v in t.to_vector()]

def dequant_weights(rec):
    codes = [float(v) for v in rec.data.to_vector()]
    scales = [float(v) for v in rec.qparams.scales.to_vector()]
    rows = len(scales)
    cols = len(codes) // rows
    return [[codes[r * cols + c] * scales[r] for c in range(cols)] for r in range(rows)]

dev1 = lt.Device(lt.DeviceType.CPU, 1)
print("scheme-quantized-linear to() bias grad-contract, CPU:0 -> CPU:1")

w = [[0.5, -0.25, 0.75, 1.0], [1.5, -1.0, 0.25, 0.5], [-0.5, 0.5, 1.25, -0.75], [0.25, 0.75, -1.5, 2.0]]
b = [0.1, -0.2, 0.3, -0.4]
lin = lt.nn.Linear(4, 4, True)
lin.weight = lt.tensor(w)
lin.bias = lt.tensor(b)
model = lt.nn.Sequential([lin])
qmodel = lt.quantization.quantize(model, "int8-per-channel")
kids = dict(qmodel.named_children())
kid = kids["0"]
check("child is SchemeQuantizedLinear", "SchemeQuantizedLinear" in type(kid).__name__)
check("scheme is int8-per-channel", kid.scheme_name == "int8-per-channel")
pre = kid.parameters()
check("parameters 1", len(pre) == 1)
check("bias is member", len(pre) == 1 and pre[0] is kid.bias)
check("bias grad null", pre[0].grad is None)
check("bias device CPU:0", pre[0].device.type == lt.DeviceType.CPU and pre[0].device.index == 0)
check("record data device CPU:0", kid.record.data.device.type == lt.DeviceType.CPU and kid.record.data.device.index == 0)
b_before = flat(pre[0])
rec_before = [int(v) for v in kid.record.data.to_vector()]
x0 = lt.tensor([[0.5, 0.5, 0.5, 0.5]])
y0 = kid.forward(x0)
dq = dequant_weights(kid.record)
ref0 = F.linear(torch.tensor([[0.5, 0.5, 0.5, 0.5]]), torch.tensor(dq), torch.tensor(b))
check("forward matches torch on dequant weights", max(abs(a - c) for a, c in zip(flat(y0), [float(v) for v in ref0.flatten().tolist()])) < 1e-4)
kid.to(dev1)
post = kid.parameters()
check("bias identity preserved", len(post) == 1 and post[0] is pre[0])
check("bias is member post-to", len(post) == 1 and post[0] is kid.bias)
check("bias device moved", post[0].device.type == lt.DeviceType.CPU and post[0].device.index == 1)
check("bias values preserved", flat(post[0]) == b_before)
check("record data preserved", [int(v) for v in kid.record.data.to_vector()] == rec_before)
check("record data device moved", kid.record.data.device.type == lt.DeviceType.CPU and kid.record.data.device.index == 1)
x = lt.tensor([[0.5, 0.5, 0.5, 0.5]]).to(dev1)
y = kid.forward(x)
check("forward values preserved", flat(y) == flat(y0))
check("forward matches torch on dequant weights post-to", max(abs(a - c) for a, c in zip(flat(y), [float(v) for v in ref0.flatten().tolist()])) < 1e-4)
loss = y.sum()
loss.backward()
check("bias grad populated", post[0].grad is not None)
check("bias grad all 1.0", post[0].grad is not None and all(abs(v - 1.0) < 1e-6 for v in flat(post[0].grad)))
tm = torch.nn.Linear(4, 4)
tm.weight.data = torch.tensor(w)
tm.bias.data = torch.tensor(b)
tloss = tm(torch.tensor([[0.5, 0.5, 0.5, 0.5]])).sum()
tloss.backward()
check("torch parity bias grad", post[0].grad is not None and max(abs(a - c) for a, c in zip(flat(post[0].grad), [float(v) for v in tm.bias.grad.tolist()])) < 1e-6)

lin2 = lt.nn.Linear(4, 4, True)
lin2.weight = lt.tensor(w)
lin2.bias = lt.tensor(b)
q2 = lt.quantization.quantize(lt.nn.Sequential([lin2]), "int8-per-channel")
k2 = dict(q2.named_children())["0"]
opt = lt.optim.SGD(k2.parameters(), lr=0.01)
k2.to(dev1)
p2 = k2.parameters()
w2_before = flat(p2[0])
loss2 = k2.forward(x).sum()
loss2.backward()
opt.step()
w2_after = flat(p2[0])
check("optimizer-before-to steps bias", all(abs((a - c) - (-0.01)) < 1e-6 for a, c in zip(w2_after, w2_before)))

lin3 = lt.nn.Linear(4, 4, True)
lin3.weight = lt.tensor(w)
lin3.bias = lt.tensor(b)
q3 = lt.quantization.quantize(lt.nn.Sequential([lin3]), "int8-per-channel")
k3 = dict(q3.named_children())["0"]
k3.to(dev1)
opt3 = lt.optim.SGD(k3.parameters(), lr=0.01)
p3 = k3.parameters()
w3_before = flat(p3[0])
loss3 = k3.forward(x).sum()
loss3.backward()
opt3.step()
w3_after = flat(p3[0])
check("optimizer-after-to steps bias", all(abs((a - c) - (-0.01)) < 1e-6 for a, c in zip(w3_after, w3_before)))

lin4 = lt.nn.Linear(4, 4, True)
lin4.weight = lt.tensor(w)
lin4.bias = lt.tensor(b)
q4 = lt.quantization.quantize(lt.nn.Sequential([lin4]), "int8-per-channel")
k4 = dict(q4.named_children())["0"]
loss4 = k4.forward(x0).sum()
loss4.backward()
g4_pre = k4.parameters()[0].grad
check("existing grad populated pre-to", g4_pre is not None and all(abs(v - 1.0) < 1e-6 for v in flat(g4_pre)))
k4.to(dev1)
p4 = k4.parameters()
check("existing grad survives move", p4[0].grad is not None and all(abs(v - 1.0) < 1e-6 for v in flat(p4[0].grad)))
check("existing grad device moved", p4[0].grad is not None and p4[0].grad.device.type == lt.DeviceType.CPU and p4[0].grad.device.index == 1)

print("TOTAL %d passed, %d failed" % (passed, failed))
sys.exit(0 if failed == 0 else 1)
