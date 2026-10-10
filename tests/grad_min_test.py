import sys
import litetorch as lt

dev = lt.Device("cuda:0")
m = lt.nn.Sequential([
    lt.nn.Linear(4, 4),
    lt.nn.ReLU(),
    lt.nn.Linear(4, 2),
])
print(f"before .to: requires_grad={m.parameters()[0].requires_grad}", flush=True)
m.to(dev)
params = m.parameters()
print(f"after .to: requires_grad={params[0].requires_grad}", flush=True)

xt = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4] * 8, [8, 4], dev)
yt = lt.Tensor.from_vector([float(i % 2) for i in range(8)], [8], dev)
out = m.forward(xt)
print(f"out.requires_grad={out.requires_grad}", flush=True)
loss = lt.Ops.cross_entropy_loss(out, yt)
print(f"loss.requires_grad={loss.requires_grad}", flush=True)
loss.backward()
lt.cuda_synchronize()
for i, p in enumerate(params):
    print(f"param{i}: grad is {'None' if p.grad is None else 'present'}", flush=True)
print("done", flush=True)
