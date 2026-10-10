import sys
import litetorch as lt

dev = lt.Device("cuda:0")
m = lt.nn.Sequential([
    lt.nn.Linear(4, 4),
    lt.nn.ReLU(),
    lt.nn.Linear(4, 2),
])
m.to(dev)
params = m.parameters()
opt = lt.optim.Adam(params, lr=1e-3)

xt = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4] * 8, [8, 4], dev)
yt = lt.Tensor.from_vector([float(i % 2) for i in range(8)], [8], dev)

opt.zero_grad()
out = m.forward(xt)
loss = lt.Ops.cross_entropy_loss(out, yt)
loss.backward()
lt.cuda_synchronize()
for i, p in enumerate(params):
    g = p.grad
    status = "None" if g is None else "present"
    print(f"param{i}: grad {status}", flush=True)
opt.step()
print("done", flush=True)
