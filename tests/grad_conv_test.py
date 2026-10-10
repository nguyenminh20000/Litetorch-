import litetorch as lt

dev = lt.Device("cuda:0")
m = lt.nn.Sequential([
    lt.nn.Conv2d(1, 4, 3, padding=1),
    lt.nn.ReLU(),
    lt.nn.Flatten(),
    lt.nn.Linear(4 * 4 * 4, 2),
])
m.to(dev)
params = m.parameters()
print(f"num params: {len(params)}", flush=True)
xt = lt.Tensor.from_vector([0.1] * (2 * 1 * 4 * 4), [2, 1, 4, 4], dev)
yt = lt.Tensor.from_vector([0.0, 1.0], [2], dev)
out = m.forward(xt)
print(f"out.requires_grad={out.requires_grad}", flush=True)
loss = lt.Ops.cross_entropy_loss(out, yt)
loss.backward()
for i, p in enumerate(params):
    print(f"param{i}: grad {'None' if p.grad is None else 'present'}", flush=True)
print("done", flush=True)
