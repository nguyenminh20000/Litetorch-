import litetorch as lt

dev = lt.Device("cuda:0")
lin = lt.nn.Linear(4, 2)
lin.to(dev)
w = lin.weight
print(f"weight requires_grad={w.requires_grad}", flush=True)
xt = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4] * 8, [8, 4], dev)
out = lin.forward(xt)
print(f"out.requires_grad={out.requires_grad}", flush=True)
loss = lt.Ops.sum(out)
loss.backward()
print(f"weight.grad is {'None' if w.grad is None else 'present'}", flush=True)
print("done", flush=True)
