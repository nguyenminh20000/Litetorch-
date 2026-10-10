import litetorch as lt

for dev_str in ["cpu", "cuda:0"]:
    dev = lt.Device(dev_str)
    lin = lt.nn.Linear(4, 2)
    if dev_str != "cpu":
        lin.to(dev)
    w = lin.weight
    print(f"[{dev_str}] weight requires_grad={w.requires_grad}", flush=True)
    xt = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4] * 8, [8, 4], dev)
    out = lin.forward(xt)
    print(f"[{dev_str}] out.requires_grad={out.requires_grad}", flush=True)
    loss = lt.Ops.sum(out)
    print(f"[{dev_str}] loss.requires_grad={loss.requires_grad}", flush=True)
    loss.backward()
    print(f"[{dev_str}] weight.grad is {'None' if w.grad is None else 'present'}", flush=True)
print("done", flush=True)
