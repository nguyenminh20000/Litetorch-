import sys
import litetorch as lt

for dev_str in ["cpu", "cuda:0"]:
    dev = lt.Device(dev_str)
    m = lt.nn.Sequential([
        lt.nn.Linear(4, 4),
        lt.nn.ReLU(),
        lt.nn.Linear(4, 2),
    ])
    if dev_str != "cpu":
        m.to(dev)
    params = m.parameters()
    xt = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4] * 8, [8, 4], dev)
    yt = lt.Tensor.from_vector([float(i % 2) for i in range(8)], [8], dev)
    out = m.forward(xt)
    loss = lt.Ops.cross_entropy_loss(out, yt)
    loss.backward()
    n_none = sum(1 for p in params if p.grad is None)
    print(f"[{dev_str}] grads None: {n_none}/{len(params)}", flush=True)
print("done", flush=True)
