import os
import sys
import time

import numpy as np
import torch.multiprocessing as mp

import litetorch as lt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def diag_rank(rank, world_size, port):
    dev = lt.Device(f"cuda:{rank}")
    pg = lt.distributed.ProcessGroup.get()
    pg.init(rank, world_size, "127.0.0.1", port)

    m = lt.nn.Sequential([
        lt.nn.Linear(4, 4),
        lt.nn.ReLU(),
        lt.nn.Linear(4, 2),
    ])
    m.to(dev)
    for p in m.parameters():
        pg.broadcast(p, 0)
    pg.sync_comm()

    params = m.parameters()
    if rank == 0:
        v0 = np.array(params[0].to(lt.Device("cpu")).to_vector())
        print(f"[diag] rank0 param0 mean={v0.mean():.6f} std={v0.std():.6f}", flush=True)
    if rank == 1:
        v1 = np.array(params[0].to(lt.Device("cpu")).to_vector())
        print(f"[diag] rank1 param0 mean={v1.mean():.6f} std={v1.std():.6f}", flush=True)

    xt = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4] * 8, [8, 4], dev)
    yt = lt.Tensor.from_vector([float(i % 2) for i in range(8)], [8], dev)
    opt = lt.optim.Adam(params, lr=1e-3)
    scale = lt.Tensor.from_vector([float(world_size)], [1], dev)

    opt.zero_grad()
    out = m.forward(xt)
    loss = lt.Ops.cross_entropy_loss(out, yt)
    loss.backward()
    lt.cuda_synchronize()

    for i, p in enumerate(params):
        g = p.grad
        if g is None:
            print(f"[diag] rank{rank} param{i}: grad is None!", flush=True)
            continue
        gv = np.array(g.to(lt.Device("cpu")).to_vector())
        print(f"[diag] rank{rank} param{i}: pre-allreduce grad mean={gv.mean():.6f} std={gv.std():.6f} max={gv.max():.6f}", flush=True)

    for p in params:
        g = p.grad
        if g is not None:
            pg.all_reduce(g)
    pg.sync_comm()

    for i, p in enumerate(params):
        g = p.grad
        if g is None:
            continue
        gv = np.array(g.to(lt.Device("cpu")).to_vector())
        print(f"[diag] rank{rank} param{i}: post-allreduce grad mean={gv.mean():.6f} std={gv.std():.6f} max={gv.max():.6f}", flush=True)
        p.grad = lt.Ops.div(g, scale)
    lt.cuda_synchronize()

    for i, p in enumerate(params):
        g = p.grad
        if g is None:
            continue
        gv = np.array(g.to(lt.Device("cpu")).to_vector())
        print(f"[diag] rank{rank} param{i}: post-div grad mean={gv.mean():.6f} std={gv.std():.6f} max={gv.max():.6f}", flush=True)

    opt.step()
    lt.cuda_synchronize()
    v = np.array(params[0].to(lt.Device("cpu")).to_vector())
    print(f"[diag] rank{rank} param0 after step: mean={v.mean():.6f} std={v.std():.6f}", flush=True)
    pg.shutdown()


def main():
    mp.spawn(diag_rank, args=(2, 29521), nprocs=2, join=True)


if __name__ == "__main__":
    main()
