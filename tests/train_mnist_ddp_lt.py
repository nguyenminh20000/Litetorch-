import os
import sys
import time

import numpy as np
import torch.multiprocessing as mp

import litetorch as lt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mnist import get_data, batches, build_lt, eval_lt

BATCH = 128
EPOCHS = 2
LR = 1e-3


def train_rank(rank, world_size, port, data):
    dev = lt.Device(f"cuda:{rank}")
    pg = lt.distributed.ProcessGroup.get()
    pg.init(rank, world_size, "127.0.0.1", port)

    (x_train, y_train), (x_test, y_test) = data
    n = len(x_train)
    lo, hi = (n // world_size) * rank, (n // world_size) * (rank + 1)
    xs, ys = x_train[lo:hi], y_train[lo:hi]

    model = build_lt()
    model.to(dev)
    for p in model.parameters():
        pg.broadcast(p, 0)
    opt = lt.optim.Adam(model.parameters(), lr=LR)
    scale = lt.Tensor.from_vector([float(world_size)], [1], dev)

    print(f"[lt-ddp] rank {rank}: preloading batches...", flush=True)
    lt_batches = []
    for xb, yb in batches(xs, ys, BATCH):
        b = len(xb)
        xt = lt.Tensor.from_vector(xb.reshape(-1).tolist(), [b, 1, 28, 28], dev)
        yt = lt.Tensor.from_vector([float(v) for v in yb], [b], dev)
        lt_batches.append((xt, yt))
    lt.cuda_synchronize()

    forward = model.forward
    xent = lt.Ops.cross_entropy_loss
    zero_grad = opt.zero_grad
    step = opt.step
    params = model.parameters()
    div = lt.Ops.div

    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xt, yt in lt_batches:
            zero_grad()
            out = forward(xt)
            loss = xent(out, yt)
            loss.backward()
            for p in params:
                g = p.grad
                if g is not None:
                    pg.all_reduce(g)
            pg.sync_comm()
            for p in params:
                g = p.grad
                if g is not None:
                    p.grad = div(g, scale)
            step()
        lt.cuda_synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        if rank == 0:
            print(f"[lt-ddp] epoch {ep+1}/{EPOCHS}: {dt:.2f}s", flush=True)

    if rank == 0:
        acc = eval_lt(model, x_test, y_test, dev)
        print(f"[lt-ddp] test acc: {acc:.4f}", flush=True)
        print(f"[lt-ddp] avg epoch: {sum(times)/len(times):.2f}s", flush=True)
    pg.shutdown()


def main():
    data = get_data()
    world_size = 2
    port = 29518
    mp.spawn(train_rank, args=(world_size, port, data), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
