import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as tnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mnist import get_data, batches, build_torch, eval_torch

BATCH = 128
EPOCHS = 2
LR = 1e-3


def train_rank(rank, world_size, port, data):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    (x_train, y_train), (x_test, y_test) = data
    n = len(x_train)
    lo, hi = (n // world_size) * rank, (n // world_size) * (rank + 1)
    xs, ys = x_train[lo:hi], y_train[lo:hi]

    torch.manual_seed(0)
    model = build_torch().cuda(rank)
    model = tnn.parallel.DistributedDataParallel(model, device_ids=[rank])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = tnn.CrossEntropyLoss()

    times = []
    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        for xb, yb in batches(xs, ys, BATCH):
            xt = torch.from_numpy(xb).unsqueeze(1).cuda(rank)
            yt = torch.from_numpy(yb).cuda(rank)
            opt.zero_grad()
            loss = ce(model(xt), yt)
            loss.backward()
            opt.step()
        torch.cuda.synchronize(rank)
        dt = time.perf_counter() - t0
        times.append(dt)
        if rank == 0:
            print(f"[torch-ddp] epoch {ep+1}/{EPOCHS}: {dt:.2f}s", flush=True)

    acc = 0.0
    if rank == 0:
        acc = eval_torch(model.module, x_test, y_test)
        print(f"[torch-ddp] test acc: {acc:.4f}", flush=True)
        print(f"[torch-ddp] avg epoch: {sum(times)/len(times):.2f}s", flush=True)
    dist.destroy_process_group()
    return acc


def main():
    data = get_data()
    world_size = 2
    port = 29519
    mp.spawn(train_rank, args=(world_size, port, data), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
