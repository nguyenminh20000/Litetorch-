import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import multiprocessing as mp

import litetorch as lt

PORT = 29731


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([
            lt.nn.Linear(4096, 4096, True),
            lt.nn.GELU(),
            lt.nn.Linear(4096, 4096, True)
        ])
        model.to(cpu)
        lt.distributed.fsdp_fully_shard(model)
        x = lt.Tensor.from_vector([0.01] * 8192, [2, 4096], cpu)
        y = model(x)
        print("Rank %d shape forward: %s" % (rank, list(y.shape)))
        loss = y.sum()
        loss.backward()
        print("Rank %d da backward xong" % rank)
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(180)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    main()
