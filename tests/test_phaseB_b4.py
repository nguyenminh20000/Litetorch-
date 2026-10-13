import os
import sys
import multiprocessing as mp

import litetorch as lt

PORT = 29682
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-5):
    va, vb = a.to_vector(), b.to_vector()
    if len(va) != len(vb):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(va, vb))


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        lt.distributed.FSDPTP.init(2, 1)
        r["init_ok"] = lt.distributed.FSDPTP.is_initialized()
        r["tp_size"] = lt.distributed.FSDPTP.tp_size() == 2
        r["dp_size"] = lt.distributed.FSDPTP.dp_size() == 1
        r["tp_rank"] = lt.distributed.FSDPTP.tp_rank() == rank
        r["dp_rank"] = lt.distributed.FSDPTP.dp_rank() == 0

        wref = [0.1 * (i + 1) for i in range(64)]
        bref = [0.01 * (i + 1) for i in range(8)]
        w_shard = wref[rank * 32:(rank + 1) * 32]
        b_shard = bref[rank * 4:(rank + 1) * 4]

        col = lt.nn.ColumnParallelLinear(8, 8, True)
        col.weight.copy_(lt.Tensor.from_vector(w_shard, [4, 8], cpu))
        col.bias.copy_(lt.Tensor.from_vector(b_shard, [4], cpu))

        lt.distributed.FSDPTP.fully_shard_tp(col)
        r["shard_ok"] = True

        sd = lt.distributed.FSDPTP.state_dict(col)
        r["sd_has_weight"] = "weight" in sd
        r["sd_has_bias"] = "bias" in sd
        if "weight" in sd:
            w_full = sd["weight"]
            r["sd_weight_shape"] = list(w_full.shape) == [8, 8]
            r["sd_weight_vals"] = allclose(
                w_full, lt.Tensor.from_vector(wref, [8, 8], cpu))
        if "bias" in sd:
            b_full = sd["bias"]
            r["sd_bias_shape"] = list(b_full.shape) == [8]
            r["sd_bias_vals"] = allclose(
                b_full, lt.Tensor.from_vector(bref, [8], cpu))

        wref2 = [0.2 * (i + 1) for i in range(64)]
        bref2 = [0.02 * (i + 1) for i in range(8)]
        sd2 = {
            "weight": lt.Tensor.from_vector(wref2, [8, 8], cpu),
            "bias": lt.Tensor.from_vector(bref2, [8], cpu),
        }
        lt.distributed.FSDPTP.load_state_dict(col, sd2)
        w_shard2 = wref2[rank * 32:(rank + 1) * 32]
        b_shard2 = bref2[rank * 4:(rank + 1) * 4]
        r["load_weight"] = allclose(
            col.weight, lt.Tensor.from_vector(w_shard2, [4, 8], cpu))
        r["load_bias"] = allclose(
            col.bias, lt.Tensor.from_vector(b_shard2, [4], cpu))

        xref = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 2.0, -2.0]
        x = lt.Tensor.from_vector(xref, [1, 8], cpu)
        x.requires_grad = True
        y = col.forward(x)
        loss = y.sum()
        loss.backward()
        r["fwd_bwd_ok"] = y is not None and x.grad is not None

    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
        print(f"[ERROR rank {rank}] {e}", flush=True)
        import traceback
        traceback.print_exc()
    out[rank] = r


def main():
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    out = manager.dict()
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT, out))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(120)
        if p.is_alive():
            p.terminate()

    all_results = {}
    for rank in range(2):
        if rank in out:
            for k, v in out[rank].items():
                all_results[f"rank{rank}_{k}"] = v

    print("\n===== RESULTS =====", flush=True)
    passed = sum(1 for v in all_results.values() if v is True)
    total = sum(1 for v in all_results.values() if isinstance(v, bool))
    for k in sorted(all_results):
        v = all_results[k]
        if isinstance(v, bool):
            print(f"[{'PASS' if v else 'FAIL'}] {k}", flush=True)
        else:
            print(f"[INFO] {k}: {v}", flush=True)
    print(f"\n{passed}/{total} passed", flush=True)
    sys.exit(0 if passed == total and total > 0 else 1)


if __name__ == "__main__":
    main()
