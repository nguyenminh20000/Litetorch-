import os
import sys
import shutil
import multiprocessing as mp

import litetorch as lt

PORT = 29681
CKPT = "/tmp/lt_d1_ckpt"


def worker_save(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        wref = [0.1 * (i + 1) for i in range(32)]
        bref = [0.01 * (i + 1) for i in range(8)]
        w_shard = lt.Tensor.from_vector(wref[rank * 16:(rank + 1) * 16], [16], cpu)
        b_full = lt.Tensor.from_vector(bref, [8], cpu)
        local = {"weight": w_shard, "bias": b_full}
        specs = {
            "weight": lt.distributed.make_sharded_spec([32], "F32", 0, world_size),
            "bias": lt.distributed.make_replicated_spec([8], "F32"),
        }
        lt.distributed.save_distributed(local, specs, CKPT, 42)
        r["save_ok"] = True
        r["shard_file"] = os.path.exists(os.path.join(CKPT, f"rank-{rank:05d}-of-{world_size:05d}.safetensors"))
        r["meta_file"] = os.path.exists(os.path.join(CKPT, ".metadata.json"))
        meta = lt.distributed.read_dist_metadata(CKPT)
        r["meta_ws"] = meta.world_size == world_size
        r["meta_step"] = meta.step == 42
        r["meta_tensors"] = set(meta.tensors.keys()) == {"weight", "bias"}
        r["meta_shard_dim"] = meta.tensors["weight"].shard_dim == 0
        r["meta_replicated"] = meta.tensors["bias"].shard_dim == -1
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out[rank] = r
    lt.distributed.ProcessGroup.get().shutdown()


def worker_load_same(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        w_shard = lt.Tensor.zeros([16], cpu)
        b_full = lt.Tensor.zeros([8], cpu)
        lt.distributed.load_distributed({"weight": w_shard, "bias": b_full}, CKPT)
        wref = [0.1 * (i + 1) for i in range(32)]
        bref = [0.01 * (i + 1) for i in range(8)]
        wv = w_shard.to_vector()
        bv = b_full.to_vector()
        r["weight_ok"] = all(abs(a - b) < 1e-5 for a, b in zip(wv, wref[rank * 16:(rank + 1) * 16]))
        r["bias_ok"] = all(abs(a - b) < 1e-6 for a, b in zip(bv, bref))
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out[rank] = r
    lt.distributed.ProcessGroup.get().shutdown()


def worker_load_reshard(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        w_full = lt.Tensor.zeros([32], cpu)
        b_full = lt.Tensor.zeros([8], cpu)
        lt.distributed.load_distributed({"weight": w_full, "bias": b_full}, CKPT)
        wref = [0.1 * (i + 1) for i in range(32)]
        bref = [0.01 * (i + 1) for i in range(8)]
        wv = w_full.to_vector()
        bv = b_full.to_vector()
        r["weight_ok"] = all(abs(a - b) < 1e-5 for a, b in zip(wv, wref))
        r["bias_ok"] = all(abs(a - b) < 1e-6 for a, b in zip(bv, bref))
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out[rank] = r
    lt.distributed.ProcessGroup.get().shutdown()


def run_workers(fn, world_size, port):
    ctx = mp.get_context("spawn")
    with mp.Manager() as mgr:
        out = mgr.dict()
        procs = [ctx.Process(target=fn, args=(rk, world_size, port, out)) for rk in range(world_size)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        alive = [p.is_alive() for p in procs]
        for p in procs:
            if p.is_alive():
                p.terminate()
        if any(alive):
            return None
        return {k: dict(v) for k, v in out.items()}


def check_results(res, label):
    ok = True
    for rank in sorted(res):
        for k, v in sorted(res[rank].items()):
            if k == "exception":
                print(f"[FAIL] {label} rank{rank} exception: {v}", flush=True)
                ok = False
            elif v is not True:
                print(f"[FAIL] {label} rank{rank} {k}={v}", flush=True)
                ok = False
            else:
                print(f"[PASS] {label} rank{rank} {k}", flush=True)
    return ok


def main():
    if os.path.exists(CKPT):
        shutil.rmtree(CKPT)
    all_ok = True

    print("=== save (2 procs) ===", flush=True)
    res = run_workers(worker_save, 2, PORT)
    if res is None or len(res) != 2:
        print("[FAIL] save workers hung", flush=True)
        sys.exit(1)
    all_ok &= check_results(res, "save")

    print("=== load same world size (2 procs) ===", flush=True)
    res = run_workers(worker_load_same, 2, PORT + 1)
    if res is None or len(res) != 2:
        print("[FAIL] load workers hung", flush=True)
        sys.exit(1)
    all_ok &= check_results(res, "load_same")

    print("=== reshard 2 -> 1 ===", flush=True)
    res = run_workers(worker_load_reshard, 1, PORT + 2)
    if res is None or len(res) != 1:
        print("[FAIL] reshard workers hung", flush=True)
        sys.exit(1)
    all_ok &= check_results(res, "reshard")

    print(f"TOTAL: {'ALL PASS' if all_ok else 'FAILURES'}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
