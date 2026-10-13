import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import multiprocessing as mp

import litetorch as lt

CKPT = "/tmp/lt_d3_ckpt"
PORT_BASE = 29771


def run_workers(fn, world_size, port, *args):
    ctx = mp.get_context("spawn")
    with mp.Manager() as mgr:
        out = mgr.dict()
        procs = [ctx.Process(target=fn, args=(rk, world_size, port, out) + args)
                 for rk in range(world_size)]
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


def worker_rdzv(rank, world_size, port, out):
    r = {}
    try:
        c = lt.distributed.RendezvousClient("127.0.0.1", port, 15)
        my = c.add("next_rank", 1) - 1
        r["rank_assign"] = my == rank
        c.set(f"rank{rank}/addr", f"10.0.0.{rank}")
        peer = 1 - rank if world_size == 2 else rank
        val = c.get(f"rank{peer}/addr", 15)
        r["peer_value"] = val == f"10.0.0.{peer}"
        c.set(f"rank{rank}/done", "1")
        other = c.get(f"rank{peer}/done", 15)
        r["peer_done"] = other == "1"
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out[rank] = r


def test_rendezvous_store():
    print("=== rendezvous store (2 procs) ===", flush=True)
    port = PORT_BASE
    store = lt.distributed.RendezvousStore("127.0.0.1", 0)
    port = store.port()
    t = threading.Thread(target=store.run, daemon=True)
    t.start()
    time.sleep(0.3)
    res = run_workers(worker_rdzv, 2, port)
    store.stop()
    if res is None or len(res) != 2:
        print("[FAIL] rdzv workers hung", flush=True)
        return False
    return check_results(res, "rdzv")


def test_heartbeat():
    print("=== heartbeat failure detection ===", flush=True)
    ok = True
    store = lt.distributed.RendezvousStore("127.0.0.1", 0)
    port = store.port()
    t = threading.Thread(target=store.run, daemon=True)
    t.start()
    time.sleep(0.3)
    try:
        agent = lt.distributed.HeartbeatAgent("127.0.0.1", port, 0, 200)
        agent.start()
        time.sleep(0.8)
        dead = lt.distributed.detect_dead_ranks("127.0.0.1", port, 1500)
        if 0 in dead:
            print("[FAIL] heartbeat rank 0 wrongly dead", flush=True)
            ok = False
        else:
            print("[PASS] heartbeat rank 0 alive", flush=True)
        beats = lt.distributed.RendezvousClient("127.0.0.1", port, 5).get_heartbeats()
        if 0 not in beats:
            print("[FAIL] heartbeat not recorded", flush=True)
            ok = False
        else:
            print("[PASS] heartbeat recorded", flush=True)
        agent.stop()
        time.sleep(1.2)
        dead = lt.distributed.detect_dead_ranks("127.0.0.1", port, 500)
        if 0 not in dead:
            print("[FAIL] heartbeat stopped rank not detected dead", flush=True)
            ok = False
        else:
            print("[PASS] heartbeat stopped rank detected dead", flush=True)
    except Exception as e:
        print(f"[FAIL] heartbeat exception: {type(e).__name__}: {e}", flush=True)
        ok = False
    store.stop()
    return ok


def make_shards(rank, world_size):
    cpu = lt.Device("cpu")
    wref = [0.1 * (i + 1) for i in range(32)]
    bref = [0.01 * (i + 1) for i in range(8)]
    local = {
        "weight": lt.Tensor.from_vector(wref[rank * 16:(rank + 1) * 16], [16], cpu),
        "bias": lt.Tensor.from_vector(bref, [8], cpu),
    }
    specs = {
        "weight": lt.distributed.make_sharded_spec([32], "F32", 0, world_size),
        "bias": lt.distributed.make_replicated_spec([8], "F32"),
    }
    return local, specs


def worker_save_consistent(rank, world_size, port, out, step):
    r = {}
    try:
        lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
        local, specs = make_shards(rank, world_size)
        lt.distributed.save_consistent(local, specs, CKPT, step)
        r["save_ok"] = True
        if rank == 0:
            r["step_dir"] = os.path.isdir(os.path.join(CKPT, f"step-{step}"))
            r["meta_ok"] = os.path.exists(os.path.join(CKPT, f"step-{step}", ".metadata.json"))
            r["no_tmp"] = not os.path.exists(os.path.join(CKPT, f".tmp-step-{step}"))
            r["latest"] = lt.distributed.latest_consistent_step(CKPT) == step
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
    out[rank] = r
    lt.distributed.ProcessGroup.get().shutdown()


def worker_load_latest(rank, world_size, port, out):
    r = {}
    try:
        lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
        cpu = lt.Device("cpu")
        w_shard = lt.Tensor.zeros([16], cpu)
        b_full = lt.Tensor.zeros([8], cpu)
        lt.distributed.load_latest({"weight": w_shard, "bias": b_full}, CKPT)
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


def test_two_phase():
    print("=== two-phase consistent checkpoint (2 procs) ===", flush=True)
    if os.path.exists(CKPT):
        shutil.rmtree(CKPT)
    os.makedirs(CKPT, exist_ok=True)
    stale = os.path.join(CKPT, ".tmp-step-7")
    os.makedirs(stale, exist_ok=True)
    open(os.path.join(stale, "junk"), "w").write("stale")

    res = run_workers(worker_save_consistent, 2, PORT_BASE + 1, 7)
    if res is None or len(res) != 2:
        print("[FAIL] save_consistent workers hung", flush=True)
        return False
    ok = check_results(res, "save_consistent")
    if not os.path.isdir(os.path.join(CKPT, "step-7")):
        print("[FAIL] stale tmp not cleaned / step dir missing", flush=True)
        ok = False
    else:
        print("[PASS] stale tmp cleaned, step dir committed", flush=True)

    print("=== save step 8, latest should be 8 ===", flush=True)
    res = run_workers(worker_save_consistent, 2, PORT_BASE + 2, 8)
    if res is None or len(res) != 2:
        print("[FAIL] save step8 workers hung", flush=True)
        return False
    ok &= check_results(res, "save_step8")
    latest = lt.distributed.latest_consistent_step(CKPT)
    if latest != 8:
        print(f"[FAIL] latest_consistent_step={latest} want 8", flush=True)
        ok = False
    else:
        print("[PASS] latest_consistent_step=8", flush=True)

    print("=== load_latest (2 procs) ===", flush=True)
    res = run_workers(worker_load_latest, 2, PORT_BASE + 3)
    if res is None or len(res) != 2:
        print("[FAIL] load_latest workers hung", flush=True)
        return False
    ok &= check_results(res, "load_latest")

    print("=== incomplete tmp ignored by latest ===", flush=True)
    bad = os.path.join(CKPT, ".tmp-step-99")
    os.makedirs(bad, exist_ok=True)
    latest = lt.distributed.latest_consistent_step(CKPT)
    if latest != 8:
        print(f"[FAIL] tmp dir leaked into latest: {latest}", flush=True)
        ok = False
    else:
        print("[PASS] tmp dir ignored", flush=True)
    shutil.rmtree(bad, ignore_errors=True)
    return ok


WORKER_SCRIPT = '''
import os, sys
import litetorch as lt

CKPT = sys.argv[1]
attempt = int(os.environ.get("RESTART_ATTEMPT", "0"))
rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
port = int(os.environ.get("MASTER_PORT", "29500"))

lt.distributed.init_process_group(rank, world, "127.0.0.1", port)
cpu = lt.Device("cpu")
if attempt == 0:
    w = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
    specs = {"w": lt.distributed.make_sharded_spec([8], "F32", 0, world)}
    local = {"w": w}
    lt.distributed.save_consistent(local, specs, CKPT, 1)
    print(f"rank {rank} attempt 0: checkpoint saved, simulating failure", flush=True)
    lt.distributed.ProcessGroup.get().shutdown()
    sys.exit(1)
else:
    w = lt.Tensor.zeros([4], cpu)
    lt.distributed.load_latest({"w": w}, CKPT)
    v = w.to_vector()
    want = [float(rank + 1)] * 4
    ok = all(abs(a - b) < 1e-6 for a, b in zip(v, want))
    print(f"rank {rank} attempt {attempt}: resumed values ok={ok}", flush=True)
    lt.distributed.ProcessGroup.get().shutdown()
    sys.exit(0 if ok else 2)
'''


def test_ltrun_restart():
    print("=== lt-run --max-restarts elastic recovery ===", flush=True)
    tmpd = tempfile.mkdtemp(prefix="lt_d3_ltrun_")
    ckpt = os.path.join(tmpd, "ckpt")
    script = os.path.join(tmpd, "worker.py")
    with open(script, "w") as f:
        f.write(WORKER_SCRIPT)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, os.path.join(repo, "tools", "lt-run"),
           "--standalone", "--nproc-per-node", "2",
           "--max-restarts", "2", "--restart-delay", "1",
           "--rdzv-timeout", "30",
           script, ckpt]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        out = p.stdout + p.stderr
        print(out[-2000:], flush=True)
        ok = True
        if p.returncode != 0:
            print(f"[FAIL] lt-run exit code {p.returncode}", flush=True)
            ok = False
        else:
            print("[PASS] lt-run exit 0 after restart", flush=True)
        if "resumed values ok=True" not in out:
            print("[FAIL] worker did not resume from checkpoint", flush=True)
            ok = False
        else:
            print("[PASS] worker resumed from checkpoint", flush=True)
        if "restarting" not in out.lower():
            print("[FAIL] no restart happened", flush=True)
            ok = False
        else:
            print("[PASS] restart triggered", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        print("[FAIL] lt-run timed out", flush=True)
        return False
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


def main():
    all_ok = True
    all_ok &= test_rendezvous_store()
    all_ok &= test_heartbeat()
    all_ok &= test_two_phase()
    all_ok &= test_ltrun_restart()
    print(f"TOTAL: {'ALL PASS' if all_ok else 'FAILURES'}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
