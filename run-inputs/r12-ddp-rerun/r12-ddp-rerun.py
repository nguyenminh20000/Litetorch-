import json
import os
import subprocess
import sys
import shutil

TREE = os.environ.get("LITETORCH_TREE")
if not TREE:
    print("DDP BACKWARD FAIL: LITETORCH_TREE not set")
    sys.exit(2)
sys.path.insert(0, os.path.join(TREE, "python"))

try:
    import litetorch as lt
except Exception as e:
    print("DDP BACKWARD FAIL: import litetorch raised %r" % (e,))
    sys.exit(1)

try:
    import torch
except Exception as e:
    print("DDP BACKWARD FAIL: import torch raised %r (reference unavailable)" % (e,))
    sys.exit(1)

WORKER = r'''
import json
import os
import sys

TREE = os.environ.get("LITETORCH_TREE")
if not TREE:
    print("worker FAIL: LITETORCH_TREE not set")
    sys.exit(2)
sys.path.insert(0, os.path.join(TREE, "python"))
import litetorch as lt

def run_phase(rank, dev, cpu, xvals, seed):
    lt.manual_seed(seed)
    model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
    params = model.parameters()
    model.to(dev)
    ddp = lt.distributed.DistributedDataParallel(model)
    xx = lt.Tensor.from_vector(xvals, [1, 4], dev, False, lt.DataType.FP32)
    if xx.device.type != lt.DeviceType.GPU:
        raise RuntimeError("input device is %s, not GPU" % (xx.device,))
    loss = ddp.forward(xx).sum()
    ddp.backward(loss)
    wg = params[0].grad
    bg = params[1].grad
    if wg is None or bg is None:
        raise RuntimeError("missing grad after backward (weight grad %s, bias grad %s)" % (wg is None, bg is None))
    w0 = params[0].to(cpu).to_vector()
    b0 = params[1].to(cpu).to_vector()
    return {
        "w_init": w0,
        "b_init": b0,
        "wgrad": wg.to(cpu).to_vector(),
        "bgrad": bg.to(cpu).to_vector(),
    }

def main():
    rank = int(sys.argv[1])
    outdir = sys.argv[2]
    result = {"rank": rank, "ok": False}
    try:
        lt.cuda.set_device(rank)
        lt.distributed.init_process_group(rank, 2, "127.0.0.1", 29650)
        try:
            dev = lt.Device("cuda:%d" % rank)
            cpu = lt.Device("cpu")
            pa = run_phase(rank, dev, cpu, [0.5, 0.5, 0.5, 0.5], 7)
            pb = run_phase(rank, dev, cpu, [0.5, 0.5, 0.5, 0.5] if rank == 0 else [1.0, 1.0, 1.0, 1.0], 7)
            result.update({
                "ok": True,
                "w_init": pa["w_init"],
                "b_init": pa["b_init"],
                "phaseA_w": pa["wgrad"],
                "phaseA_b": pa["bgrad"],
                "phaseB_w": pb["wgrad"],
                "phaseB_b": pb["bgrad"],
            })
        finally:
            lt.distributed.shutdown()
    except Exception as e:
        result["error"] = "%s: %s" % (type(e).__name__, e)
    with open(os.path.join(outdir, "rank%d.json" % rank), "w") as f:
        json.dump(result, f)
    if not result["ok"]:
        print("worker rank %d FAIL: %s" % (rank, result.get("error", "unknown")))
        return 1
    print("worker rank %d OK" % rank)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

PORT_INFO = "port 29650"


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def torch_grads(w_flat, b_flat, xvals):
    lin = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(w_flat).reshape(4, 4))
        lin.bias.copy_(torch.tensor(b_flat))
    xt = torch.tensor([xvals])
    loss = lin(xt).sum()
    loss.backward()
    return lin.weight.grad.flatten().tolist(), lin.bias.grad.tolist()


def mean_grads(g1, g2):
    return [(a + b) / 2.0 for a, b in zip(g1, g2)]


def main():
    backend = lt.get_backend_name()
    print("DDP PREFLIGHT backend=%s cuda_available=%s lt_count=%s" % (
        backend, lt.cuda.is_available(), lt.cuda.device_count()))
    try:
        nccl = lt.distributed.NCCLBridge.get().is_available()
        print("DDP PREFLIGHT nccl_available=%s (info only; collectives fail loudly if broken)" % (nccl,))
    except Exception as e:
        print("DDP PREFLIGHT nccl query raised %r (info only)" % (e,))
    if backend not in ("CUDA", "ROCm"):
        print("DDP PREFLIGHT FAIL: backend is %s, not a native GPU backend" % backend)
        print("DDP BACKWARD FAIL")
        return 1
    if not lt.cuda.is_available():
        print("DDP PREFLIGHT FAIL: lt.cuda.is_available() is False")
        print("DDP BACKWARD FAIL")
        return 1
    if lt.cuda.device_count() < 2:
        print("DDP PREFLIGHT FAIL: lt.cuda.device_count() is %s, need >= 2" % lt.cuda.device_count())
        print("DDP BACKWARD FAIL")
        return 1
    print("DDP PREFLIGHT PASS")
    outdir = "/tmp/r12ddp_rerun"
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
    worker_path = os.path.join(outdir, "ddp_worker.py")
    with open(worker_path, "w") as f:
        f.write(WORKER)
    env = dict(os.environ)
    env["LITETORCH_TREE"] = TREE
    procs = []
    for r in (0, 1):
        procs.append(subprocess.Popen(
            [sys.executable, worker_path, str(r), outdir],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = []
    rcs = []
    for p in procs:
        try:
            o, _ = p.communicate(timeout=300)
            outs.append(o)
            rcs.append(p.returncode)
        except subprocess.TimeoutExpired:
            p.kill()
            outs.append("TIMEOUT")
            rcs.append(-1)
    for o in outs:
        print(o[-1000:])
    fails = []
    if rcs != [0, 0]:
        print("DDP BACKWARD FAIL: worker return codes %s (need [0, 0])" % (rcs,))
        print("DDP BACKWARD FAIL")
        return 1
    try:
        with open(os.path.join(outdir, "rank0.json")) as f:
            r0 = json.load(f)
        with open(os.path.join(outdir, "rank1.json")) as f:
            r1 = json.load(f)
    except Exception as e:
        print("DDP BACKWARD FAIL: cannot read worker results: %r" % (e,))
        print("DDP BACKWARD FAIL")
        return 1
    if not r0.get("ok") or not r1.get("ok"):
        print("DDP BACKWARD FAIL: rank0 error %s; rank1 error %s" % (r0.get("error"), r1.get("error")))
        print("DDP BACKWARD FAIL")
        return 1
    print("DDP WORKERS OK: both ranks completed forward AND backward")
    if r0["w_init"] != r1["w_init"] or r0["b_init"] != r1["b_init"]:
        print("DDP BACKWARD FAIL: init mismatch across ranks (same seed 7 must give identical weights)")
        print("DDP BACKWARD FAIL")
        return 1
    print("DDP INIT OK: identical weights across ranks (seed 7)")
    refA_w, refA_b = torch_grads(r0["w_init"], r0["b_init"], [0.5, 0.5, 0.5, 0.5])
    refB1_w, refB1_b = torch_grads(r0["w_init"], r0["b_init"], [0.5, 0.5, 0.5, 0.5])
    refB2_w, refB2_b = torch_grads(r0["w_init"], r0["b_init"], [1.0, 1.0, 1.0, 1.0])
    refB_w = mean_grads(refB1_w, refB2_w)
    refB_b = mean_grads(refB1_b, refB2_b)
    checks = [
        ("phaseA cross-rank weight", maxdiff(r0["phaseA_w"], r1["phaseA_w"]), 1e-6),
        ("phaseA cross-rank bias", maxdiff(r0["phaseA_b"], r1["phaseA_b"]), 1e-6),
        ("phaseA weight vs torch ref", maxdiff(r0["phaseA_w"], refA_w), 1e-4),
        ("phaseA bias vs torch ref", maxdiff(r0["phaseA_b"], refA_b), 1e-4),
        ("phaseB cross-rank weight", maxdiff(r0["phaseB_w"], r1["phaseB_w"]), 1e-6),
        ("phaseB cross-rank bias", maxdiff(r0["phaseB_b"], r1["phaseB_b"]), 1e-6),
        ("phaseB weight vs mean torch ref", maxdiff(r0["phaseB_w"], refB_w), 1e-4),
        ("phaseB bias vs mean torch ref", maxdiff(r0["phaseB_b"], refB_b), 1e-4),
    ]
    for name, d, tol in checks:
        ok = d <= tol
        print("DDP CHECK %s: max_abs_diff=%.3e tol=%.0e %s" % (name, d, tol, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append(name)
    if fails:
        print("DDP BACKWARD FAIL: failed checks: %s" % (fails,))
        print("DDP BACKWARD FAIL")
        return 1
    print("DDP BACKWARD PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
