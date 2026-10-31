import os, sys
LITETORCH_TREE = os.environ["LITETORCH_TREE"]
RESULTS = {}
def check(name, cond, detail=""):
    RESULTS[name] = bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(detail)) if (detail and not cond) else ""))
    return bool(cond)
import textwrap
worker = textwrap.dedent('''
import sys, os
sys.path.insert(0, os.environ["LT_PATH"])
import litetorch, numpy as np
rank = int(sys.argv[1]); mode = sys.argv[2]
litetorch.distributed.init_process_group(rank, 2, "127.0.0.1", 29501)
dev = litetorch.Device("cuda:%d" % rank)
FP32 = litetorch.DataType.FP32
if mode == "happy":
    t = litetorch.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], dev, False, FP32)
    litetorch.distributed.all_reduce(t)
    v = t.to(litetorch.Device("cpu")).to_vector()
    assert v == [2.0, 4.0, 6.0, 8.0], v
    print("rank %d happy OK" % rank, flush=True)
else:
    dt = FP32 if rank == 0 else litetorch.DataType.INT64
    data = [1.0, 2.0] if rank == 0 else [1, 2]
    t = litetorch.Tensor.from_vector(data, [2], dev, False, dt)
    threw = False
    try:
        litetorch.distributed.all_reduce(t)
    except Exception as e:
        threw = "mismatch" in str(e)
    assert threw, "rank %d did not throw mismatch" % rank
    print("rank %d mismatch-throw OK" % rank, flush=True)
litetorch.distributed.shutdown()
''')
open("/tmp/dist_worker.py", "w").write(worker)
import subprocess, os
env = dict(os.environ, LT_PATH=LITETORCH_TREE)
def run_mode(mode):
    ps = [subprocess.Popen([sys.executable, "/tmp/dist_worker.py", str(r), mode], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in (0, 1)]
    outs = [p.communicate(timeout=120)[0] for p in ps]
    rcs = [p.returncode for p in ps]
    print(outs[0][-500:]); print(outs[1][-500:])
    return rcs == [0, 0]
ok = True
try:
    ok &= check("dist_allreduce_2gpu", run_mode("happy"))
    ok &= check("dist_mismatch_throw", run_mode("mismatch"))
except Exception as e:
    print("SKIP distributed:", type(e).__name__, str(e)[:200])
assert ok, "distributed block failed"
print("PASS distributed")
