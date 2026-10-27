import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 E1-9 FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook
import sys
import litetorch as lt
print("build_commit", lt.build_commit())

assert lt.is_gpu_available()
t = lt.tensor([1.0, 2.0]).to(lt.Device("cuda"))
assert str(t.device) == "gpu:0"
print("on GPU:", t.device)
print("B5 E1-9 PASS")
