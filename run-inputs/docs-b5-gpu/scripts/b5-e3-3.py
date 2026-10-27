import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 e3-3 FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook
import sys
import litetorch as lt
lt.manual_seed(42)
assert lt.cuda.is_available(), "no CUDA backend"
assert lt.cuda.device_count() >= 1, "no GPU device"
print("cuda available", lt.cuda.is_available())
print("device count", lt.cuda.device_count())
print("memory before", lt.cuda.memory_allocated())
held = []
for i in range(6):
    t = lt.randn([1024, 1024]).to(lt.Device("cuda"))
    held.append(t)
    assert str(t.device) == "gpu:0", "silent fallback to CPU"
    print("after alloc", i, "memory allocated", lt.cuda.memory_allocated())
assert lt.cuda.memory_allocated() > 0, "no GPU memory registered"
print("survived allocations", len(held))
print("B5 e3-3 PASS")
