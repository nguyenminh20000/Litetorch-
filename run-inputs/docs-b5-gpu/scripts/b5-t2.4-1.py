import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 T2.4-1 FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook

import litetorch as lt

gpu = lt.Device("cuda")
device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(4, 16, True),
    lt.nn.ReLU(),
    lt.nn.Linear(16, 2, True)
])
model.to(gpu)
params = model.parameters()
if params[0].device.type != lt.DeviceType.GPU:
    raise RuntimeError("no GPU device: refusing to run the GPU variant on CPU")

print("training on:", params[0].device)
print("B5 T2.4-1 PASS")
