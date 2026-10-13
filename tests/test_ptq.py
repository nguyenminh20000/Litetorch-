import sys
sys.path.insert(0, "python")
import numpy as np
import litetorch as lt
from litetorch import nn
from litetorch.quantization import prepare, convert, PTQConfig

passed = 0
failed = 0

def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name}")

def to_np(t):
    return np.array(t.tolist(), dtype=np.float64)

print("== PTQ prepare/calibrate/convert ==")
np.random.seed(2)
model = nn.Sequential([nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)])
x_np = (np.random.randn(4, 16)).astype(np.float32)
x = lt.tensor(x_np.tolist())
ref_out = to_np(model(x))

config = PTQConfig()
config.bits = 8
prepared = prepare(model, config)
check("prepare returns PreparedModel", prepared is not None)

batches = [lt.tensor((np.random.randn(4, 16)).astype(np.float32).tolist()) for _ in range(3)]
prepared.calibrate(lambda: batches.pop(0), num_batches=3)
check("calibrate completes", True)

qmodel = convert(prepared)
check("convert returns model", qmodel is not None)

q_out = to_np(qmodel(x))
abs_err = np.abs(q_out - ref_out)
rel_err = abs_err / (np.abs(ref_out) + 1e-3)
check("ptq output mean_rel_err<5e-2", float(rel_err.mean()) < 5e-2)
check("ptq output max_abs_err<2e-2", float(abs_err.max()) < 2e-2)

print("== PTQ preserves structure ==")
check("quantized model has children", len(qmodel.named_children()) == 3)

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
