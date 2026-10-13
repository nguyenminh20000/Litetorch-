import sys
sys.path.insert(0, "python")
import numpy as np
import litetorch as lt
from litetorch import nn
from litetorch.quantization import quantize_dynamic_linear, quantize_dynamic_lstm, quantize_dynamic_gru

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

print("== Dynamic quant Linear ==")
np.random.seed(3)
lin = nn.Linear(16, 8)
x_np = (np.random.randn(4, 16)).astype(np.float32)
x = lt.tensor(x_np.tolist())
ref = to_np(lin(x))
dlin = quantize_dynamic_linear(lin)
out = to_np(dlin(x))
rel_err = np.abs(out - ref) / (np.abs(ref) + 1e-3)
check("dynamic linear rel_err<2e-2", float(rel_err.max()) < 2e-2)

print("== Dynamic quant LSTM ==")
lstm = nn.LSTM(8, 16, num_layers=1)
xl = lt.tensor((np.random.randn(5, 2, 8)).astype(np.float32).tolist())
ref_lstm = lstm(xl)
qlstm = quantize_dynamic_lstm(lstm)
out_lstm = qlstm(xl)
if isinstance(ref_lstm, tuple):
    ref_np = to_np(ref_lstm[0])
    out_np = to_np(out_lstm[0])
else:
    ref_np = to_np(ref_lstm)
    out_np = to_np(out_lstm)
rel_err_l = np.abs(out_np - ref_np) / (np.abs(ref_np) + 1e-3)
check("dynamic lstm rel_err<5e-2", float(rel_err_l.max()) < 5e-2)

print("== Dynamic quant GRU ==")
gru = nn.GRU(8, 16, num_layers=1)
ref_gru = gru(xl)
qgru = quantize_dynamic_gru(gru)
out_gru = qgru(xl)
if isinstance(ref_gru, tuple):
    ref_gnp = to_np(ref_gru[0])
    out_gnp = to_np(out_gru[0])
else:
    ref_gnp = to_np(ref_gru)
    out_gnp = to_np(out_gru)
rel_err_g = np.abs(out_gnp - ref_gnp) / (np.abs(ref_gnp) + 1e-3)
check("dynamic gru rel_err<5e-2", float(rel_err_g.max()) < 5e-2)

print(f"\nTOTAL: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
