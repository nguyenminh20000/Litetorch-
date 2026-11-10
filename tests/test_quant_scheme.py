import sys
sys.path.insert(0, "python")
import shutil
import numpy as np
import litetorch as lt
from litetorch import nn
from litetorch.quantization import (
    register_scheme, unregister_scheme, has_scheme, list_schemes,
    quantize, quantize_ptq, save_quantized, load_quantized,
    prepare, PTQConfig, QParams, QuantizedTensor, SchemeQATLinear,
)
import torch
import torch.nn.functional as F

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

def raises_with(fn, needle):
    try:
        fn()
    except Exception as e:
        return needle in str(e)
    return False

print("== registry listing ==")
names = list_schemes()
for n in ["int8-per-channel", "dynamic-int8", "fake-quant", "nf4-block64"]:
    check(f"list_schemes contains {n}", n in names)
check("has_scheme unknown is False", not has_scheme("py-no-such-scheme"))
check("get via quantize unknown raises", raises_with(
    lambda: quantize(nn.Sequential([nn.Linear(2, 2)]), "py-no-such-scheme"), "not registered"))

print("== custom Python scheme ==")
def py_calibrate(t, cal, name, params):
    x = to_np(t)
    bits = int(params.get("bits", "8"))
    bound = float(2 ** (bits - 1) - 1)
    mn, mx = float(x.min()), float(x.max())
    scale = (mx - mn) / (2.0 * bound)
    if scale == 0.0:
        scale = 1.0
    zp = round(-bound - mn / scale)
    zp = max(-bound, min(bound, zp))
    qp = QParams()
    qp.scale = scale
    qp.zero_point = zp
    qp.bits = bits
    return qp

def py_quantize(t, qp, params):
    x = to_np(t)
    bound = float(2 ** (qp.bits - 1) - 1)
    q = np.round(x / qp.scale + qp.zero_point)
    q = np.clip(q, -bound, bound)
    rec = QuantizedTensor()
    rec.data = lt.tensor(q.astype(np.float32).tolist())
    rec.qparams = qp
    rec.scheme_name = "py-affine"
    rec.scheme_params = dict(params)
    rec.logical_shape = list(q.shape)
    rec.logical_dtype = "FP32"
    return rec

def py_dequantize(rec):
    d = to_np(rec.data)
    qp = rec.qparams
    out = (d - qp.zero_point) * qp.scale
    return lt.tensor(out.astype(np.float32).tolist())

register_scheme("py-affine", {}, py_calibrate, py_quantize, py_dequantize)
check("python scheme registered", has_scheme("py-affine"))

np.random.seed(7)
w_np = (np.random.randn(3, 4)).astype(np.float32) * 0.5
b_np = (np.random.randn(3)).astype(np.float32) * 0.1
lin = nn.Linear(4, 3)
lin.weight = lt.tensor(w_np.tolist())
lin.bias = lt.tensor(b_np.tolist())
model = nn.Sequential([lin, nn.ReLU()])
x_np = (np.random.randn(2, 4)).astype(np.float32)
x = lt.tensor(x_np.tolist())
qmodel = quantize(model, "py-affine")
kids = dict(qmodel.named_children())
check("walker replaced linear with SchemeQuantizedLinear", "SchemeQuantizedLinear" in type(kids["0"]).__name__)
got = to_np(qmodel(x))
bound = 127.0
mn, mx = float(w_np.min()), float(w_np.max())
scale = (mx - mn) / (2.0 * bound)
zp = round(-bound - mn / scale)
zp = max(-bound, min(bound, zp))
qw = np.clip(np.round(w_np / scale + zp), -bound, bound)
dq = (qw - zp) * scale
ref = np.maximum(F.linear(torch.tensor(x_np), torch.tensor(dq), torch.tensor(b_np)).numpy(), 0.0)
check("py-affine walker forward matches manual dequant plus torch linear", np.max(np.abs(got - ref)) < 1e-4)

print("== int8-per-channel torch parity ==")
np.random.seed(11)
w_np = (np.random.randn(4, 5)).astype(np.float32)
b_np = (np.random.randn(4)).astype(np.float32) * 0.1
lin = nn.Linear(5, 4)
lin.weight = lt.tensor(w_np.tolist())
lin.bias = lt.tensor(b_np.tolist())
model = nn.Sequential([lin])
qmodel = quantize(model, "int8-per-channel")
kids = dict(qmodel.named_children())
rec = kids["0"].record
scales = to_np(rec.qparams.scales)
check("int8 scales equal row maxabs over 127", np.max(np.abs(scales - np.max(np.abs(w_np), axis=1) / 127.0)) < 1e-6)
codes = to_np(rec.data)
scales_t = torch.tensor(scales.astype(np.float32))
zeros_t = torch.zeros(4, dtype=torch.int64)
tq = torch.quantize_per_channel(torch.tensor(w_np), scales_t, zeros_t, 0, torch.qint8)
tq_codes = tq.int_repr().numpy().astype(np.float64)
check("int8 codes match torch quantize_per_channel within 1 LSB", np.max(np.abs(codes - tq_codes)) <= 1.0)
tq_dq = tq.dequantize().numpy()
lt_dq = (codes * scales[:, None])
check("int8 dequant matches torch dequant within 1 scale", np.max(np.abs(lt_dq - tq_dq)) <= np.max(scales) + 1e-6)
x_np = (np.random.randn(3, 5)).astype(np.float32)
x = lt.tensor(x_np.tolist())
got = to_np(qmodel(x))
ref = F.linear(torch.tensor(x_np), torch.tensor(lt_dq.astype(np.float32)), torch.tensor(b_np)).numpy()
check("int8 walker forward matches torch linear on dequant weights", np.max(np.abs(got - ref)) < 1e-4)

print("== nf4-block64 ==")
np.random.seed(13)
w_np = (np.random.randn(130)).astype(np.float32)
lin = nn.Linear(130, 1)
lin.weight = lt.tensor(w_np.reshape(1, 130).tolist())
lin.bias = lt.tensor([0.0])
model = nn.Sequential([lin])
qmodel = quantize(model, "nf4-block64", {"block_size": "64"})
kids = dict(qmodel.named_children())
rec = kids["0"].record
scales = to_np(rec.qparams.scales)
expect_scales = np.array([np.max(np.abs(w_np[0:64])), np.max(np.abs(w_np[64:128])), np.max(np.abs(w_np[128:130]))])
check("nf4 scales equal block maxabs", np.max(np.abs(scales - expect_scales)) < 1e-6)
check("nf4 record scheme params carry block_size", rec.scheme_params.get("block_size") == "64")
x_np = (np.random.randn(2, 130)).astype(np.float32)
x = lt.tensor(x_np.tolist())
got = to_np(qmodel(x))
check("nf4 forward runs and is finite", np.all(np.isfinite(got)))
eye = lt.tensor(np.eye(130, dtype=np.float32).tolist())
dq = to_np(qmodel(eye))[:, 0]
per_elem_scale = np.concatenate([np.full(64, scales[0]), np.full(64, scales[1]), np.full(2, scales[2])])
check("nf4 per-element dequant within half grid gap of original",
      np.max(np.abs(dq - w_np)) <= 0.16 * per_elem_scale.max() + 1e-5)
shutil.rmtree("py_w54_tmp3", ignore_errors=True)
save_quantized(qmodel, "py_w54_tmp3")
ref_nf4 = to_np(qmodel(x))
fresh_nf4 = nn.Sequential([nn.Linear(130, 1)])
loaded_nf4 = load_quantized(fresh_nf4, "py_w54_tmp3")
got_nf4 = to_np(loaded_nf4(x))
check("nf4 save load round-trip forward exactly equal", np.array_equal(got_nf4, ref_nf4))
kids_nf4 = dict(loaded_nf4.named_children())
check("nf4 loaded record keeps NF4 stored dtype", kids_nf4["0"].record.scheme_name == "nf4-block64")
shutil.rmtree("py_w54_tmp3", ignore_errors=True)

print("== dynamic-int8 ==")
np.random.seed(17)
model = nn.Sequential([nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 2)])
x_np = (np.random.randn(2, 6)).astype(np.float32)
x = lt.tensor(x_np.tolist())
qmodel = quantize(model, "dynamic-int8")
kids = dict(qmodel.named_children())
check("dynamic-int8 keeps 3 children", len(kids) == 3)
out = qmodel(x)
check("dynamic-int8 forward runs and is finite", np.all(np.isfinite(to_np(out))))

print("== LSTM walker ==")
np.random.seed(19)
lstm = nn.LSTM(8, 16, num_layers=1)
xl = lt.tensor((np.random.randn(5, 2, 8)).astype(np.float32).tolist())
qlstm = quantize(lstm, "fake-quant")
out = qlstm(xl)
out_np = to_np(out[0]) if isinstance(out, tuple) else to_np(out)
check("fake-quant LSTM forward runs and is finite", np.all(np.isfinite(out_np)))

print("== PTQ tier ==")
np.random.seed(23)
model = nn.Sequential([nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)])
x_np = (np.random.randn(2, 8)).astype(np.float32)
x = lt.tensor(x_np.tolist())
prepared = prepare(model, PTQConfig())
check("quantize_ptq without calibrate raises", raises_with(
    lambda: quantize_ptq(prepared, "fake-quant"), "calibrate"))
prepared.calibrate(lambda: lt.tensor(x_np.tolist()), 2)
qmodel = quantize_ptq(prepared, "fake-quant")
kids = dict(qmodel.named_children())
check("ptq walker replaced linears", "SchemeQuantizedLinear" in type(kids["0"]).__name__ and "SchemeQuantizedLinear" in type(kids["2"]).__name__)
out = qmodel(x)
check("ptq forward runs and is finite", np.all(np.isfinite(to_np(out))))

print("== QAT tier ==")
np.random.seed(29)
w_np = (np.random.randn(3, 4)).astype(np.float32) * 0.5
b_np = (np.random.randn(3)).astype(np.float32) * 0.1
lin = nn.Linear(4, 3)
lin.weight = lt.tensor(w_np.tolist())
lin.bias = lt.tensor(b_np.tolist())
qat = SchemeQATLinear(lin, "fake-quant")
x_np = (np.random.randn(2, 4)).astype(np.float32)
x = lt.tensor(x_np.tolist())
qat.train()
tout = to_np(qat(x))
check("qat training forward runs and is finite", np.all(np.isfinite(tout)))
qat.eval()
eout = to_np(qat(x))
ref = F.linear(torch.tensor(x_np), torch.tensor(w_np), torch.tensor(b_np)).numpy()
check("qat eval forward equals torch linear", np.max(np.abs(eout - ref)) < 1e-4)
qat.train()
tout2 = to_np(qat(x))
mn, mx = float(w_np.min()), float(w_np.max())
w_scale = (mx - mn) / 254.0
w_zp = round(-127.0 - mn / w_scale)
w_zp = max(-127.0, min(127.0, w_zp))
fq_w = torch.fake_quantize_per_tensor_affine(torch.tensor(w_np), float(w_scale), int(w_zp), -127, 127)
mn2, mx2 = float(x_np.min()), float(x_np.max())
a_scale = (mx2 - mn2) / 254.0
a_zp = round(-127.0 - mn2 / a_scale)
a_zp = max(-127.0, min(127.0, a_zp))
fq_x = torch.fake_quantize_per_tensor_affine(torch.tensor(x_np), float(a_scale), int(a_zp), -127, 127)
ref2 = F.linear(fq_x, fq_w, torch.tensor(b_np)).numpy()
check("qat training forward matches torch fake-quant linear within 1 LSB",
      np.max(np.abs(tout2 - ref2)) <= max(w_scale, a_scale) * 1.5 + 1e-4)

print("== save and load ==")
np.random.seed(31)
model = nn.Sequential([nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)])
x_np = (np.random.randn(2, 4)).astype(np.float32)
x = lt.tensor(x_np.tolist())
qmodel = quantize(model, "int8-per-channel")
ref = to_np(qmodel(x))
shutil.rmtree("py_w54_tmp", ignore_errors=True)
save_quantized(qmodel, "py_w54_tmp")
fresh = nn.Sequential([nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)])
loaded = load_quantized(fresh, "py_w54_tmp")
got = to_np(loaded(x))
check("save load round-trip forward exactly equal", np.array_equal(got, ref))
kids = dict(loaded.named_children())
check("loaded children are SchemeQuantizedLinear", "SchemeQuantizedLinear" in type(kids["0"]).__name__)
shutil.rmtree("py_w54_tmp", ignore_errors=True)

print("== save load custom scheme requires re-registration ==")
np.random.seed(37)
model = nn.Sequential([nn.Linear(4, 3)])
qmodel = quantize(model, "py-affine")
shutil.rmtree("py_w54_tmp2", ignore_errors=True)
save_quantized(qmodel, "py_w54_tmp2")
unregister_scheme("py-affine")
check("py-affine unregistered", not has_scheme("py-affine"))
fresh = nn.Sequential([nn.Linear(4, 3)])
check("load without re-registration raises", raises_with(
    lambda: load_quantized(fresh, "py_w54_tmp2"), "not registered"))
register_scheme("py-affine", {}, py_calibrate, py_quantize, py_dequantize)
loaded = load_quantized(fresh, "py_w54_tmp2")
kids = dict(loaded.named_children())
check("load after re-registration restores scheme", kids["0"].record.scheme_name == "py-affine")
unregister_scheme("py-affine")
shutil.rmtree("py_w54_tmp2", ignore_errors=True)

print("== LSTM/GRU cell save load ==")
def cell_out(model, x):
    o = model(x)
    return o[0] if isinstance(o, tuple) else o

def params_agree(cell, members):
    reg_vals = [to_np(t) for t in cell.parameters()]
    for m in members:
        mv = to_np(m)
        if not any(np.array_equal(rv, mv) for rv in reg_vals):
            return False
    return True

np.random.seed(41)
lstm = nn.LSTM(8, 16, num_layers=1)
xl = lt.tensor((np.random.randn(5, 2, 8)).astype(np.float32).tolist())
qlstm = quantize(lstm, "fake-quant")
ref = to_np(cell_out(qlstm, xl))
shutil.rmtree("py_w59_tmp_lstm", ignore_errors=True)
save_quantized(qlstm, "py_w59_tmp_lstm")
fresh = nn.LSTM(8, 16, num_layers=1)
loaded = load_quantized(fresh, "py_w59_tmp_lstm")
got = to_np(cell_out(loaded, xl))
check("lstm cell save load round-trip forward exactly equal", np.array_equal(got, ref))
check("lstm loaded biases equal quantized biases",
      np.array_equal(to_np(loaded.cells[0].bias_ih), to_np(qlstm.cells[0].bias_ih)) and
      np.array_equal(to_np(loaded.cells[0].bias_hh), to_np(qlstm.cells[0].bias_hh)))
check("lstm named parameters agree with members after load",
      params_agree(loaded.cells[0], [loaded.cells[0].weight_ih, loaded.cells[0].weight_hh,
                                     loaded.cells[0].bias_ih, loaded.cells[0].bias_hh]))
shutil.rmtree("py_w59_tmp_lstm", ignore_errors=True)

np.random.seed(43)
gru = nn.GRU(8, 16, num_layers=1)
xg = lt.tensor((np.random.randn(5, 2, 8)).astype(np.float32).tolist())
qgru = quantize(gru, "fake-quant")
ref = to_np(cell_out(qgru, xg))
shutil.rmtree("py_w59_tmp_gru", ignore_errors=True)
save_quantized(qgru, "py_w59_tmp_gru")
fresh = nn.GRU(8, 16, num_layers=1)
loaded = load_quantized(fresh, "py_w59_tmp_gru")
got = to_np(cell_out(loaded, xg))
check("gru cell save load round-trip forward exactly equal", np.array_equal(got, ref))
check("gru loaded biases equal quantized biases",
      np.array_equal(to_np(loaded.cells[0].bias_ih), to_np(qgru.cells[0].bias_ih)) and
      np.array_equal(to_np(loaded.cells[0].bias_hh), to_np(qgru.cells[0].bias_hh)))
check("gru named parameters agree with members after load",
      params_agree(loaded.cells[0], [loaded.cells[0].weight_ih, loaded.cells[0].weight_hh,
                                     loaded.cells[0].bias_ih, loaded.cells[0].bias_hh]))
shutil.rmtree("py_w59_tmp_gru", ignore_errors=True)

np.random.seed(47)
lstm_nb = nn.LSTM(8, 16, num_layers=1, bias=False)
xn = lt.tensor((np.random.randn(5, 2, 8)).astype(np.float32).tolist())
qlstm_nb = quantize(lstm_nb, "fake-quant")
ref = to_np(cell_out(qlstm_nb, xn))
shutil.rmtree("py_w59_tmp_nobias", ignore_errors=True)
save_quantized(qlstm_nb, "py_w59_tmp_nobias")
fresh = nn.LSTM(8, 16, num_layers=1, bias=False)
loaded = load_quantized(fresh, "py_w59_tmp_nobias")
got = to_np(cell_out(loaded, xn))
check("bias-free lstm cell save load round-trip forward exactly equal", np.array_equal(got, ref))
check("bias-free lstm loaded cell has no biases", loaded.cells[0].bias_ih is None and loaded.cells[0].bias_hh is None)
shutil.rmtree("py_w59_tmp_nobias", ignore_errors=True)

np.random.seed(49)
gru_nb = nn.GRU(8, 16, num_layers=1, bias=False)
xn = lt.tensor((np.random.randn(5, 2, 8)).astype(np.float32).tolist())
qgru_nb = quantize(gru_nb, "fake-quant")
ref = to_np(cell_out(qgru_nb, xn))
shutil.rmtree("py_w59_tmp_nobias_gru", ignore_errors=True)
save_quantized(qgru_nb, "py_w59_tmp_nobias_gru")
fresh = nn.GRU(8, 16, num_layers=1, bias=False)
loaded = load_quantized(fresh, "py_w59_tmp_nobias_gru")
got = to_np(cell_out(loaded, xn))
check("bias-free gru cell save load round-trip forward exactly equal", np.array_equal(got, ref))
check("bias-free gru loaded cell has no biases", loaded.cells[0].bias_ih is None and loaded.cells[0].bias_hh is None)
shutil.rmtree("py_w59_tmp_nobias_gru", ignore_errors=True)

print("== fake-quant calibrate include-0 (w76) ==")

def fixed_qparams(vals):
    mn = min(min(vals), 0.0)
    mx = max(max(vals), 0.0)
    scale = (mx - mn) / 254.0
    if scale == 0.0:
        scale = 1.0
    zp = round(-127.0 - mn / scale)
    zp = max(-127.0, min(127.0, zp))
    return scale, int(zp)

def ste_fake_quant(x, scale, zp):
    fq = torch.fake_quantize_per_tensor_affine(x, scale, zp, -127, 127)
    return x + (fq - x).detach()

W76 = [[0.5, 1.0, 1.5, 2.0], [0.25, 0.75, 1.25, 1.75], [0.1, 0.2, 0.3, 0.4], [2.0, 1.0, 0.5, 0.25]]
B76 = [0.1, -0.2, 0.3, -0.4]

def make_qat76():
    lin = nn.Linear(4, 4)
    lin.weight = lt.tensor(W76)
    lin.bias = lt.tensor(B76)
    qat = SchemeQATLinear(lin, "fake-quant")
    qat.train()
    return qat

def torch_qat76(x_list, w_scale, w_zp, a_scale, a_zp):
    w = torch.tensor(W76, requires_grad=True)
    x = torch.tensor(x_list)
    b = torch.tensor(B76)
    w_fq = ste_fake_quant(w, w_scale, w_zp)
    x_fq = ste_fake_quant(x, a_scale, a_zp)
    y = F.linear(x_fq, w_fq, b)
    y.sum().backward()
    return y.detach().numpy(), w.grad.numpy(), x_fq.detach().numpy()

w76_w_scale, w76_w_zp = fixed_qparams([v for row in W76 for v in row])

xc76 = lt.tensor([[0.5, 0.5, 0.5, 0.5]])
qatc76 = make_qat76()
pc76 = qatc76.parameters()
outc76 = qatc76(xc76)
yc76 = to_np(outc76)
check("qat constant input forward differs from bias", float(np.max(np.abs(yc76 - np.array(B76)))) > 0.1)
outc76.sum().backward()
gc76 = to_np(pc76[0].grad)
a76_scale, a76_zp = fixed_qparams([0.5, 0.5, 0.5, 0.5])
yt76, gt76, xq76 = torch_qat76([[0.5, 0.5, 0.5, 0.5]], w76_w_scale, w76_w_zp, a76_scale, a76_zp)
tol76 = max(w76_w_scale, a76_scale) * 1.5 + 1e-4
check("qat constant input forward matches torch fake-quant linear within 1 LSB", float(np.max(np.abs(yc76 - yt76))) <= tol76)
check("qat constant input weight grad is non-zero", bool(np.all(np.abs(gc76) > 0.1)))
check("qat constant input weight grad equals fake-quantized input", float(np.max(np.abs(gc76 - xq76))) < 1e-6)
check("qat constant input weight grad matches torch STE grad within 1 LSB", float(np.max(np.abs(gc76 - gt76))) <= tol76)

xv76 = lt.tensor([[0.1, 0.4, 0.7, 1.0]])
qatv76 = make_qat76()
pv76 = qatv76.parameters()
outv76 = qatv76(xv76)
yv76 = to_np(outv76)
outv76.sum().backward()
gv76 = to_np(pv76[0].grad)
av76_scale, av76_zp = fixed_qparams([0.1, 0.4, 0.7, 1.0])
ytv76, gtv76, xqv76 = torch_qat76([[0.1, 0.4, 0.7, 1.0]], w76_w_scale, w76_w_zp, av76_scale, av76_zp)
tolv76 = max(w76_w_scale, av76_scale) * 1.5 + 1e-4
check("qat varied single-signed forward matches torch fake-quant linear within 1 LSB", float(np.max(np.abs(yv76 - ytv76))) <= tolv76)
check("qat varied single-signed weight grad equals fake-quantized input", float(np.max(np.abs(gv76 - xqv76))) < 1e-6)
check("qat varied single-signed weight grad matches torch STE grad within 1 LSB", float(np.max(np.abs(gv76 - gtv76))) <= tolv76)

print(f"test_quant_scheme: {passed} passed, {failed} failed")
sys.exit(0 if failed == 0 else 1)
