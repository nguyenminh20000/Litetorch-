import litetorch as lt
from litetorch import nn, Ops

print("=== fused_swiglu forward vs unfused ===")
lt.manual_seed(0)
gate = lt.randn([2, 8, 128], lt.Device("cpu"), False)
up = lt.randn([2, 8, 128], lt.Device("cpu"), False)
fused = Ops.fused_swiglu(gate, up)
ref = Ops.mul(Ops.mul(gate, Ops.sigmoid(gate)), up)
fv = fused.to_vector()
rv = ref.to_vector()
max_err = max(abs(a - b) for a, b in zip(fv, rv))
print(f"max forward err: {max_err:.2e}")
assert max_err < 1e-5, "fused_swiglu forward mismatch"
print("forward OK")

print("\n=== fused_swiglu backward vs unfused ===")
g2 = lt.randn([4, 32], lt.Device("cpu"), True)
u2 = lt.randn([4, 32], lt.Device("cpu"), True)
out_f = Ops.fused_swiglu(g2, u2)
Ops.sum(out_f).backward()
gg_f = g2.grad.to_vector()
gu_f = u2.grad.to_vector()

g3 = lt.Tensor.from_vector(g2.to_vector(), [4, 32], lt.Device("cpu"), True)
u3 = lt.Tensor.from_vector(u2.to_vector(), [4, 32], lt.Device("cpu"), True)
out_r = Ops.mul(Ops.mul(g3, Ops.sigmoid(g3)), u3)
Ops.sum(out_r).backward()
gg_r = g3.grad.to_vector()
gu_r = u3.grad.to_vector()
max_egg = max(abs(a - b) for a, b in zip(gg_f, gg_r))
max_egu = max(abs(a - b) for a, b in zip(gu_f, gu_r))
print(f"max grad_gate err: {max_egg:.2e}, max grad_up err: {max_egu:.2e}")
assert max_egg < 1e-4 and max_egu < 1e-4, "fused_swiglu backward mismatch"
print("backward OK")

print("\n=== fused_swiglu shape mismatch raises ===")
try:
    Ops.fused_swiglu(lt.randn([4, 8], lt.Device("cpu"), False), lt.randn([4, 16], lt.Device("cpu"), False))
    raise AssertionError("should have raised")
except Exception as e:
    assert "shape" in str(e).lower(), f"unexpected error: {e}"
print("shape check OK")

print("\n=== fused_qkv_proj forward vs 3x matmul ===")
lt.manual_seed(1)
B, T, Dm = 2, 8, 64
Dq, Dk, Dv = 32, 16, 16
x = lt.randn([B, T, Dm], lt.Device("cpu"), False)
wq = lt.randn([Dq, Dm], lt.Device("cpu"), False)
wk = lt.randn([Dk, Dm], lt.Device("cpu"), False)
wv = lt.randn([Dv, Dm], lt.Device("cpu"), False)
q_f, k_f, v_f = Ops.fused_qkv_proj(x, wq, wk, wv)
q_r = Ops.matmul(x, Ops.t(wq))
k_r = Ops.matmul(x, Ops.t(wk))
v_r = Ops.matmul(x, Ops.t(wv))
for name, f, r in [("q", q_f, q_r), ("k", k_f, k_r), ("v", v_f, v_r)]:
    assert list(f.shape) == list(r.shape), f"{name} shape mismatch"
    err = max(abs(a - b) for a, b in zip(f.to_vector(), r.to_vector()))
    print(f"{name}: max err {err:.2e}")
    assert err < 1e-4, f"fused_qkv {name} mismatch"
print("qkv forward OK")

print("\n=== fused_qkv_proj backward ===")
xf = lt.Tensor.from_vector(x.to_vector(), [B, T, Dm], lt.Device("cpu"), True)
wqf = lt.Tensor.from_vector(wq.to_vector(), [Dq, Dm], lt.Device("cpu"), True)
wkf = lt.Tensor.from_vector(wk.to_vector(), [Dk, Dm], lt.Device("cpu"), True)
wvf = lt.Tensor.from_vector(wv.to_vector(), [Dv, Dm], lt.Device("cpu"), True)
qf, kf, vf = Ops.fused_qkv_proj(xf, wqf, wkf, wvf)
Ops.add(Ops.add(Ops.sum(qf), Ops.sum(kf)), Ops.sum(vf)).backward()
assert xf.grad is not None and wqf.grad is not None, "missing grads"
assert wkf.grad is not None and wvf.grad is not None, "missing grads"
xr = lt.Tensor.from_vector(x.to_vector(), [B, T, Dm], lt.Device("cpu"), True)
wqr = lt.Tensor.from_vector(wq.to_vector(), [Dq, Dm], lt.Device("cpu"), True)
wkr = lt.Tensor.from_vector(wk.to_vector(), [Dk, Dm], lt.Device("cpu"), True)
wvr = lt.Tensor.from_vector(wv.to_vector(), [Dv, Dm], lt.Device("cpu"), True)
qr = Ops.matmul(xr, Ops.t(wqr))
kr = Ops.matmul(xr, Ops.t(wkr))
vr = Ops.matmul(xr, Ops.t(wvr))
Ops.add(Ops.add(Ops.sum(qr), Ops.sum(kr)), Ops.sum(vr)).backward()
for name, tf, tr in [("x", xf, xr), ("wq", wqf, wqr), ("wk", wkf, wkr), ("wv", wvf, wvr)]:
    err = max(abs(a - b) for a, b in zip(tf.grad.to_vector(), tr.grad.to_vector()))
    print(f"grad_{name}: max err {err:.2e}")
    assert err < 1e-3, f"fused_qkv grad_{name} mismatch"
print("qkv backward OK")

print("\n=== GPTBlock fused path forward+backward ===")
lt.manual_seed(2)
block = nn.GPTBlock(d_model=64, n_q_heads=4, n_kv_heads=2, d_ff=128)
xb = lt.randn([2, 8, 64], lt.Device("cpu"), True)
outb = block.forward(xb)
assert list(outb.shape) == [2, 8, 64], "GPTBlock shape mismatch"
Ops.sum(outb).backward()
n_grad = sum(1 for p in block.parameters() if p.grad is not None)
print(f"params with grad: {n_grad}/{len(block.parameters())}")
assert n_grad == len(block.parameters()), "GPTBlock missing grads"
assert all(abs(v) > 0 for p in block.parameters() for v in p.grad.to_vector()[:4]), "zero grads suspicious"
print("GPTBlock fused OK")

print("\nALL FUSED KERNEL TESTS PASSED")
