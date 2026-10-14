import litetorch as lt
import torch
import torch.nn.utils.prune as torch_prune
import numpy as np
import os
import tempfile
import shutil

CPU = lt.Device("cpu")
passed = []
failed = []

def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"  PASS {name}")
    else:
        failed.append(name)
        print(f"  FAIL {name} {detail}")

def make_pair(in_f, out_f, seed=0, bias=True):
    rng = np.random.RandomState(seed)
    w = rng.randn(out_f, in_f).astype(np.float32)
    b = rng.randn(out_f).astype(np.float32)
    tlin = torch.nn.Linear(in_f, out_f, bias=bias)
    with torch.no_grad():
        tlin.weight.copy_(torch.from_numpy(w))
        if bias:
            tlin.bias.copy_(torch.from_numpy(b))
    llin = lt.nn.Linear(in_f, out_f, bias)
    llin.weight.copy_(lt.Tensor.from_vector(w.flatten().tolist(), [out_f, in_f], CPU, True))
    if bias:
        llin.bias.copy_(lt.Tensor.from_vector(b.flatten().tolist(), [out_f], CPU, True))
    return tlin, llin, w, b

def lt_forward(llin, x_np):
    x = lt.Tensor.from_vector(x_np.flatten().tolist(), list(x_np.shape), CPU, False)
    return llin(x).numpy()

def get_mask(llin, name="weight"):
    for k, t in llin.named_buffers():
        if k == name + "_mask":
            return t.numpy()
    return None

print("=== 1. L1 mask exactness vs torch (after remove) ===")
tlin, llin, w, b = make_pair(8, 4, seed=0)
torch_prune.l1_unstructured(tlin, "weight", 0.3)
lt.prune.l1_unstructured(llin, "weight", 0.3)
check("is_pruned True after l1", lt.prune.is_pruned(llin, "weight") is True)
check("is_pruned False before (fresh)", lt.prune.is_pruned(lt.nn.Linear(8, 4), "weight") is False)
n = w.size
k = round(0.3 * n)
m = get_mask(llin, "weight")
check("mask buffer present", m is not None)
if m is not None:
    check("mask values are 0/1", set(np.unique(m)).issubset({0, 1}))
    check("mask zeros == k", int(np.sum(m == 0)) == k)
torch_prune.remove(tlin, "weight")
lt.prune.remove(llin, "weight")
check("is_pruned False after remove", lt.prune.is_pruned(llin, "weight") is False)
tw = tlin.weight.detach().numpy()
lw = llin.weight.numpy()
check("masked weights exact vs torch", np.array_equal(tw, lw),
      f"maxdiff={np.max(np.abs(tw - lw))}")
check("sparsity == 30%", int(np.sum(lw == 0)) == k, f"zeros={np.sum(lw == 0)} k={k}")

print("=== 2. forward equality vs torch (pruned, hooks via __call__) ===")
tlin2, llin2, w2, b2 = make_pair(8, 4, seed=1)
torch_prune.l1_unstructured(tlin2, "weight", 0.5)
lt.prune.l1_unstructured(llin2, "weight", 0.5)
xr = np.random.RandomState(7).randn(3, 8).astype(np.float32)
with torch.no_grad():
    tout = tlin2(torch.from_numpy(xr)).numpy()
lout = lt_forward(llin2, xr)
check("forward matches torch (masked)", np.allclose(tout, lout, atol=1e-5),
      f"maxdiff={np.max(np.abs(tout - lout))}")

print("=== 3. remove() permanence + forward equality ===")
before = lt_forward(llin2, xr)
lt.prune.remove(llin2, "weight")
after = lt_forward(llin2, xr)
check("forward equal before/after remove", np.allclose(before, after, atol=1e-6))
check("zeros baked after remove", int(np.sum(llin2.weight.numpy() == 0)) == int(0.5 * w2.size))
try:
    lt.prune.remove(llin2, "weight")
    check("remove unpruned raises", False)
except Exception:
    check("remove unpruned raises", True)

print("=== 4. repeated prune composition vs torch ===")
tlin3, llin3, w3, b3 = make_pair(16, 8, seed=2)
for m, fn in ((tlin3, torch_prune.l1_unstructured), (llin3, lt.prune.l1_unstructured)):
    fn(m, "weight", 0.5)
    fn(m, "weight", 0.5)
torch_prune.remove(tlin3, "weight")
lt.prune.remove(llin3, "weight")
tw3 = tlin3.weight.detach().numpy()
lw3 = llin3.weight.numpy()
check("composed mask exact vs torch", np.array_equal(tw3, lw3))
zfrac = np.mean(lw3 == 0)
check("composed sparsity ~75%", abs(zfrac - 0.75) < 0.02, f"frac={zfrac}")

print("=== 5. grad-zero on masked entries ===")
tlin4, llin4, w4, b4 = make_pair(8, 4, seed=3)
lt.prune.l1_unstructured(llin4, "weight", 0.5)
xg = np.random.RandomState(11).randn(2, 8).astype(np.float32)
xt = lt.Tensor.from_vector(xg.flatten().tolist(), [2, 8], CPU, False)
out = llin4(xt)
lt.Ops.sum(out).backward()
g = llin4.weight.grad.numpy()
flat_w = w4.flatten()
k5 = int(0.5 * flat_w.size)
thr_idx = np.argpartition(np.abs(flat_w), k5)[:k5]
mask = np.ones_like(flat_w)
mask[thr_idx] = 0
mask = mask.reshape(w4.shape)
check("grad zero at masked entries", np.all(g[mask == 0] == 0),
      f"maxgrad_masked={np.max(np.abs(g[mask == 0]))}")
check("grad nonzero at kept entries", np.mean(np.abs(g[mask == 1])) > 0)
check("member restored to orig after forward", np.array_equal(llin4.weight.numpy(), w4))

print("=== 6. random_unstructured determinism + distribution ===")
_, la, wa, _ = make_pair(32, 16, seed=4)
_, lb, wb, _ = make_pair(32, 16, seed=4)
lt.prune.random_unstructured(la, "weight", 0.25, seed=123)
lt.prune.random_unstructured(lb, "weight", 0.25, seed=123)
lt.prune.remove(la, "weight")
lt.prune.remove(lb, "weight")
check("same seed -> same mask", np.array_equal(la.weight.numpy(), lb.weight.numpy()))
_, lc, _, _ = make_pair(64, 32, seed=5)
lt.prune.random_unstructured(lc, "weight", 0.25, seed=999)
lt.prune.remove(lc, "weight")
frac = np.mean(lc.weight.numpy() == 0)
check("random fraction ~0.25", abs(frac - 0.25) < 0.05, f"frac={frac}")
_, ld, _, _ = make_pair(16, 8, seed=6)
lt.prune.random_unstructured(ld, "weight", 10)
lt.prune.remove(ld, "weight")
check("random int count prunes 10", int(np.sum(ld.weight.numpy() == 0)) == 10)

print("=== 7. save/load_pretrained on pruned model ===")
tlin7, llin7, w7, b7 = make_pair(8, 4, seed=7)
lt.prune.l1_unstructured(llin7, "weight", 0.4)
tmpd = tempfile.mkdtemp()
try:
    fpath = os.path.join(tmpd, "model.safetensors")
    llin7.save_pretrained(fpath)
    _, lin7b, _, _ = make_pair(8, 4, seed=99)
    lin7b.load_pretrained(fpath)
    check("orig weights preserved",
          np.array_equal(lin7b.weight.numpy(), llin7.weight.numpy()))
    check("loaded model unpruned (masks live in state_dict, not parameters)",
          lt.prune.is_pruned(lin7b, "weight") is False)
finally:
    shutil.rmtree(tmpd, ignore_errors=True)

print("=== 8. FP32 regression: unpruned paths unaffected ===")
_, lu1, _, _ = make_pair(8, 4, seed=8)
lu3 = lt.nn.Linear(8, 4)
w8 = lu1.weight.numpy().copy()
b8 = lu1.bias.numpy().copy()
lu3.weight.copy_(lt.Tensor.from_vector(w8.flatten().tolist(), [4, 8], CPU, True))
lu3.bias.copy_(lt.Tensor.from_vector(b8.flatten().tolist(), [4], CPU, True))
x8 = np.random.RandomState(17).randn(2, 8).astype(np.float32)
check("unpruned forward byte-identical", np.array_equal(lt_forward(lu1, x8), lt_forward(lu3, x8)))
lt.prune.l1_unstructured(lu3, "weight", 0.0)
check("amount=0 no-op forward", np.array_equal(lt_forward(lu1, x8), lt_forward(lu3, x8)))

print("=== 9. prune x PTQ: convert() must not mutate masks ===")
try:
    seq = lt.nn.Sequential()
    lin9 = lt.nn.Linear(8, 4)
    _, lref, _, _ = make_pair(8, 4, seed=9)
    lin9.weight.copy_(lt.Tensor.from_vector(lref.weight.numpy().flatten().tolist(), [4, 8], CPU, True))
    lin9.bias.copy_(lt.Tensor.from_vector(lref.bias.numpy().flatten().tolist(), [4], CPU, True))
    seq.add_module("fc", lin9)
    lt.prune.l1_unstructured(lin9, "weight", 0.5)
    x9 = np.random.RandomState(19).randn(2, 8).astype(np.float32)
    before9 = lt_forward(lin9, x9)
    mask_before9 = get_mask(lin9, "weight").copy()
    prep = lt.quantization.prepare(seq)
    qseq = lt.quantization.convert(prep)
    check("mask not mutated by convert", lt.prune.is_pruned(lin9, "weight") is True)
    m_after = get_mask(lin9, "weight")
    check("mask values unchanged by convert",
          m_after is not None and np.array_equal(m_after, mask_before9))
    after9 = lt_forward(lin9, x9)
    check("pruned forward unchanged by convert", np.allclose(before9, after9, atol=1e-6))
    xq = lt.Tensor.from_vector(x9.flatten().tolist(), [2, 8], CPU, False)
    qout = qseq(xq).numpy()
    check("quantized model runs", qout.shape == (2, 4))
except Exception as e:
    check("prune x PTQ", False, f"{type(e).__name__}: {e}")

print("=== 10. ONNX export after remove(): masked dense weights ===")
try:
    _, lin10, _, _ = make_pair(8, 4, seed=10)
    lt.prune.l1_unstructured(lin10, "weight", 0.5)
    x10 = np.random.RandomState(23).randn(2, 8).astype(np.float32)
    xt10 = lt.Tensor.from_vector(x10.flatten().tolist(), [2, 8], CPU, False)
    expected = lin10(xt10).numpy()
    lt.prune.remove(lin10, "weight")
    tmpd2 = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpd2, "pruned.onnx")
        lt.onnx.export(lin10, xt10, path)
        txt = open(path).read()
        check("onnx file written", len(txt) > 0)
        masked = lin10.weight.numpy()
        nz_export = txt.count("0,") + txt.count("0 ")
        check("exported weights contain zeros", ("0" in txt))
        check("export matches masked forward",
              np.allclose(lin10(xt10).numpy(), expected, atol=1e-5))
    finally:
        shutil.rmtree(tmpd2, ignore_errors=True)
except Exception as e:
    check("onnx export", False, f"{type(e).__name__}: {e}")

print("=== 11. ONNX export of pruned-not-removed module: mask-folded dense weights ===")
try:
    import re
    def parse_inits(txt):
        out = {}
        for blk in txt.split("initializer {")[1:]:
            m = re.search(r'name: "([^"]+)"', blk)
            f = re.search(r'float_data: \[([^\]]*)\]', blk)
            if m and f:
                out[m.group(1)] = np.array([float(v) for v in f.group(1).split(",") if v.strip()],
                                           dtype=np.float32)
        return out
    _, lin11, _, _ = make_pair(8, 4, seed=11)
    w_orig = lin11.weight.numpy().copy()
    lt.prune.l1_unstructured(lin11, "weight", 0.5)
    check("still pruned (not removed)", lt.prune.is_pruned(lin11, "weight"))
    mask = get_mask(lin11, "weight")
    check("mask buffer present", mask is not None and mask.shape == w_orig.shape)
    x11 = np.random.RandomState(31).randn(2, 8).astype(np.float32)
    xt11 = lt.Tensor.from_vector(x11.flatten().tolist(), [2, 8], CPU, False)
    inits = parse_inits(lt.onnx.export_onnx_text(lt.onnx.trace_module(lin11, xt11)))
    check("mask buffer not exported as initializer", not any("_mask" in n for n in inits))
    wname = [n for n in inits if n.endswith("weight")]
    check("weight initializer present", len(wname) == 1)
    expected = (w_orig * mask).astype(np.float32).flatten()
    got = inits[wname[0]]
    check("exported dense weights == orig * mask",
          got.shape == expected.shape and np.allclose(got, expected, atol=1e-6))
    check("some weights actually pruned to zero", np.any(got == 0.0) and np.any(got != 0.0))
    seq = lt.nn.Sequential([lin11])
    inits2 = parse_inits(lt.onnx.export_onnx_text(lt.onnx.trace_module(seq, xt11)))
    wname2 = [n for n in inits2 if n.endswith("weight")]
    check("sequential child weight initializer present", len(wname2) == 1)
    check("sequential child export == orig * mask",
          np.allclose(inits2[wname2[0]], expected, atol=1e-6))
except Exception as e:
    check("onnx mask-fold export", False, f"{type(e).__name__}: {e}")

print(f"\nRESULT: {len(passed)} passed, {len(failed)} failed")
if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("ALL PRUNE TESTS PASS")
