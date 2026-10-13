import sys
sys.path.insert(0, "python")

import litetorch as lt
from litetorch import nn

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

def expect_throw(name, fn):
    try:
        fn()
        check(name, False)
    except Exception:
        check(name, True)

def max_abs_diff(a, b):
    import numpy as np
    return float(np.max(np.abs(np.array(a.numpy()) - np.array(b.numpy()))))

print("== fold_conv_bn numerics (CPU) ==")
conv = nn.Conv2d(3, 8, 3, padding=1)
bn = nn.BatchNorm2d(8)
bn.eval()
x = lt.randn([2, 3, 16, 16])
ref = bn.forward(conv.forward(x))
folded = lt.inference.fold_conv_bn(conv, bn)
out = folded.forward(x)
d = max_abs_diff(ref, out)
print(f"  fold max abs diff: {d}")
check("fold matches conv+bn", d < 1e-4)
check("folded has bias", folded.bias is not None)

print("== fold_conv_bn error cases ==")
expect_throw("null conv throws", lambda: lt.inference.fold_conv_bn(None, bn))
expect_throw("null bn throws", lambda: lt.inference.fold_conv_bn(conv, None))
bn_bad = nn.BatchNorm2d(4)
expect_throw("channel mismatch throws", lambda: lt.inference.fold_conv_bn(conv, bn_bad))

print("== fuse_conv_bn on Sequential ==")
seq = nn.Sequential([nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU()])
seq.eval()
x2 = lt.randn([1, 3, 8, 8])
ref2 = seq.forward(x2)
rep = lt.inference.fuse_conv_bn(seq)
check("one pair folded", rep.conv_bn_folded == 1)
out2 = seq.forward(x2)
d2 = max_abs_diff(ref2, out2)
print(f"  fused seq max abs diff: {d2}")
check("fused seq matches", d2 < 1e-4)
check("report notes present", len(rep.notes) == 1)

print("== detect_layernorm_fusion ==")
blk = nn.Sequential([nn.Linear(16, 16), nn.LayerNorm([16]), nn.Linear(16, 16), nn.LayerNorm([16])])
rep2 = lt.inference.detect_layernorm_fusion(blk)
check("detects 2 layernorms", rep2.layernorm_detected == 2)

print("== verify_linear_bias_fusion ==")
check("matmul_bias path ok", lt.inference.verify_linear_bias_fusion())

print("== CUDA graph fail-loud on CPU ==")
g = lt.inference.CudaGraph()
check("not captured initially", not g.captured())
expect_throw("replay without capture throws", lambda: g.replay())
expect_throw("capture on CPU throws", lambda: lt.inference.capture_cuda_graph(blk, x))
expect_throw("null model throws", lambda: lt.inference.capture_cuda_graph(None, x))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
