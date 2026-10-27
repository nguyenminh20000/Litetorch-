import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 r5-flash-attention-qkv FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook
import litetorch as lt
gpu = lt.Device("cuda")
qkv = lt.Tensor.from_vector([0.1 * (i % 6 + 1) for i in range(192)], [1, 4, 48], gpu, False)
out = lt.Ops.flash_attention_qkv(qkv, 2)
print(list(out.shape))
assert list(out.shape) == [1, 2, 4, 8]
try:
    import torch
    import torch.nn.functional as F
    qkv_t = torch.tensor([0.1 * (i % 6 + 1) for i in range(192)], dtype=torch.float32).reshape(1, 4, 48)
    q, k, v = qkv_t.split(16, dim=-1)
    q = q.reshape(1, 4, 2, 8).transpose(1, 2)
    k = k.reshape(1, 4, 2, 8).transpose(1, 2)
    v = v.reshape(1, 4, 2, 8).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out_t = torch.tensor(out.to_vector(), dtype=torch.float32).reshape(1, 2, 4, 8)
    diff = float((ref - out_t).abs().max())
    print("parity max_abs_diff", diff)
    assert diff <= 1e-4, "parity fail"
except ImportError:
    print("TORCH-ABSENT")
print("B5 r5-flash-attention-qkv PASS")
