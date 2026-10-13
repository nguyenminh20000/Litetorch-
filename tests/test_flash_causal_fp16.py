import sys
sys.path.insert(0, "python")
import math
import litetorch as lt
ops = lt.Ops

def manual_causal_attention(q, k, v):
    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)
    qv = q.to_vector()
    kv = k.to_vector()
    vv = v.to_vector()
    out = []
    for b in range(B):
        for h in range(H):
            for i in range(T):
                scores = []
                for j in range(T):
                    if j > i:
                        scores.append(float('-inf'))
                    else:
                        dot = sum(qv[((b * H + h) * T + i) * D + d] * kv[((b * H + h) * T + j) * D + d] for d in range(D))
                        scores.append(dot * scale)
                mx = max(s for s in scores if s != float('-inf'))
                exps = [math.exp(s - mx) if s != float('-inf') else 0.0 for s in scores]
                denom = sum(exps)
                for d in range(D):
                    val = sum(exps[j] * vv[((b * H + h) * T + j) * D + d] for j in range(T)) / denom
                    out.append(val)
    return out

def test_causal_fp32():
    d = lt.Device('cuda:0')
    B, H, T, Dd = 1, 2, 8, 16
    q = lt.Tensor.randn([B, H, T, Dd], d, True)
    k = lt.Tensor.randn([B, H, T, Dd], d, True)
    v = lt.Tensor.randn([B, H, T, Dd], d, True)
    out = ops.flash_attention(q, k, v, causal=True)
    ref = manual_causal_attention(q, k, v)
    got = out.to_vector()
    maxdiff = max(abs(a - b) for a, b in zip(got, ref))
    print(f"[{'PASS' if maxdiff < 1e-4 else 'FAIL'}] causal fp32 fwd maxdiff={maxdiff:.2e}")

def test_causal_fp16():
    d = lt.Device('cuda:0')
    B, H, T, Dd = 1, 2, 8, 16
    q = lt.Tensor.randn([B, H, T, Dd], d, True).cast(lt.DataType.FP16)
    k = lt.Tensor.randn([B, H, T, Dd], d, True).cast(lt.DataType.FP16)
    v = lt.Tensor.randn([B, H, T, Dd], d, True).cast(lt.DataType.FP16)
    out = ops.flash_attention(q, k, v, causal=True)
    assert out.dtype == lt.DataType.FP16, f"dtype {out.dtype}"
    got = out.cast(lt.DataType.FP32).to_vector()
    ref = manual_causal_attention(q.cast(lt.DataType.FP32), k.cast(lt.DataType.FP32), v.cast(lt.DataType.FP32))
    maxdiff = max(abs(a - b) for a, b in zip(got, ref))
    print(f"[{'PASS' if maxdiff < 1e-2 else 'FAIL'}] causal fp16 fwd maxdiff={maxdiff:.2e}")

def test_causal_bwd():
    d = lt.Device('cuda:0')
    B, H, T, Dd = 1, 1, 4, 8
    q = lt.Tensor.randn([B, H, T, Dd], d, True)
    k = lt.Tensor.randn([B, H, T, Dd], d, True)
    v = lt.Tensor.randn([B, H, T, Dd], d, True)
    out = ops.flash_attention(q, k, v, causal=True)
    loss = out.sum()
    loss.backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    qg = q.grad.to_vector()
    assert all(math.isfinite(x) for x in qg), "non-finite grad"
    print(f"[PASS] causal bwd grads finite, q.grad[0]={qg[0]:.4f}")

def test_qkv_fp16_causal():
    d = lt.Device('cuda:0')
    B, T, Nh, Dd = 1, 8, 2, 16
    C = Nh * Dd
    qkv = lt.Tensor.randn([B, T, 3 * C], d, True).cast(lt.DataType.FP16)
    out = ops.flash_attention_qkv(qkv, Nh, causal=True)
    assert out is not None
    print(f"[PASS] qkv fp16 causal fwd ok, shape={out.shape}")

if __name__ == '__main__':
    test_causal_fp32()
    test_causal_fp16()
    test_causal_bwd()
    test_qkv_fp16_causal()
    print("done")
