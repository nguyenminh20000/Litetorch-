import math
import random

import litetorch as lt
from litetorch import nn

CPU = lt.Device("cpu")
DEV = lt.Device("cuda:0") if lt.is_gpu_available() else CPU
_sync = getattr(lt, "cuda_synchronize", lambda: None)

VOCAB = 128
D_MODEL = 64
N_LAYERS = 2
N_QH = 4
N_KVH = 2
D_FF = 128
BATCH = 2
SEQ = 16


def rand_tokens(n, vocab):
    return [float(random.randrange(vocab)) for _ in range(n)]


def scalar(t):
    _sync()
    return t.to(CPU).to_vector()[0]


def test_param_count():
    print("=== Param count ===")
    m = nn.GPT2B(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_q_heads=N_QH, n_kv_heads=N_KVH, d_ff=D_FF)
    params = m.parameters()
    total = sum(p.numel() for p in params)
    head_dim = D_MODEL // N_QH
    attn = D_MODEL * N_QH * head_dim + 2 * D_MODEL * N_KVH * head_dim + D_MODEL * D_MODEL
    ffn = 3 * D_MODEL * D_FF
    norms = 2 * head_dim + 2 * D_MODEL
    per_block = attn + ffn + norms
    expected = VOCAB * D_MODEL + N_LAYERS * per_block + D_MODEL
    print(f"total={total} expected={expected}")
    assert total == expected, f"param mismatch {total} vs {expected}"
    print("Param count OK")


def test_tied_weights():
    print("=== Weight tying ===")
    m = nn.GPT2B(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_q_heads=N_QH, n_kv_heads=N_KVH, d_ff=D_FF)
    assert m.lm_head.weight is m.tok_emb.weight, "lm_head not tied to embedding"
    assert m.tied is True
    print("Tied OK")


def test_forward_shape():
    print("=== Forward shape ===")
    m = nn.GPT2B(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_q_heads=N_QH, n_kv_heads=N_KVH, d_ff=D_FF)
    m.to(DEV)
    toks = lt.Tensor.from_vector(rand_tokens(BATCH * SEQ, VOCAB), [BATCH, SEQ], DEV, False)
    out = m.forward(toks)
    _sync()
    assert list(out.shape) == [BATCH, SEQ, VOCAB], f"shape {out.shape}"
    v = out.to(CPU).to_vector()
    assert all(math.isfinite(x) for x in v), "logits not finite"
    print(f"Forward OK: {list(out.shape)}")


def test_pipeline_split():
    print("=== Pipeline split ===")
    m = nn.GPT2B(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_q_heads=N_QH, n_kv_heads=N_KVH, d_ff=D_FF)
    m.to(DEV)
    d0 = DEV
    d1 = DEV
    m.pipeline_to(d0, d1, split=1)
    assert m.n_stages == 2 and m.stage_split == 1
    toks = lt.Tensor.from_vector(rand_tokens(BATCH * SEQ, VOCAB), [BATCH, SEQ], d0, False)
    h0 = m.stage_forward(toks, 0)
    _sync()
    assert list(h0.shape) == [BATCH, SEQ, D_MODEL], f"stage0 shape {h0.shape}"
    logits = m.forward(toks)
    _sync()
    assert list(logits.shape) == [BATCH, SEQ, VOCAB], f"logits shape {logits.shape}"
    v = logits.to(CPU).to_vector()
    assert all(math.isfinite(x) for x in v), "pipeline logits not finite"
    m.to(DEV)
    assert m.n_stages == 1
    print("Pipeline split OK")


def test_train_loss_decreases():
    print("=== Train loss decreases ===")
    random.seed(0)
    lt.manual_seed(0)
    m = nn.GPT2B(vocab_size=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_q_heads=N_QH, n_kv_heads=N_KVH, d_ff=D_FF)
    m.to(DEV)
    m.lm_head.weight = lt.Tensor.randn([VOCAB, D_MODEL], DEV, True)
    m.tied = False
    params = m.parameters()
    opt = lt.optim.Adam(params, lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    toks = lt.Tensor.from_vector(rand_tokens(BATCH * SEQ, VOCAB), [BATCH, SEQ], DEV, False)
    losses = []
    for step in range(10):
        opt.zero_grad()
        logits = m.forward(toks)
        inp = logits.view([BATCH * SEQ, VOCAB])
        tgt = toks.view([BATCH * SEQ])
        loss = loss_fn.forward(inp, tgt)
        lv = scalar(loss)
        assert math.isfinite(lv), f"loss not finite at step {step}"
        losses.append(lv)
        loss.backward()
        for p in params:
            assert p.grad is not None, "grad is None"
        opt.step()
    print(f"losses: {[f'{x:.4f}' for x in losses]}")
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"Loss decreases OK: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    test_param_count()
    test_tied_weights()
    test_forward_shape()
    test_pipeline_split()
    test_train_loss_decreases()
    print("\nALL GPT2B TESTS PASSED")
