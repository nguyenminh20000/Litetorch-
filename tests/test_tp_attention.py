import multiprocessing as mp
import sys
import time

import numpy as np

import litetorch as lt

PORT = 29621

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def ref_attn_np(x, wq, bq, wk, bk, wv, bv, wo, bo, H, KV, D):
    B, T, E = x.shape
    q = x @ wq.T + bq
    k = x @ wk.T + bk
    v = x @ wv.T + bv
    q4 = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k4 = k.reshape(B, T, KV, D).transpose(0, 2, 1, 3)
    v4 = v.reshape(B, T, KV, D).transpose(0, 2, 1, 3)
    rep = H // KV
    k4 = np.repeat(k4, rep, axis=1)
    v4 = np.repeat(v4, rep, axis=1)
    s = q4 @ k4.transpose(0, 1, 3, 2) / np.sqrt(D)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    c = p @ v4
    cf = c.transpose(0, 2, 1, 3).reshape(B, T, H * D)
    return cf @ wo.T + bo


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    ws = world_size

    E, H, KV, D = 8, 4, 2, 2
    B, T = 2, 4
    qh_r, kvh_r = H // ws, KV // ws

    wq_f = (np.arange(H * D * E).reshape(H * D, E) * 0.01).astype(np.float32)
    bq_f = (np.arange(H * D) * 0.01).astype(np.float32)
    wk_f = (np.arange(KV * D * E).reshape(KV * D, E) * 0.01 + 0.5).astype(np.float32)
    bk_f = (np.arange(KV * D) * 0.01 + 0.5).astype(np.float32)
    wv_f = (np.arange(KV * D * E).reshape(KV * D, E) * 0.01 + 1.0).astype(np.float32)
    bv_f = (np.arange(KV * D) * 0.01 + 1.0).astype(np.float32)
    wo_f = (np.arange(E * H * D).reshape(E, H * D) * 0.01 + 1.5).astype(np.float32)
    bo_f = (np.arange(E) * 0.01 + 1.5).astype(np.float32)
    x_np = (np.arange(B * T * E).reshape(B, T, E) * 0.01).astype(np.float32)

    def tw(mat, rg=True):
        return lt.Tensor.from_vector(mat.flatten().tolist(), list(mat.shape), cpu, rg)

    attn = lt.nn.TPAttention(E, H, KV, True)
    attn.w_q = tw(wq_f[rank * qh_r * D:(rank + 1) * qh_r * D])
    attn.b_q = tw(bq_f[rank * qh_r * D:(rank + 1) * qh_r * D])
    attn.w_k = tw(wk_f[rank * kvh_r * D:(rank + 1) * kvh_r * D])
    attn.b_k = tw(bk_f[rank * kvh_r * D:(rank + 1) * kvh_r * D])
    attn.w_v = tw(wv_f[rank * kvh_r * D:(rank + 1) * kvh_r * D])
    attn.b_v = tw(bv_f[rank * kvh_r * D:(rank + 1) * kvh_r * D])
    attn.w_o = tw(wo_f[:, rank * qh_r * D:(rank + 1) * qh_r * D])
    attn.b_o = tw(bo_f)
    x_t = tw(x_np, False)

    out_t = attn.forward(x_t)
    out_np = np.array(out_t.to_vector(), dtype=np.float32).reshape(B, T, E)
    ref = ref_attn_np(x_np, wq_f, bq_f, wk_f, bk_f, wv_f, bv_f, wo_f, bo_f, H, KV, D)
    r["gqa_fwd"] = bool(np.allclose(out_np, ref, atol=1e-4))

    H2 = 4
    attn2 = lt.nn.TPAttention(E, H2, H2, True)
    qh2 = H2 // ws
    attn2.w_q = tw(wq_f[rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.b_q = tw(bq_f[rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.w_k = tw(wq_f[rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.b_k = tw(bq_f[rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.w_v = tw(wq_f[rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.b_v = tw(bq_f[rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.w_o = tw(wo_f[:, rank * qh2 * D:(rank + 1) * qh2 * D])
    attn2.b_o = tw(bo_f)
    out2 = attn2.forward(x_t)
    out2_np = np.array(out2.to_vector(), dtype=np.float32).reshape(B, T, E)
    ref2 = ref_attn_np(x_np, wq_f, bq_f, wq_f, bq_f, wq_f, bq_f, wo_f, bo_f, H2, H2, D)
    r["mha_fwd"] = bool(np.allclose(out2_np, ref2, atol=1e-4))

    wqr = lt.Tensor.from_vector(wq_f.flatten().tolist(), [H2 * D, E], cpu, True)
    bqr = lt.Tensor.from_vector(bq_f.flatten().tolist(), [H2 * D], cpu, True)
    wkr = lt.Tensor.from_vector(wq_f.flatten().tolist(), [H2 * D, E], cpu, True)
    bkr = lt.Tensor.from_vector(bq_f.flatten().tolist(), [H2 * D], cpu, True)
    wvr = lt.Tensor.from_vector(wq_f.flatten().tolist(), [H2 * D, E], cpu, True)
    bvr = lt.Tensor.from_vector(bq_f.flatten().tolist(), [H2 * D], cpu, True)
    wor = lt.Tensor.from_vector(wo_f.flatten().tolist(), [E, H2 * D], cpu, True)
    bor = lt.Tensor.from_vector(bo_f.flatten().tolist(), [E], cpu, True)
    xr = lt.Tensor.from_vector(x_np.flatten().tolist(), [B, T, E], cpu, True)

    def rlin(xx, w, b):
        return lt.ops.add(lt.ops.matmul(xx, w.transpose(0, 1)), b.view(1, H2 * D))

    q = rlin(xr, wqr, bqr)
    k = rlin(xr, wkr, bkr)
    v = rlin(xr, wvr, bvr)
    q4 = q.view(B, T, H2, D).transpose(1, 2).reshape(B * H2, T, D)
    k4 = k.view(B, T, H2, D).transpose(1, 2).reshape(B * H2, T, D)
    v4 = v.view(B, T, H2, D).transpose(1, 2).reshape(B * H2, T, D)
    s = lt.ops.bmm(q4, k4.transpose(1, 2))
    s = lt.ops.mul(s, lt.Tensor.from_vector([1.0 / np.sqrt(D)], [1], cpu))
    p = lt.ops.softmax(s, -1)
    c = lt.ops.bmm(p, v4)
    cf = c.view(B, H2, T, D).transpose(1, 2).reshape(B, T, H2 * D)
    ro = lt.ops.add(lt.ops.matmul(cf, wor.transpose(0, 1)), bor.view(1, 1, E))
    loss_ref = ro.sum()
    loss_ref.backward()

    loss_tp = out2.sum()
    loss_tp.backward()
    r["loss_match"] = abs(loss_tp.item() - loss_ref.item()) < 1e-4

    def gslice(g, rows, r0):
        return np.array(g.to_vector(), dtype=np.float32).reshape(rows, -1)[r0:r0 + rows // ws]

    r["bwd_wq"] = bool(np.allclose(
        np.array(attn2.w_q.grad.to_vector(), dtype=np.float32).reshape(qh2 * D, E),
        np.array(wqr.grad.to_vector(), dtype=np.float32).reshape(H2 * D, E)[rank * qh2 * D:(rank + 1) * qh2 * D],
        atol=1e-4))
    r["bwd_wo"] = bool(np.allclose(
        np.array(attn2.w_o.grad.to_vector(), dtype=np.float32).reshape(E, qh2 * D),
        np.array(wor.grad.to_vector(), dtype=np.float32).reshape(E, H2 * D)[:, rank * qh2 * D:(rank + 1) * qh2 * D],
        atol=1e-4))
    r["bwd_bo"] = bool(np.allclose(
        np.array(attn2.b_o.grad.to_vector(), dtype=np.float32),
        np.array(bor.grad.to_vector(), dtype=np.float32), atol=1e-4))

    try:
        bad = lt.nn.TPAttention(8, 3, -1, True)
        r["tp_div_throws"] = False
    except Exception:
        r["tp_div_throws"] = True
    try:
        bad = lt.nn.TPAttention(8, 4, 3, True)
        r["kv_div_throws"] = False
    except Exception:
        r["kv_div_throws"] = True

    V, Dd = 8, 4
    Vr = V // ws
    emb = lt.nn.VocabParallelEmbedding(V, Dd)
    wfull_np = (np.arange(V * Dd).reshape(V, Dd) * 0.01).astype(np.float32)
    emb.weight = tw(wfull_np[rank * Vr:(rank + 1) * Vr])
    ids_np = np.array([[0, 3, 4, 7], [1, 2, 5, 6]], dtype=np.int32)
    ids = lt.Tensor.from_vector(ids_np.flatten().tolist(), [2, 4], cpu, False, lt.DataType.INT32)
    eout = emb.forward(ids)
    eout_np = np.array(eout.to_vector(), dtype=np.float32).reshape(2, 4, Dd)
    r["emb_fwd"] = bool(np.allclose(eout_np, wfull_np[ids_np], atol=1e-5))

    h_np = (np.arange(2 * 3 * Dd).reshape(2, 3, Dd) * 0.01).astype(np.float32)
    h = tw(h_np, False)
    lsh = emb.logits(h)
    lsh_np = np.array(lsh.to_vector(), dtype=np.float32).reshape(2, 3, Vr)
    r["logits_shard"] = bool(np.allclose(lsh_np, h_np @ wfull_np.T[:, rank * Vr:(rank + 1) * Vr], atol=1e-4))

    N = 3
    hn_np = (np.arange(N * Dd).reshape(N, Dd) * 0.02).astype(np.float32)
    hn = tw(hn_np, False)
    lflat = emb.logits(hn)
    tgt_np = np.array([0, 7, 3], dtype=np.int32)
    tgt = lt.Tensor.from_vector(tgt_np.tolist(), [N], cpu, False, lt.DataType.INT32)
    loss = emb.sharded_cross_entropy(lflat, tgt)

    wfull = lt.Tensor.from_vector(wfull_np.flatten().tolist(), [V, Dd], cpu, True)
    hnr = lt.Tensor.from_vector(hn_np.flatten().tolist(), [N, Dd], cpu, True)
    full_logits = lt.ops.matmul(hnr, wfull.transpose(0, 1))
    ref_loss = lt.ops.cross_entropy_loss(full_logits, tgt)
    r["ce_fwd"] = abs(loss.item() - ref_loss.item()) < 1e-4

    loss.backward()
    ref_loss.backward()
    r["ce_bwd"] = bool(np.allclose(
        np.array(emb.weight.grad.to_vector(), dtype=np.float32).reshape(Vr, Dd),
        np.array(wfull.grad.to_vector(), dtype=np.float32).reshape(V, Dd)[rank * Vr:(rank + 1) * Vr],
        atol=1e-4))

    try:
        bad = lt.nn.VocabParallelEmbedding(7, 4)
        r["vocab_div_throws"] = False
    except Exception:
        r["vocab_div_throws"] = True

    nb = lt.distributed.NCCLBridge.get()
    r["nccl_no_crash"] = True
    if nb.is_available():
        lt.distributed.init_tp_groups(2)
        r["tp_groups_init"] = True
    else:
        r["tp_groups_init"] = True

    lt.distributed.barrier()
    out[rank] = r


def main():
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        out = mgr.dict()
        procs = [ctx.Process(target=worker, args=(rank, 2, PORT, out)) for rank in (0, 1)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        failed_proc = any(p.exitcode != 0 for p in procs)
        res = dict(out)

    names = ["gqa_fwd", "mha_fwd", "loss_match", "bwd_wq", "bwd_wo", "bwd_bo",
             "tp_div_throws", "kv_div_throws",
             "emb_fwd", "logits_shard", "ce_fwd", "ce_bwd", "vocab_div_throws",
             "nccl_no_crash", "tp_groups_init"]
    for rank in (0, 1):
        rr = res.get(rank, {})
        for n in names:
            check(f"rank{rank} {n}", rr.get(n, False))
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    if failed_proc:
        print("WARNING: a worker process exited non-zero", flush=True)
        failed += 1
    print(f"== SUMMARY: {passed} PASS, {failed} FAIL in {time.perf_counter()-t0:.1f}s ==", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
