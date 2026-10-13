import os
import sys
import multiprocessing as mp

import litetorch as lt

PORT = 29671
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def allclose(a, b, tol=1e-5):
    va, vb = a.to_vector(), b.to_vector()
    if len(va) != len(vb):
        return False
    return all(abs(x - y) <= tol + tol * abs(y) for x, y in zip(va, vb))


def worker(rank, world_size, port, out):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    r = {}
    try:
        r["pg_init"] = lt.distributed.ProcessGroup.get().is_initialized()

        t = lt.Tensor.from_vector([float(rank + 1)], [1], cpu)
        y = lt.distributed.tp_all_reduce(t)
        r["tp_ar_fwd"] = y.to_vector() == [3.0]

        x = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
        x.requires_grad = True
        y = lt.distributed.tp_all_reduce(x)
        r["tp_ar_fwd2"] = y.to_vector() == [2.0, 4.0]
        loss = y.sum()
        loss.backward()
        r["tp_ar_bwd"] = x.grad.to_vector() == [1.0, 1.0]

        shard = lt.Tensor.from_vector([float(rank * 2 + 1), float(rank * 2 + 2)], [2], cpu)
        full = lt.distributed.tp_all_gather(shard)
        r["tp_ag_fwd"] = full.to_vector() == [1.0, 2.0, 3.0, 4.0]

        wref = [0.1 * (i + 1) for i in range(64)]
        bref = [0.01 * (i + 1) for i in range(8)]
        xref = [0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 2.0, -2.0]
        w_shard = wref[rank * 32:(rank + 1) * 32]
        b_shard = bref[rank * 4:(rank + 1) * 4]

        ref = lt.nn.Linear(8, 8, True)
        ref.weight.copy_(lt.Tensor.from_vector(wref, [8, 8], cpu))
        ref.bias.copy_(lt.Tensor.from_vector(bref, [8], cpu))
        x_ref = lt.Tensor.from_vector(xref, [1, 8], cpu)
        x_ref.requires_grad = True
        y_ref = ref.forward(x_ref)

        col = lt.nn.ColumnParallelLinear(8, 8, True)
        col.weight.copy_(lt.Tensor.from_vector(w_shard, [4, 8], cpu))
        col.bias.copy_(lt.Tensor.from_vector(b_shard, [4], cpu))
        x_col = lt.Tensor.from_vector(xref, [1, 8], cpu)
        x_col.requires_grad = True
        y_shard = col.forward(x_col)
        y_col_full = lt.distributed.tp_all_gather(y_shard)
        r["col_fwd"] = allclose(y_col_full, y_ref)

        loss_ref = y_ref.sum()
        loss_ref.backward()
        loss_col = y_shard.sum()
        loss_col.backward()
        r["col_wgrad"] = allclose(col.weight.grad, narrow_ref_grad(ref.weight.grad, rank))
        r["col_bgrad"] = allclose(col.bias.grad, narrow_ref_bgrad(ref.bias.grad, rank))
        r["col_xgrad"] = allclose(x_col.grad, x_ref.grad)

        x_shard_vals = xref[rank * 4:(rank + 1) * 4]
        row = lt.nn.RowParallelLinear(8, 8, True)
        row.weight.copy_(lt.Tensor.from_vector(col_shard(wref, rank), [8, 4], cpu))
        row.bias.copy_(lt.Tensor.from_vector(bref, [8], cpu))
        x_row = lt.Tensor.from_vector(x_shard_vals, [1, 4], cpu)
        x_row.requires_grad = True
        y_row = row.forward(x_row)
        r["row_fwd"] = allclose(y_row, y_ref)

        ref2 = lt.nn.Linear(8, 8, True)
        ref2.weight.copy_(lt.Tensor.from_vector(wref, [8, 8], cpu))
        ref2.bias.copy_(lt.Tensor.from_vector(bref, [8], cpu))
        xr2 = lt.Tensor.from_vector(x_shard_vals, [1, 4], cpu)
        xr2.requires_grad = True
        x_full = lt.Tensor.from_vector(xref, [1, 8], cpu)
        x_full.requires_grad = True
        y_ref2 = ref2.forward(x_full)
        loss_ref2 = y_ref2.sum()
        loss_ref2.backward()
        loss_row = y_row.sum()
        loss_row.backward()
        r["row_wgrad"] = allclose(row.weight.grad, narrow_ref_grad_row(ref2.weight.grad, rank))
        r["row_xgrad"] = allclose(x_row.grad, narrow_ref_xgrad(x_full.grad, rank))

        l1 = lt.nn.Linear(8, 8, True)
        l1.weight.copy_(lt.Tensor.from_vector(wref, [8, 8], cpu))
        l1.bias.copy_(lt.Tensor.from_vector(bref, [8], cpu))
        l2 = lt.nn.Linear(8, 8, True)
        l2.weight.copy_(lt.Tensor.from_vector(wref, [8, 8], cpu))
        l2.bias.copy_(lt.Tensor.from_vector(bref, [8], cpu))
        xm = lt.Tensor.from_vector(xref, [1, 8], cpu)
        xm.requires_grad = True
        h_ref = lt.Ops.gelu(l1.forward(xm))
        y_mlp_ref = l2.forward(h_ref)
        y_mlp_ref.sum().backward()

        c1 = lt.nn.ColumnParallelLinear(8, 8, True)
        c1.weight.copy_(lt.Tensor.from_vector(wref[rank * 32:(rank + 1) * 32], [4, 8], cpu))
        c1.bias.copy_(lt.Tensor.from_vector(b_shard, [4], cpu))
        r1 = lt.nn.RowParallelLinear(8, 8, True)
        r1.weight.copy_(lt.Tensor.from_vector(col_shard(wref, rank), [8, 4], cpu))
        r1.bias.copy_(lt.Tensor.from_vector(bref, [8], cpu))
        x_mlp = lt.Tensor.from_vector(xref, [1, 8], cpu)
        x_mlp.requires_grad = True
        h_shard = c1.forward(x_mlp)
        a_shard = lt.Ops.gelu(h_shard)
        y_mlp = r1.forward(a_shard)
        r["mlp_fwd"] = allclose(y_mlp, y_mlp_ref)
        y_mlp.sum().backward()
        r["mlp_xgrad"] = allclose(x_mlp.grad, xm.grad)

        lt.distributed.barrier()
    except Exception as e:
        r["exception"] = f"{type(e).__name__}: {e}"
        import traceback
        traceback.print_exc()
    out[rank] = r


def col_shard(wref, rank):
    out = []
    for i in range(8):
        out.extend(wref[i * 8 + rank * 4:i * 8 + (rank + 1) * 4])
    return out


def narrow_ref_grad(g, rank):
    v = g.to_vector()
    return lt.Tensor.from_vector(v[rank * 32:(rank + 1) * 32], [4, 8], lt.Device("cpu"))


def narrow_ref_bgrad(g, rank):
    v = g.to_vector()
    return lt.Tensor.from_vector(v[rank * 4:(rank + 1) * 4], [4], lt.Device("cpu"))


def narrow_ref_grad_row(g, rank):
    v = g.to_vector()
    out = []
    for i in range(8):
        out.extend(v[i * 8 + rank * 4:i * 8 + (rank + 1) * 4])
    return lt.Tensor.from_vector(out, [8, 4], lt.Device("cpu"))


def narrow_ref_xgrad(g, rank):
    v = g.to_vector()
    return lt.Tensor.from_vector(v[rank * 4:(rank + 1) * 4], [1, 4], lt.Device("cpu"))


def main():
    ctx = mp.get_context("spawn")
    with mp.Manager() as mgr:
        out = mgr.dict()
        procs = [ctx.Process(target=worker, args=(rk, 2, PORT, out)) for rk in (0, 1)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(120)
        alive = [p.is_alive() for p in procs]
        for p in procs:
            if p.is_alive():
                p.terminate()
        if any(alive):
            print("FAIL: worker hang", flush=True)
            sys.exit(1)
        r0, r1 = out[0], out[1]
        keys = sorted(set(r0) | set(r1))
        npass, nfail = 0, 0
        for k in keys:
            v0, v1 = r0.get(k), r1.get(k)
            ok = (v0 is True) and (v1 is True)
            if k == "exception":
                print(f"rank0: {v0}\nrank1: {v1}", flush=True)
                ok = False
            print(f"[{'PASS' if ok else 'FAIL'}] {k} (r0={v0} r1={v1})", flush=True)
            npass, nfail = npass + (1 if ok else 0), nfail + (0 if ok else 1)
        print(f"TOTAL: {npass} passed, {nfail} failed", flush=True)
        sys.exit(0 if nfail == 0 else 1)


if __name__ == "__main__":
    main()
