import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import litetorch as lt

PORT = 29671

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}", flush=True)


def expect_raise(name, fn):
    try:
        fn()
        check(name, False)
    except Exception:
        check(name, True)


def test_device_mesh():
    m = lt.distributed.DeviceMesh([2, 2, 2], "dp_pp_tp")
    check("mesh_num_dims", m.num_dims() == 3)
    check("mesh_numel", m.numel() == 8)
    check("mesh_dim_size", m.dim_size(0) == 2 and m.dim_size(2) == 2)

    ok = True
    for r in range(8):
        c = m.get_coords(r)
        if m.get_rank(c) != r:
            ok = False
    check("mesh_rank_coords_roundtrip", ok)
    check("mesh_get_rank_known", m.get_rank([1, 0, 1]) == 5)
    check("mesh_get_coords_known", list(m.get_coords(5)) == [1, 0, 1])

    g = m.get_dim_group(2, [1, 0, 0])
    check("mesh_dim_group_tp", list(g) == [4, 5])
    g = m.get_dim_group(0, [0, 1, 1])
    check("mesh_dim_group_dp", list(g) == [3, 7])
    g = m.get_dim_group(1, [1, 0, 1])
    check("mesh_dim_group_pp", list(g) == [5, 7])

    m1 = lt.distributed.DeviceMesh([4], "dp")
    check("mesh_1d_roundtrip", all(m1.get_rank(m1.get_coords(r)) == r for r in range(4)))

    expect_raise("mesh_bad_shape", lambda: lt.distributed.DeviceMesh([2, 0], "x"))
    expect_raise("mesh_bad_coords", lambda: m.get_rank([2, 0, 0]))
    expect_raise("mesh_bad_rank", lambda: m.get_coords(8))
    expect_raise("mesh_bad_dim", lambda: m.get_dim_group(3, [0, 0, 0]))


def brute_groups(rank, world, tp, pp):
    tp_g = [r for r in range(world) if r // tp == rank // tp]
    dp_g = [r for r in range(world) if r % (tp * pp) == rank % (tp * pp)]
    pp_g = [r for r in range(world)
            if (r // (tp * pp)) * tp + r % tp == (rank // (tp * pp)) * tp + rank % tp]
    return tp_g, pp_g, dp_g


def test_3d_mapping():
    p3 = lt.distributed.Parallel3D.get()
    cases = [(1, 1, 1), (2, 1, 2), (1, 2, 2), (2, 2, 4), (4, 2, 8), (2, 1, 4)]
    allok = True
    for tp, pp, world in cases:
        dp = world // (tp * pp)
        for rank in range(world):
            p3.init(rank, world, tp, pp)
            er_tp, er_pp, er_dp = rank % tp, (rank // tp) % pp, rank // (tp * pp)
            bt, bp, bd = brute_groups(rank, world, tp, pp)
            ok = (p3.tp_rank() == er_tp and p3.pp_rank() == er_pp and p3.dp_rank() == er_dp
                  and list(p3.tp_group()) == bt and list(p3.pp_group()) == bp
                  and list(p3.dp_group()) == bd
                  and p3.tp_size() == tp and p3.pp_size() == pp and p3.dp_size() == dp
                  and list(p3.coords()) == [er_dp, er_pp, er_tp]
                  and p3.global_rank() == rank and p3.world_size() == world)
            if not ok:
                allok = False
                print(f"  mismatch tp={tp} pp={pp} world={world} rank={rank}", flush=True)
    p3.reset()
    check("3d_mapping_all_cases", allok)
    check("3d_reset", not p3.is_initialized())

    expect_raise("3d_bad_world", lambda: p3.init(0, 3, 2, 1))
    expect_raise("3d_bad_tp", lambda: p3.init(0, 4, 0, 1))
    expect_raise("3d_bad_rank", lambda: p3.init(4, 4, 2, 1))
    expect_raise("3d_uninit_access", lambda: p3.tp_rank())


def test_3d_single_proc():
    lt.distributed.init_process_group(0, 1, "127.0.0.1", PORT)
    try:
        lt.distributed.init_3d_parallel(1, 1)
        p3 = lt.distributed.Parallel3D.get()
        check("3d_1proc_init", p3.is_initialized())
        check("3d_1proc_groups", list(p3.tp_group()) == [0] and list(p3.pp_group()) == [0]
              and list(p3.dp_group()) == [0])
        check("3d_1proc_ranks", p3.tp_rank() == 0 and p3.pp_rank() == 0 and p3.dp_rank() == 0)
        expect_raise("3d_1proc_bad_config", lambda: lt.distributed.init_3d_parallel(2, 1))
        lt.distributed.shutdown_3d_parallel()
        check("3d_1proc_shutdown", not p3.is_initialized())
    finally:
        lt.distributed.shutdown()


if __name__ == "__main__":
    test_device_mesh()
    test_3d_mapping()
    test_3d_single_proc()
    fails = [n for n, ok in results if not ok]
    print(f"\nTOTAL: {len(results) - len(fails)}/{len(results)} passed", flush=True)
    sys.exit(1 if fails else 0)
