import os
import tempfile
import litetorch as lt
from litetorch import Ops

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond

def expect_raises(name, fn):
    try:
        fn()
    except RuntimeError as e:
        return check(name, True, f"raised: {str(e)[:60]}")
    except Exception as e:
        return check(name, False, f"wrong exception: {type(e).__name__}: {e}")
    return check(name, False, "no exception raised")

def make_tensors():
    return {
        "w": lt.randn([64, 32]),
        "b": lt.randn([32]),
    }

def test_basic_roundtrip(tmp):
    ok = True
    path = os.path.join(tmp, "ckpt_basic.safetensors")
    saver = lt.AsyncCheckpointSaver()
    ts = make_tensors()
    w_ref = ts["w"].clone()
    saver.save_async(ts, path)
    saver.wait()
    ok &= check("basic: file written", os.path.exists(path))
    loaded = lt.load_safetensors(path)
    ok &= check("basic: keys match", set(loaded.keys()) == {"w", "b"})
    d = Ops.abs((loaded["w"] - w_ref)).max().item()
    ok &= check("basic: values exact", d == 0.0, f"maxdiff={d}")
    return ok

def test_snapshot_isolation(tmp):
    ok = True
    path = os.path.join(tmp, "ckpt_snap.safetensors")
    saver = lt.AsyncCheckpointSaver()
    ts = make_tensors()
    w_ref = ts["w"].clone()
    saver.save_async(ts, path)
    lt.nn.init.fill_(ts["w"], 0.0)
    lt.nn.init.fill_(ts["b"], 0.0)
    saver.wait()
    loaded = lt.load_safetensors(path)
    d = Ops.abs((loaded["w"] - w_ref)).max().item()
    ok &= check("snapshot: staged copy immune to post-save mutation", d == 0.0, f"maxdiff={d}")
    return ok

def test_max_one_inflight(tmp):
    ok = True
    path1 = os.path.join(tmp, "ckpt_f1.safetensors")
    path2 = os.path.join(tmp, "ckpt_f2.safetensors")
    saver = lt.AsyncCheckpointSaver()
    big = {"big": lt.randn([4000, 4000])}
    saver.save_async(big, path1)
    ok &= check("inflight: is_saving true right after save_async", saver.is_saving())
    ok &= expect_raises("inflight: second save_async rejected", lambda: saver.save_async(big, path2))
    saver.wait()
    ok &= check("inflight: is_saving false after wait", not saver.is_saving())
    ok &= check("inflight: first file written", os.path.exists(path1))
    ok &= check("inflight: rejected save left no file", not os.path.exists(path2))
    return ok

def test_double_buffer_reuse(tmp):
    ok = True
    path = os.path.join(tmp, "ckpt_reuse.safetensors")
    saver = lt.AsyncCheckpointSaver()
    ts = make_tensors()
    saver.save_async(ts, path)
    saver.wait()
    lt.nn.init.fill_(ts["w"], 7.0)
    saver.save_async(ts, path)
    saver.wait()
    loaded = lt.load_safetensors(path)
    d = Ops.abs(loaded["w"] - lt.full([64, 32], 7.0)).max().item()
    ok &= check("reuse: second cycle overwrites with new values", d == 0.0, f"maxdiff={d}")
    return ok

def test_error_propagation(tmp):
    ok = True
    saver = lt.AsyncCheckpointSaver()
    bad_path = os.path.join(tmp, "no_such_dir", "ckpt.safetensors")
    saver.save_async(make_tensors(), bad_path)
    ok &= expect_raises("error: wait rethrows worker failure", saver.wait)
    return ok

def test_input_validation():
    ok = True
    saver = lt.AsyncCheckpointSaver()
    ok &= expect_raises("valid: empty tensor map rejected", lambda: saver.save_async({}, "/tmp/x.safetensors"))
    ok &= expect_raises("valid: empty filepath rejected", lambda: saver.save_async(make_tensors(), ""))
    ok &= expect_raises("valid: null tensor rejected", lambda: saver.save_async({"t": None}, "/tmp/x.safetensors"))
    return ok

def test_destructor_waits(tmp):
    ok = True
    path = os.path.join(tmp, "ckpt_dtor.safetensors")
    ts = make_tensors()
    w_ref = ts["w"].clone()
    saver = lt.AsyncCheckpointSaver()
    saver.save_async(ts, path)
    del saver
    ok &= check("dtor: file complete without explicit wait", os.path.exists(path))
    loaded = lt.load_safetensors(path)
    d = Ops.abs((loaded["w"] - w_ref)).max().item()
    ok &= check("dtor: values exact", d == 0.0, f"maxdiff={d}")
    return ok

if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="lt_async_ckpt_")
    all_ok = True
    all_ok &= test_basic_roundtrip(tmp)
    all_ok &= test_snapshot_isolation(tmp)
    all_ok &= test_max_one_inflight(tmp)
    all_ok &= test_double_buffer_reuse(tmp)
    all_ok &= test_error_propagation(tmp)
    all_ok &= test_input_validation()
    all_ok &= test_destructor_waits(tmp)
    print("ALL PASS" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)
