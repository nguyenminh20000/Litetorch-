import litetorch as lt
import gc

def test_checkpoint_detach():
    d = lt.Device("cpu")
    x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], d, True)
    w = lt.Tensor.from_vector([0.5, 0.5, 0.5, 0.5], [2, 2], d, True)
    def fn(inp):
        return lt.ops.matmul(inp, w)
    y = lt.checkpoint(fn, x)
    y2 = lt.ops.matmul(x, w)
    assert y.to_vector() == y2.to_vector(), "checkpoint forward mismatch"
    print("[PASS] checkpoint forward matches")

def test_checkpoint_grad():
    d = lt.Device("cpu")
    x = lt.Tensor.from_vector([1.0, 2.0], [1, 2], d, True)
    w = lt.Tensor.from_vector([3.0, 4.0, 5.0, 6.0], [2, 2], d, True)
    def fn(inp):
        return lt.ops.matmul(inp, w)
    y = lt.checkpoint(fn, x)
    s = lt.ops.sum(y)
    s.backward()
    assert x.grad is not None, "x.grad is None"
    assert w.grad is not None, "w.grad is None"
    xv = x.grad.to_vector()
    assert abs(xv[0] - 7.0) < 1e-4 and abs(xv[1] - 11.0) < 1e-4, f"x.grad wrong: {xv}"
    print("[PASS] checkpoint grad correct:", xv)

def test_rng_state():
    s1 = lt.get_rng_state()
    assert isinstance(s1, str) and len(s1) > 0, "empty rng state"
    lt.manual_seed(42)
    s2 = lt.get_rng_state()
    lt.manual_seed(123)
    s3 = lt.get_rng_state()
    assert s2 != s3, "rng state not changing with seed"
    lt.set_rng_state(s2)
    s4 = lt.get_rng_state()
    assert s2 == s4, "rng state restore failed"
    print("[PASS] rng save/restore works")

if __name__ == "__main__":
    test_checkpoint_detach()
    test_checkpoint_grad()
    test_rng_state()
    print("ALL CHECKPOINT TESTS PASSED")
