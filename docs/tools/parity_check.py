import argparse
import sys

import numpy as np
import torch

try:
    import litetorch as lt
except ImportError:
    lt = None

TOLERANCES = {
    "FP32": 1e-4,
    "FP64": 1e-8,
    "FP16": 1e-2,
    "BF16": 1e-2,
    "FP8_E4M3": 1e-2,
    "FP8_E5M2": 1e-2,
    "INT8": 0.0,
    "INT4": 0.0,
    "NF4": 1e-2,
    "INT32": 0.0,
    "INT64": 0.0,
    "UINT8": 0.0,
    "BOOL": 0.0,
}

REDUCTION_TOL_FP32 = 1e-3


def tolerance_for(dtype_name, reduction=False):
    if reduction and dtype_name == "FP32":
        return REDUCTION_TOL_FP32
    return TOLERANCES.get(dtype_name, 1e-4)


def to_torch(lt_tensor, shape, dtype=np.float32):
    if lt is None:
        raise ImportError("litetorch is not importable in this environment")
    cpu = lt_tensor.to(lt.Device("cpu"))
    arr = np.array(cpu.to_vector(), dtype=dtype).reshape(shape)
    return torch.tensor(arr)


def max_abs_diff(a, b):
    return float(torch.max(torch.abs(a.float() - b.float())).item())


def check_values(name, torch_tensor, lt_tensor, shape, dtype_name="FP32", tol=None):
    if tol is None:
        tol = tolerance_for(dtype_name)
    try:
        b = to_torch(lt_tensor, shape)
        a = torch_tensor.float() if hasattr(torch_tensor, "float") else torch.tensor(torch_tensor)
        ok = torch.allclose(a.float(), b.float(), atol=tol, rtol=tol)
        diff = max_abs_diff(a, b)
        print("correctness %s: %s (tol=%g max_abs_diff=%.3g)" % (name, "OK" if ok else "MISMATCH", tol, diff), flush=True)
        return ok, diff
    except Exception as e:
        print("correctness %s: ERROR %s" % (name, e), flush=True)
        return False, float("nan")


def check_grads(name, torch_tensor, lt_tensor, shape, dtype_name="FP32", tol=None):
    return check_values(name + " [grad]", torch_tensor, lt_tensor, shape, dtype_name=dtype_name, tol=tol)


def compare(name, torch_fn, lt_fn, shape, dtype_name="FP32", tol=None, check_grad=False):
    try:
        a = torch_fn()
        b = lt_fn()
    except Exception as e:
        print("correctness %s: ERROR %s" % (name, e), flush=True)
        return False
    ok, _ = check_values(name, a, b, shape, dtype_name=dtype_name, tol=tol)
    if check_grad:
        try:
            a_g = a.grad if hasattr(a, "grad") else None
            b_g = b.grad if hasattr(b, "grad") else None
            if a_g is None or b_g is None:
                print("correctness %s [grad]: ERROR missing grad" % name, flush=True)
                return False
            ok_g, _ = check_grads(name, a_g, b_g, shape, dtype_name=dtype_name, tol=tol)
            ok = ok and ok_g
        except Exception as e:
            print("correctness %s [grad]: ERROR %s" % (name, e), flush=True)
            return False
    return ok


def self_test():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ok = torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    print("parity_check self-test torch allclose: %s" % ("OK" if ok else "MISMATCH"), flush=True)
    if lt is None:
        print("parity_check self-test litetorch: SKIP (litetorch not importable; build first)", flush=True)
        return False
    try:
        t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.Device("cpu"), False, lt.DataType.FP32)
        ok2, diff = check_values("self-test lt from_vector", a, t, [2, 2])
        print("parity_check self-test litetorch: %s max_abs_diff=%.3g" % ("OK" if ok2 else "MISMATCH", diff), flush=True)
        return ok and ok2
    except Exception as e:
        print("parity_check self-test litetorch: ERROR %s" % e, flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return 0 if self_test() else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
