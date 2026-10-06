import sys
import time

import torch

import litetorch as lt

GPU = lt.Device("gpu:0")


def sync():
    torch.cuda.synchronize()


def bench(name, torch_fn, lt_fn, iters=100, warmup=20):
    for _ in range(warmup):
        torch_fn()
    sync()
    for _ in range(warmup):
        lt_fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch_fn()
    sync()
    t_torch = (time.perf_counter() - t0) / iters * 1000.0
    t0 = time.perf_counter()
    for _ in range(iters):
        lt_fn()
    sync()
    t_lt = (time.perf_counter() - t0) / iters * 1000.0
    print(
        f"{name:38s} torch {t_torch:9.3f} ms | lt {t_lt:9.3f} ms | x{t_torch / t_lt:6.2f}",
        flush=True,
    )


def rand_torch(*shape, dtype=torch.float32):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def rand_lt(*shape, dtype=lt.DataType.FP32):
    n = 1
    for s in shape:
        n *= s
    return lt.Tensor.from_vector([0.1] * n, list(shape), GPU, False, dtype)


def main():
    assert torch.cuda.is_available(), "no torch cuda"
    assert lt.is_gpu_available(), "no litetorch gpu"
    print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0), flush=True)

    for n in (512, 1024, 2048, 4096):
        a_t, b_t = rand_torch(n, n), rand_torch(n, n)
        a_l, b_l = rand_lt(n, n), rand_lt(n, n)
        bench(
            f"matmul fp32 [{n}x{n}]",
            lambda: torch.mm(a_t, b_t),
            lambda: lt.Ops.matmul(a_l, b_l),
            iters=50 if n >= 2048 else 100,
        )

    for n in (1024, 2048, 4096):
        a_t = rand_torch(n, n, dtype=torch.float16)
        b_t = rand_torch(n, n, dtype=torch.float16)
        a_l = rand_lt(n, n, dtype=lt.DataType.FP16)
        b_l = rand_lt(n, n, dtype=lt.DataType.FP16)
        bench(
            f"matmul fp16 [{n}x{n}]",
            lambda: torch.mm(a_t, b_t),
            lambda: lt.Ops.matmul(a_l, b_l),
            iters=50 if n >= 2048 else 100,
        )

    a_t = rand_torch(16, 512, 512)
    b_t = rand_torch(16, 512, 512)
    a_l = rand_lt(16, 512, 512)
    b_l = rand_lt(16, 512, 512)
    bench("bmm fp32 [16,512,512]", lambda: torch.bmm(a_t, b_t), lambda: lt.Ops.bmm(a_l, b_l))

    x_t = rand_torch(32, 64, 56, 56)
    w_t = rand_torch(128, 64, 3, 3)
    x_l = rand_lt(32, 64, 56, 56)
    w_l = rand_lt(128, 64, 3, 3)
    bench(
        "conv2d fp32 [32,64,56,56]x[128,64,3,3]",
        lambda: torch.nn.functional.conv2d(x_t, w_t, padding=1),
        lambda: lt.Ops.conv2d(x_l, w_l, padding=1),
        iters=50,
    )

    n = 16 * 1024 * 1024
    a_t, b_t = rand_torch(n), rand_torch(n)
    a_l, b_l = rand_lt(n), rand_lt(n)
    bench("add fp32 16M", lambda: a_t + b_t, lambda: lt.Ops.add(a_l, b_l))
    bench("mul fp32 16M", lambda: a_t * b_t, lambda: lt.Ops.mul(a_l, b_l))
    bench("relu fp32 16M", lambda: torch.relu(a_t), lambda: lt.Ops.relu(a_l))

    a_t = rand_torch(1024, 4096)
    a_l = rand_lt(1024, 4096)
    bench(
        "softmax fp32 [1024,4096]",
        lambda: torch.softmax(a_t, dim=-1),
        lambda: lt.Ops.softmax(a_l, -1),
    )

    a_t = rand_torch(1024, 4096)
    a_l = rand_lt(1024, 4096)
    w_t = rand_torch(4096)
    bench(
        "layer_norm fp32 [1024,4096]",
        lambda: torch.nn.functional.layer_norm(a_t, (4096,), w_t),
        lambda: lt.Ops.layer_norm(a_l, [4096], rand_lt(4096)),
    )

    a_t = rand_torch(n)
    a_l = rand_lt(n)
    bench("sum fp32 16M", lambda: torch.sum(a_t), lambda: lt.Ops.sum(a_l))

    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
