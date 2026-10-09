import time
import torch
import sys

sys.path.insert(0, '/tmp/lt-bench')
import litetorch as lt


def stress_conv(depth=8, width=128, size=32, batch=64, iters=20):
    print(f"== stress conv: depth={depth} width={width} size={size} batch={batch} ==", flush=True)
    torch.manual_seed(0)
    layers_t = []
    for i in range(depth):
        layers_t.append(torch.nn.Conv2d(width if i else 3, width, 3, padding=1))
        layers_t.append(torch.nn.ReLU())
    model_t = torch.nn.Sequential(*layers_t).cuda()
    x_t = torch.randn(batch, 3, size, size).cuda()
    for _ in range(3):
        model_t(x_t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model_t(x_t)
    torch.cuda.synchronize()
    t_torch = (time.time() - t0) / iters * 1000

    dev = lt.Device('gpu:0')
    layers_l = []
    for i in range(depth):
        c = lt.nn.Conv2d(width if i else 3, width, 3, padding=1)
        c.to(dev)
        layers_l.append(c)
        layers_l.append(lt.nn.ReLU())
    model_l = lt.nn.Sequential(layers_l)
    model_l.to(dev)
    x_l = lt.Tensor.from_vector(torch.randn(batch * 3 * size * size).tolist(), [batch, 3, size, size], dev)
    for _ in range(3):
        model_l.forward(x_l)
    lt.cuda_synchronize()
    t0 = time.time()
    for _ in range(iters):
        model_l.forward(x_l)
    lt.cuda_synchronize()
    t_lt = (time.time() - t0) / iters * 1000
    print(f"STRESS CONV: torch {t_torch:.2f}ms | lt {t_lt:.2f}ms | x{t_torch/t_lt:.2f}", flush=True)


def stress_linear(depth=8, width=1024, batch=512, iters=50):
    print(f"== stress linear: depth={depth} width={width} batch={batch} ==", flush=True)
    torch.manual_seed(0)
    layers_t = []
    for i in range(depth):
        layers_t.append(torch.nn.Linear(width, width))
        layers_t.append(torch.nn.ReLU())
    model_t = torch.nn.Sequential(*layers_t).cuda()
    x_t = torch.randn(batch, width).cuda()
    for _ in range(3):
        model_t(x_t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model_t(x_t)
    torch.cuda.synchronize()
    t_torch = (time.time() - t0) / iters * 1000

    dev = lt.Device('gpu:0')
    layers_l = []
    for i in range(depth):
        l = lt.nn.Linear(width, width)
        l.to(dev)
        layers_l.append(l)
        layers_l.append(lt.nn.ReLU())
    model_l = lt.nn.Sequential(layers_l)
    model_l.to(dev)
    x_l = lt.Tensor.from_vector(torch.randn(batch * width).tolist(), [batch, width], dev)
    for _ in range(3):
        model_l.forward(x_l)
    lt.cuda_synchronize()
    t0 = time.time()
    for _ in range(iters):
        model_l.forward(x_l)
    lt.cuda_synchronize()
    t_lt = (time.time() - t0) / iters * 1000
    print(f"STRESS LINEAR: torch {t_torch:.2f}ms | lt {t_lt:.2f}ms | x{t_torch/t_lt:.2f}", flush=True)


def stress_large_batch():
    print("== stress large batch (memory pressure) ==", flush=True)
    torch.manual_seed(0)
    for batch in [64, 128, 256, 512]:
        try:
            model_t = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            ).cuda()
            x_t = torch.randn(batch, 3, 64, 64).cuda()
            for _ in range(3):
                model_t(x_t)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                model_t(x_t)
            torch.cuda.synchronize()
            t_torch = (time.time() - t0) / 10 * 1000

            dev = lt.Device('gpu:0')
            c1 = lt.nn.Conv2d(3, 64, 3, padding=1); c1.to(dev)
            c2 = lt.nn.Conv2d(64, 128, 3, padding=1); c2.to(dev)
            r = lt.nn.ReLU()
            model_l = lt.nn.Sequential([c1, r, c2, r]); model_l.to(dev)
            x_l = lt.Tensor.from_vector(torch.randn(batch * 3 * 64 * 64).tolist(), [batch, 3, 64, 64], dev)
            for _ in range(3):
                model_l.forward(x_l)
            lt.cuda_synchronize()
            t0 = time.time()
            for _ in range(10):
                model_l.forward(x_l)
            lt.cuda_synchronize()
            t_lt = (time.time() - t0) / 10 * 1000
            print(f"  batch={batch}: torch {t_torch:.2f}ms | lt {t_lt:.2f}ms | x{t_torch/t_lt:.2f}", flush=True)
        except Exception as e:
            print(f"  batch={batch}: FAILED {e}", flush=True)
            break


if __name__ == '__main__':
    stress_conv()
    stress_linear()
    stress_large_batch()
    print("STRESS DONE", flush=True)
