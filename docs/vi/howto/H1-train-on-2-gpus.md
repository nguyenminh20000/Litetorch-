# How-To H1 — Huấn luyện trên 2 GPU

Hai process, mỗi process một GPU, nối với nhau bằng một `ProcessGroup`.
Cùng một cách khởi chạy dùng được trên CPU (trang này, chạy được ngay)
và trên GPU (H1.3, Kaggle). LiteTorch không có tương đương `torchrun`:
bạn tự spawn các process bằng `multiprocessing` và mỗi rank gọi
`init_process_group` trước.

Điều kiện tiên quyết: LiteTorch đã được cài và import được. Các ví dụ
CPU dưới đây chạy với LiteTorch 88a4ba1. Các bước GPU được đánh dấu
`REQUIRES-GPU` và kiểm chứng trên Kaggle ở Batch B5.

> Các khác biệt với PyTorch liên quan ở đây: D-02 (`.to(device)` lặng
> lẽ fallback về CPU khi accelerator không có — luôn kiểm tra `t.device`
> sau đó), D-03 (`.to()` nhận một `Device`, không có dạng chuỗi).

## H1.1 — Khởi chạy hai rank và chạy collectives

Spawn hai process bằng `mp.get_context("spawn")`. Mỗi rank khởi tạo
process group của mình, sau đó `all_reduce` cộng một tensor trên các
rank và `broadcast` sao chép tensor của rank 0 sang mọi rank. Luôn gọi
`shutdown()` trong khối `finally` — một rank thoát mà không shutdown
sẽ để rank còn lại chờ mãi trên socket.

<!-- ex: id=H1.1-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29711


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        pg = lt.distributed.ProcessGroup.get()
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu, False)
        pg.all_reduce(t)
        print("Rank %d all_reduce sum: %s" % (rank, t.to_vector()))
        b = lt.Tensor.from_vector([0.0] * 4, [4], cpu, False)
        if rank == 0:
            b = lt.Tensor.from_vector([7.0] * 4, [4], cpu, False)
        pg.broadcast(b, 0)
        print("Rank %d broadcast: %s" % (rank, b.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(120)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: cả hai
rank đều in `all_reduce sum: [3.0, 3.0, 3.0, 3.0]` (1 + 2) và
`broadcast: [7.0, 7.0, 7.0, 7.0]`, exit code `[0, 0]`.

Các collective còn lại có cùng dạng: `all_gather(shard, full)`,
`reduce_scatter(shard, full)`, `send_tensor(tensor, dst)` /
`recv_tensor(tensor, src)`, `barrier()`. Xem H2.4 để thấy `all_gather` /
`reduce_scatter` được dùng thực tế.

## H1.2 — Collective không khớp sẽ throw trên mọi rank

Trước khi reduce, các rank trao đổi header dtype và shape. Nếu không
khớp, mọi rank đều raise
`RuntimeError: collective tensor dtype/shape mismatch across ranks` —
không rank nào bị treo, không có kết quả sai lặng lẽ. Cách sửa luôn là
một: làm cho dtype và shape của tensor giống hệt nhau trên mọi rank
trước khi gọi collective.

<!-- ex: id=H1.2-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29712


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        pg = lt.distributed.ProcessGroup.get()
        n = 4 if rank == 0 else 3
        t = lt.Tensor.from_vector([1.0] * n, [n], cpu, False)
        try:
            pg.all_reduce(t)
            print("Rank %d: NO ERROR (unexpected)" % rank)
        except RuntimeError as e:
            print("Rank %d: threw as expected: %s" % (rank, str(e)))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(60)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: cả hai
rank đều in
`threw as expected: collective tensor dtype/shape mismatch across ranks`,
exit code `[0, 0]`. Đây cũng là kiểm tra mà đường NCCL 2-GPU áp dụng
trước khi chạy collective — lỗi shape fail loud thay vì làm hỏng phép
reduce.

## H1.3 — Cùng cách khởi chạy trên 2 GPU (REQUIRES-GPU)

Trên GPU cách làm giống hệt, chỉ khác một điểm mỗi rank: chuyển mô
hình sang GPU device của rank đó và kiểm tra device đã dính. Đừng bao
giờ cho rằng phép chuyển đã thành công — trên máy không có CUDA,
`.to()` lặng lẽ trả về một tensor CPU (D-02).

<!-- ex: id=H1.3-1 device=gpu backend=cuda runtime=~60s torch-parity=n/a status=verified-2026-10-31@120fdedc -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29713


def worker(rank, world_size, port):
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        gpu = lt.Device("cuda:%d" % rank)
        model = lt.nn.Sequential([
            lt.nn.Linear(1024, 1024, True),
            lt.nn.GELU(),
            lt.nn.Linear(1024, 1024, True),
        ])
        model.to(gpu)
        params = model.parameters()
        assert params[0].device.type == lt.DeviceType.GPU, "model is not on GPU"
        x = lt.Tensor.from_vector([0.01] * 2048, [2, 1024], gpu)
        y = model(x)
        loss = y.sum()
        loss.backward()
        print("Rank %d GPU step done, loss: %s" % (rank, loss.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(target=worker, args=(rank, 2, PORT))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(300)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

REQUIRES-GPU. Kiểm chứng trên Kaggle (Batch B5) với pre-flight gate
trước: dòng init `Native GPU (CUDA/HIP)` có mặt, `liblitetorch_gpu.so`
nằm cạnh extension dưới `<tree>/python/`, và
`lt.cuda.device_count()` trả về 2. Đối với một bước huấn luyện 2-GPU
thực sự có đồng bộ gradient, hãy shard mô hình bằng FSDP (H2.1) thay
vì replicate — replicate một mình không đồng bộ gradient.
