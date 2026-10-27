# How-To H2 — Phân tán: Thiết lập FSDP / TP / PP / CP

Bốn chiều song song, mỗi chiều một công thức chạy được. FSDP shard
tham số và gradient; tensor parallelism chia các lớp linear; pipeline
parallelism chia các stage trên các device; context parallelism chia
sequence. Mọi ví dụ CPU dưới đây chạy với LiteTorch 88a4ba1.

Điều kiện tiên quyết: LiteTorch đã được cài và import được. Các công
thức đa process dùng cùng cách khởi chạy `mp` spawn như H1.1 — đọc
phần đó trước.

> Các khác biệt với PyTorch liên quan ở đây: D-06 (`.long()` — ĐÃ SỬA
> agent-v1 @7c471db2, qa-r62: giờ trả về INT64; ext 88a4ba1 dùng cho
> batch này ra trước bản sửa), D-11 (indexing chỉ hỗ trợ int/slice
> trên dim 0 — dùng `Ops.gather` / `index_select` thay thế).

## H2.1 — FSDP: shard một mô hình trên 2 rank

Gọi `lt.distributed.fsdp_fully_shard(model)` sau `init_process_group`
và sau khi chuyển mô hình sang device của rank. Nó mutate mô hình tại
chỗ và trả về `None`: một `FSDPUnit` cho mỗi leaf module, các grad hook
reduce-scatter khi backward, các forward hook gather và release tự
động. Sau lời gọi này, `model(x)` + `loss.backward()` thông thường cứ
chạy — không có handle `fsdp_model` và không có bước gather/shard thủ
công nào.

FSDP shard tham số mô hình (đã flatten, mỗi rank một local shard) và
gradient. Nó **không** shard optimizer state.

<!-- ex: id=H2.1-1 device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29721


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([
            lt.nn.Linear(1024, 1024, True),
            lt.nn.GELU(),
            lt.nn.Linear(1024, 1024, True),
        ])
        model.to(cpu)
        lt.distributed.fsdp_fully_shard(model)
        lt.distributed.fsdp_set_prefetch_limit(2)
        x = lt.Tensor.from_vector([0.01] * 2048, [2, 1024], cpu)
        y = model(x)
        print("Rank %d forward shape: %s" % (rank, list(y.shape)))
        loss = y.sum()
        loss.backward()
        print("Rank %d backward done" % rank)
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
        p.join(180)
        if p.is_alive():
            p.terminate()
    print("exit codes:", [p.exitcode for p in procs])


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: cả hai
rank đều in forward shape `[2, 1024]` và `backward done`, exit code
`[0, 0]`.

Hai điều ví dụ này cố tình **không** làm. Thứ nhất, nó không khẳng
định loss giống nhau trên các rank: mỗi rank tự khởi tạo ngẫu nhiên
bản sao mô hình của mình, nên loss mỗi rank khác nhau. Trong huấn
luyện thực, hãy load cùng một checkpoint trên mọi rank (hoặc broadcast
tham số từ rank 0) trước khi gọi `fsdp_fully_shard`. Thứ hai, nó không
gọi các phương thức `FSDPUnit` (`prefetch_gather`, `ensure_gathered`,
`release_gathered`): chúng không gọi được từ Python — không có
`py::init` nào được bind và `FSDP::active_units()` cũng không được
bind, nên không có code Python nào lấy được một instance `FSDPUnit`.
Đừng ghi chúng như thể gọi được. Dọn dẹp bằng
`lt.distributed.shutdown()`; `destroy_process_group()` không tồn tại.

Không có lớp `FullyShardedDataParallel` trong LiteTorch — một ví dụ cũ
từng gọi nó, nhưng nó chưa bao giờ tồn tại trong cây này. Nếu bạn thấy
nó trong tài liệu cũ, đó là tài liệu lỗi thời.

## H2.2 — Tensor parallelism: chia một lớp linear

`ColumnParallelLinear` chia ma trận trọng số theo cột (mỗi rank tính
một lát của các feature đầu ra); `RowParallelLinear` chia theo hàng
(mỗi rank tính một tổng riêng phần trên một lát của các feature đầu
vào). Một MLP kiểu Megatron là một projection column-parallel, một hàm
phi tuyến, và một projection row-parallel. Phần này chạy trong một
process trên CPU; dưới một TP group thực, mỗi rank giữ shard của mình
và lớp row-parallel all-reduce các tổng riêng phần.

<!-- ex: id=H2.2-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()


class TensorParallelMLP(lt.nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.col_proj = lt.nn.ColumnParallelLinear(hidden_dim, ffn_dim, True)
        self.gelu = lt.nn.GELU()
        self.row_proj = lt.nn.RowParallelLinear(ffn_dim, hidden_dim, True)

    def forward(self, x):
        h = self.col_proj.forward(x)
        h = self.gelu.forward(h)
        out = self.row_proj.forward(h)
        return out


lt.manual_seed(0)
mlp = TensorParallelMLP(16, 32)
mlp.to(device)
x = lt.Tensor.from_vector([0.1] * 32, [2, 16], device, False)
y = mlp.forward(x)
print("TP MLP output shape:", list(y.shape))
print("TP MLP output values:", y.to_vector()[:4])
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: shape
`[2, 16]`. Đối với dạng đa rank, bọc cùng việc khởi tạo module trong
cách khởi chạy H1.1 và khởi tạo TP group trước
(`lt.distributed.init_tp_groups` sau `init_process_group`); lớp
`FSDPTP` kết hợp FSDP sharding với tensor parallelism.

## H2.3 — Pipeline parallelism: lịch 1F1B trên các microbatch

`PipelineParallelModule` bọc một stage. `schedule_1f1b(microbatches)`
chạy lịch one-forward-one-backward trên các microbatch để giảm thiểu
pipeline bubble. Phần này chạy trong một process trên CPU.

<!-- ex: id=H2.3-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

device = lt.auto_device()

stage_module = lt.nn.Sequential([
    lt.nn.Linear(8, 8, True),
    lt.nn.GELU(),
])
stage_module.to(device)

pp_module = lt.distributed.PipelineParallelModule(stage_module)

mb1 = lt.Tensor.from_vector([1.0] * 8, [1, 8], device, False)
mb2 = lt.Tensor.from_vector([2.0] * 8, [1, 8], device, False)

outputs = pp_module.schedule_1f1b([mb1, mb2])
print("1F1B outputs:", len(outputs))
for i, out in enumerate(outputs):
    print("microbatch %d output shape: %s" % (i, list(out.shape)))
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: 2 output,
mỗi cái shape `[1, 8]`. Để huấn luyện, dùng
`train_step_1f1b(microbatches, loss_fn)` thay thế — nó chạy forward,
backward, và bước optimizer cho mỗi microbatch.
`InterleavedPipelineModule` là biến thể interleaved (kiểu chimera)
của cùng wrapper đó.

## H2.4 — Context parallelism: shard sequence rồi gather lại

Context parallelism chia một sequence dài trên các rank. Các collective
di chuyển các shard là `all_gather` (mọi rank nhận tensor đầy đủ) và
`reduce_scatter` (mọi rank nhận shard đã reduce của mình). Công thức
này trình bày chuyển động shard mà một implementation CP được xây
dựng trên đó.

> Khoảng trống đã biết: `lt.Ops.ring_attention(q, k, v, pg, causal)`
> được bind nhưng không gọi được từ Python — truyền singleton
> `ProcessGroup` raise lỗi pybind holder (`Unable to load a custom
> holder type from a default-holder instance`). Đừng dùng nó cho tới
> khi binding được sửa. Chuyển động shard dưới đây là primitive CP
> đang dùng được hôm nay.

<!-- ex: id=H2.4-1 device=cpu backend=cpu runtime=~2s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp

import litetorch as lt

PORT = 29724


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        pg = lt.distributed.ProcessGroup.get()
        shard = lt.Tensor.from_vector([float(rank * 4 + i) for i in range(4)], [4], cpu, False)
        full = lt.Tensor.from_vector([0.0] * 8, [8], cpu, False)
        pg.all_gather(shard, full)
        print("Rank %d all_gather: %s" % (rank, full.to_vector()))
        full2 = lt.Tensor.from_vector([float(i + 1) for i in range(8)], [8], cpu, False)
        shard2 = lt.Tensor.from_vector([0.0] * 4, [4], cpu, False)
        pg.reduce_scatter(shard2, full2)
        print("Rank %d reduce_scatter: %s" % (rank, shard2.to_vector()))
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
rank đều in `all_gather: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]`
(hai shard 4 phần tử nối nhau theo thứ tự rank) và `reduce_scatter`
`[2.0, 4.0, 6.0, 8.0]` trên rank 0 / `[10.0, 12.0, 14.0, 16.0]` trên
rank 1 (tổng theo từng phần tử của `[1..8]` và `[1..8]` là `[2..16]`;
mỗi rank nhận shard 4 phần tử của mình). Exit code `[0, 0]`.

## H2.5 — DeviceMesh và DTensor: đặt tên cho sharding

Một `DeviceMesh` đặt tên cho lưới process; một `DTensor` ghép một
tensor cục bộ với mesh và một placement cho mỗi chiều (`SHARD`,
`REPLICATE`, hoặc `PARTIAL`). Đây là metadata cộng với storage cục bộ
— các collective ở H2.4 mới là thứ thực sự di chuyển dữ liệu.

<!-- ex: id=H2.5-1 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

mesh = lt.distributed.DeviceMesh([2, 2], "tp,dp")

placement_shard = lt.distributed.Placement()
placement_shard.type = lt.distributed.PlacementType.SHARD
placement_shard.dim = 0

local_data = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.auto_device(), False)
dtensor = lt.distributed.DTensor(local_data, mesh, [placement_shard])
print("DTensor local shape:", list(dtensor.local_tensor.shape))
print("DTensor mesh shape:", mesh.mesh_shape)
print("DTensor placement:", dtensor.placements[0].type, "dim", dtensor.placements[0].dim)
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Kết quả mong đợi: local
shape `[2, 2]`, mesh shape `[2, 2]`, placement `SHARD` dim `0`.
