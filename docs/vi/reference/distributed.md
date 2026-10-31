# Phân tán — Tài liệu tham khảo API

`litetorch.distributed` là bề mặt huấn luyện phân tán: process group và collective, song song 3D (DP/TP/PP), DTensor/DeviceMesh, FSDP, song song pipeline, checkpoint phân mảnh, và khởi động lại đàn hồi (rendezvous + heartbeat). Mọi tính toán đều bằng C++14; Python là lớp giao diện pybind11.

- **Nguồn chân lý:** `src/bindings/bind_distributed.cpp` (321 dòng, binding Python), `include/litetorch/distributed.h`, `fsdp_wrapper.h`, `fsdp_tp.h`, `device_mesh.h`, `dtensor.h`, `flat_param.h`, `dist_checkpoint.h`, `elastic.h`, `rendezvous.h`, `parallel_3d.h`, `pipeline.h`.
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline tài liệu là extension 88a4ba1; B5 xác minh lại tại head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Các ví dụ đa tiến trình dùng `mp.get_context("spawn")` với 2 tiến trình và một cổng riêng cho mỗi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU. Ngang bằng torch là FP32 1e-4 trừ khi tiêu đề ghi khác.
- **Sai khác liên quan:** D-02 (`to(device)` tự động rơi về CPU trong im lặng — một rank rơi về CPU vẫn tham gia group, chỉ chậm đi trong im lặng). Xem `plan/differences.md`.
- **NCCL:** các collective dùng NCCL khi có backend GPU native. **Trạng thái, đã chốt 2026-10-27 theo dist-w22: cơ chế lựa chọn `LITETORCH_NCCL_LIB` là patch đã kiểm chứng trên Kaggle, chưa merge** (dist-w15 cùng toàn bộ vòng 2-GPU w10–w13 đã được phê duyệt và kiểm chứng trên Kaggle 2026-10-19 trên cây session đã vá, nhưng chưa từng chạy bước merge; các patch vẫn nằm chưa merge trong `patches/distributed/`). Hành vi hiện tại tại HEAD b326f29 là dlopen fail-soft: `dlopen("libnccl.so")` → `dlopen("librccl.so")`, và thư viện thiếu đặt `available_=false` thay vì fail-closed. Đừng viết code đặt `LITETORCH_NCCL_LIB` với kỳ vọng nó sẽ được tôn trọng cho đến khi bộ w10→w15 merge. Trên máy chỉ có CPU, process group dùng transport CPU; không cần NCCL.
- **Chế độ lỗi:** gọi bất kỳ collective nào trước `init_process_group` đều raise; `init_tp_groups` raise nếu process group chưa được khởi tạo; shape/dtype tensor không khớp giữa các rank sẽ throw (mismatch-throw) thay vì treo; một rank không bao giờ tham gia sẽ làm các rank khác kẹt tại collective đầu tiên — luôn gọi `shutdown()` trong khối `finally`. Không có `destroy_process_group()` — dùng `shutdown()`.

## Process group và collective

### `distributed.init_process_group`

- **Chữ ký:** `init_process_group(rank, world_size, master_addr="127.0.0.1", master_port=29500) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:138`
- **Tham số:** `rank` — int, hạng của tiến trình này trong `[0, world_size)`. `world_size` — int, tổng số rank. `master_addr` — địa chỉ rendezvous, mặc định `"127.0.0.1"`. `master_port` — cổng rendezvous, mặc định 29500. Mọi rank phải truyền cùng `world_size`, `master_addr`, và `master_port`; các rank phải khác nhau.
- **Trả về:** `None`. Khởi tạo process group singleton. Gọi hai lần mà không có `shutdown()` ở giữa là lỗi.
- **Trường hợp biên:** một tiến trình thứ hai truyền `world_size` khác sẽ treo hoặc throw tại collective đầu tiên. Cổng rendezvous phải trống; dùng `rendezvous_free_port()` để chọn một cổng.
- **PyTorch:** `torch.distributed.init_process_group(backend, rank, world_size)` — tương đương về hình thức; LiteTorch không có đối số `backend` (transport được chọn theo backend sẵn có) và không có chuỗi `init_method`.

### `distributed.shutdown`

- **Chữ ký:** `shutdown() -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:141`
- **Trả về:** `None`. Dỡ process group. Gọi trong khối `finally`.
- **Trường hợp biên:** `destroy_process_group()` KHÔNG tồn tại trong LiteTorch — đừng gọi (sẽ raise `AttributeError`).
- **PyTorch:** `torch.distributed.destroy_process_group()` — cùng vai trò, tên khác.

### `distributed.barrier`

- **Chữ ký:** `barrier() -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:142`
- **Trả về:** `None`. Chặn cho đến khi mọi rank đều tới barrier.
- **Trường hợp biên:** một rank không bao giờ gọi `barrier()` sẽ deadlock các rank còn lại. Không có đối số timeout.
- **PyTorch:** `torch.distributed.barrier()` — tương đương.

### `distributed.all_reduce`

- **Chữ ký:** `all_reduce(tensor) -> None`; `all_reduce(tensor, op=ReduceOp.SUM) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:143-148`
- **Tham số:** `tensor` — tensor cần reduce tại chỗ (mỗi rank đóng góp giá trị của mình; mọi rank nhận kết quả đã reduce). `op` — một `ReduceOp`, mặc định `SUM`.
- **Trả về:** `None`. Phép reduce thực hiện tại chỗ: `tensor` giữ kết quả khi trả về.
- **Trường hợp biên:** mọi rank phải truyền tensor cùng shape và dtype nếu không lệnh gọi sẽ throw (mismatch-throw). Dạng không có `op` luôn cộng.
- **PyTorch:** `torch.distributed.all_reduce(tensor, op=...)` — tương đương, trừ việc torch trả về tensor thay vì `None`.

### `distributed.all_gather`

- **Chữ ký:** `all_gather(shard, full) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:149`
- **Tham số:** `shard` — mảnh cục bộ của rank này. `full` — tensor đầu ra đủ lớn để chứa mảnh của mọi rank, theo thứ tự rank.
- **Trả về:** `None`. `full` được điền tại chỗ.
- **Trường hợp biên:** `full` phải đủ chỗ cho `world_size` mảnh; `full` quá nhỏ sẽ throw.
- **PyTorch:** `torch.distributed.all_gather(list, tensor)` — torch nhận một danh sách tensor đầu ra; LiteTorch nhận một tensor `full` đã định kích thước.

### `distributed.reduce_scatter`

- **Chữ ký:** `reduce_scatter(shard, full) -> None`; `reduce_scatter(shard, full, op) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:150-155`
- **Tham số:** `shard` — đầu ra, lát cắt của kết quả đã reduce thuộc về rank này. `full` — đầu vào, tensor đầy đủ cần reduce và scatter. `op` — một `ReduceOp` (không có mặc định ở dạng 3 đối số).
- **Trả về:** `None`. `shard` được điền tại chỗ.
- **Trường hợp biên:** `full` phải chia đều cho các rank.
- **PyTorch:** `torch.distributed.reduce_scatter(output, input_list, op=...)` — torch nhận một danh sách đầu vào; LiteTorch nhận một tensor `full`.

### `distributed.broadcast`

- **Chữ ký:** `broadcast(tensor, src_rank) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:156`
- **Tham số:** `tensor` — tensor cần broadcast tại chỗ (giá trị của rank `src_rank` thắng). `src_rank` — int, rank nguồn.
- **Trả về:** `None`.
- **Trường hợp biên:** mọi rank phải gọi `broadcast` với cùng `src_rank`.
- **PyTorch:** `torch.distributed.broadcast(tensor, src=...)` — tương đương.

### `distributed.all_to_all`

- **Chữ ký:** `all_to_all(input, output) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:157`
- **Tham số:** `input` — tensor chứa phần của rank này dành cho mọi rank. `output` — tensor nhận phần của mọi rank dành cho rank này.
- **Trả về:** `None`.
- **PyTorch:** `torch.distributed.all_to_all(output, input)` — cùng phép toán, thứ tự đối số đảo ngược (LiteTorch là `input, output`).

### `distributed.scatter` / `distributed.gather`

- **Chữ ký:** `scatter(output, input, src) -> None`; `gather(input, output, dst) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:158-159`
- **Tham số:** `scatter`: `output` — mảnh rank này nhận được; `input` — tensor đầy đủ (chỉ có ý nghĩa trên `src`); `src` — rank nguồn. `gather`: `input` — mảnh của rank này; `output` — tensor đã gather (chỉ có ý nghĩa trên `dst`); `dst` — rank đích.
- **Trả về:** `None`.
- **PyTorch:** `torch.distributed.scatter(output, src, list)` / `gather(output_list, input, dst)` — cùng phép toán, hình thức đối số khác.

### `distributed.send_tensor` / `distributed.recv_tensor`

- **Chữ ký:** `send_tensor(tensor, dst) -> None`; `recv_tensor(tensor, src) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:160-161`
- **Tham số:** `tensor` — tensor cần gửi / bộ đệm cần nhận vào. `dst` / `src` — rank đối tác.
- **Trả về:** `None`. Điểm-điểm, blocking.
- **Trường hợp biên:** một `send_tensor` không có `recv_tensor` đối tác sẽ chặn mãi mãi. Shape phải khớp ở cả hai phía.
- **PyTorch:** `torch.distributed.send(tensor, dst)` / `recv(tensor, src)` — tương đương.

<!-- ex: id=r12-collectives device=cpu backend=cpu runtime=~15s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        lt.distributed.all_reduce(t)
        print("rank", rank, "all_reduce SUM:", t.to_vector())
        b = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
        if rank == 0:
            b = lt.Tensor.from_vector([7.0] * 4, [4], cpu)
        lt.distributed.broadcast(b, 0)
        print("rank", rank, "broadcast:", b.to_vector())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29610)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

### `distributed.ProcessGroup`

- **Chữ ký:** class, singleton qua `ProcessGroup.get()`
- **Nguồn:** `src/bindings/bind_distributed.cpp:22-41`
- **Phương thức:** `get()` (static, trả về singleton) · `init(rank, world_size, master_addr, master_port)` · `shutdown()` · `all_reduce(tensor)` / `all_reduce(tensor, op)` · `barrier()` · `all_to_all(input, output)` · `scatter(output, input, src)` · `gather(input, output, dst)` · `broadcast(tensor, src)` · `all_gather(shard, full)` · `reduce_scatter(shard, full)` / `reduce_scatter(shard, full, op)` · `send_tensor(tensor, dst)` · `recv_tensor(tensor, src)` · `sync_comm()` · `get_rank()` · `get_world_size()` · `is_initialized()`.
- **Trả về:** các phương thức phản chiếu các hàm mức module ở trên, nhưng gắn với đối tượng group. `get_rank()` / `get_world_size()` trả về int; `is_initialized()` trả về bool.
- **Trường hợp biên:** `ProcessGroup.get()` trước `init()` trả về một group chưa khởi tạo; gọi collective trên đó sẽ throw. Ưu tiên dùng các hàm mức module trừ khi bạn cần đối tượng.
- **PyTorch:** `torch.distributed.ProcessGroup` — cùng khái niệm; LiteTorch expose nó dưới dạng class cụ thể, không phải subclass chọn theo backend.

### `distributed.ReduceOp`

- **Chữ ký:** enum với `SUM`, `AVG`, `PRODUCT`, `MIN`, `MAX` (cũng được export dưới dạng tên mức cao nhất)
- **Nguồn:** `src/bindings/bind_distributed.cpp:130-137`
- **Trả về:** mỗi giá trị là một toán tử reduce cho `all_reduce` / `reduce_scatter`.
- **Trường hợp biên:** `AVG` chia cho `world_size` sau khi cộng.
- **PyTorch:** `torch.distributed.ReduceOp` — cùng các giá trị (`SUM`, `AVG`, `PRODUCT`, `MIN`, `MAX`); torch còn có `BAND`, `BOR`, `BXOR`, mà LiteTorch không có.

### `distributed.init_tp_groups`

- **Chữ ký:** `init_tp_groups(tp_size) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:14-21`
- **Tham số:** `tp_size` — int, kích thước group song song tensor.
- **Trả về:** `None`. Chia process group đã khởi tạo thành các TP group qua `NCCLBridge`.
- **Trường hợp biên:** raise `[litetorch Error] init_tp_groups: init_process_group first` nếu process group chưa được khởi tạo. `tp_size` phải chia hết `world_size`.
- **PyTorch:** không có tương đương trực tiếp; torch tạo TP group bằng `torch.distributed.new_group()`.

### `distributed.tp_all_reduce` / `distributed.tp_all_gather`

- **Chữ ký:** `tp_all_reduce(tensor) -> Tensor`; `tp_all_gather(shard) -> Tensor`
- **Nguồn:** `src/bindings/bind_distributed.cpp:193-196`
- **Tham số:** `tensor` / `shard` — tensor cần reduce / gather.
- **Trả về:** tensor đã reduce / tensor đã gather. Khác với các collective thường, các hàm này nhận biết autograd: gradient chảy qua chúng.
- **Trường hợp biên:** cần TP group đã khởi tạo (`init_tp_groups` trước).
- **PyTorch:** `torch.distributed._functional_collectives` — cùng ý tưởng (collective khả vi); LiteTorch expose chúng dưới dạng hàm thường.

## Song song 3D

### `distributed.Parallel3D`

- **Chữ ký:** class, singleton qua `Parallel3D.get()`
- **Nguồn:** `src/bindings/bind_distributed.cpp:56-74`
- **Phương thức:** `get()` (static) · `init(rank, world_size, tp_size, pp_size)` · `reset()` · `is_initialized()` · `tp_size()` · `pp_size()` · `dp_size()` · `global_rank()` · `world_size()` · `tp_rank()` · `pp_rank()` · `dp_rank()` · `coords()` · `tp_group()` · `pp_group()` · `dp_group()` · `wire_nccl()`.
- **Trả về:** `init` thiết lập lưới 3D (`dp_size = world_size // (tp_size * pp_size)`); các phương thức `*_rank()` trả về tọa độ của rank này trên mỗi trục; `tp_group()` / `pp_group()` / `dp_group()` trả về các sub-group.
- **Trường hợp biên:** `tp_size * pp_size` phải chia hết `world_size`. `wire_nccl()` nối các NCCL group; trên máy chỉ có CPU nó là no-op.
- **PyTorch:** không có tương đương đơn lẻ; torch kết hợp song song 3D từ `device_mesh` + các lệnh gọi `new_group()`.

### `distributed.init_3d_parallel` / `distributed.shutdown_3d_parallel`

- **Chữ ký:** `init_3d_parallel(tp_size, pp_size=1) -> None`; `shutdown_3d_parallel() -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:76-78`
- **Tham số:** `tp_size` — kích thước song song tensor; `pp_size` — kích thước song song pipeline, mặc định 1.
- **Trả về:** `None`. Wrapper tiện lợi quanh `Parallel3D.get().init(...)` / `reset()`. Cần process group đã khởi tạo.
- **PyTorch:** không có tương đương trực tiếp.

<!-- ex: id=r12-3d device=cpu backend=cpu runtime=~15s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        lt.distributed.init_3d_parallel(2, 1)
        p = lt.distributed.Parallel3D.get()
        print("rank", rank, "tp", p.tp_rank(), "dp", p.dp_rank(),
              "tp_size", p.tp_size(), "dp_size", p.dp_size())
        lt.distributed.shutdown_3d_parallel()
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29612)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## Device mesh và DTensor

### `distributed.DeviceMesh`

- **Chữ ký:** `DeviceMesh(shape, dim_names)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:43-53`
- **Tham số:** `shape` — danh sách int, ví dụ `[2, 2]` cho mesh 2x2. `dim_names` — chuỗi đặt tên cho các chiều của mesh (ví dụ `"dp_tp"`).
- **Thuộc tính (đọc/ghi):** `mesh_shape`, `mesh_dim_names`, `mesh_topology`.
- **Phương thức:** `get_rank(coords)` · `get_coords(rank)` · `get_dim_group(dim, coords)` · `dim_size(dim)` · `num_dims()` · `numel()`.
- **Trả về:** `get_rank` ánh xạ tọa độ thành rank toàn cục; `get_coords` là phép ngược; `numel()` là tổng số thiết bị trong mesh.
- **Trường hợp biên:** `get_dim_group` trả về sub-group cho một chiều của mesh; nó raise khi `dim` ngoài phạm vi.
- **PyTorch:** `torch.distributed.device_mesh.DeviceMesh` — cùng khái niệm; constructor của torch nhận loại thiết bị cộng shape, LiteTorch nhận shape cộng chuỗi tên.

### `distributed.PlacementType` / `distributed.Placement`

- **Chữ ký:** enum `PlacementType` với `SHARD`, `REPLICATE`, `PARTIAL`; class `Placement()` với `type` và `dim` đọc/ghi
- **Nguồn:** `src/bindings/bind_distributed.cpp:81-90`
- **Tham số:** `Placement.type` — một `PlacementType`; `Placement.dim` — chiều bị shard (đối với `SHARD`).
- **Trả về:** một placement mô tả cách một `DTensor` được bố trí trên mesh: `SHARD` chia một chiều trên mesh, `REPLICATE` sao chép toàn bộ tensor tới mọi thiết bị, `PARTIAL` giữ một kết quả từng phần (chưa reduce).
- **PyTorch:** `torch.distributed._tensor.Placement` (`Shard`, `Replicate`, `Partial`) — cùng ba loại.

### `distributed.DTensor`

- **Chữ ký:** `DTensor(local_tensor, mesh, placements)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:92-98`
- **Tham số:** `local_tensor` — mảnh cục bộ của rank này. `mesh` — một `DeviceMesh`. `placements` — danh sách `Placement`, một cho mỗi chiều của mesh.
- **Thuộc tính (đọc/ghi):** `device_mesh`, `placements`, `local_tensor`.
- **Phương thức:** `redistribute(new_mesh, new_placements)` — trả về một `DTensor` mới với bố cục khác (có thể giao tiếp).
- **Trả về:** một tensor phân tán; nó là subclass của `Tensor` (`include/litetorch/dtensor.h: DTensor : public Tensor`).
- **Trường hợp biên:** số placement phải khớp số chiều của mesh. `redistribute` sang placement `REPLICATE` sẽ kích hoạt all-gather.
- **PyTorch:** `torch.distributed._tensor.DTensor` — cùng khái niệm; `redistribute()` của torch chỉ nhận placement (cùng mesh), LiteTorch còn nhận mesh mới.

<!-- ex: id=r12-mesh device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
mesh = lt.distributed.DeviceMesh([2], "dp")
print("mesh numel:", mesh.numel(), "dims:", mesh.num_dims())
local = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
p = lt.distributed.Placement()
p.type = lt.distributed.PlacementType.SHARD
p.dim = 0
dt = lt.distributed.DTensor(local, mesh, [p])
print("dtensor local shape:", dt.local_tensor.shape)
print("placement:", dt.placements[0].type, "dim", dt.placements[0].dim)
```

## Collective chồng chéo

### `distributed.OverlappedAllReducer`

- **Chữ ký:** class, singleton qua `OverlappedAllReducer.get()`
- **Nguồn:** `src/bindings/bind_distributed.cpp:100-104`
- **Phương thức:** `get()` (static) · `push_and_all_reduce(tensor)` · `push_and_reduce_scatter(shard, full)` · `sync()`.
- **Trả về:** `push_and_*` xếp hàng một collective chạy nền; `sync()` chặn cho đến khi mọi collective đã xếp hàng hoàn tất.
- **Trường hợp biên:** tensor đã push phải còn sống cho đến `sync()`. Đừng thay đổi một tensor đã push trước `sync()`.
- **PyTorch:** không có tương đương trực tiếp; gần nhất là các op async của `torch.distributed` (`async_op=True`).

### `distributed.overlapped_all_reduce` / `distributed.overlapped_reduce_scatter` / `distributed.sync_overlapped_all_reduce`

- **Chữ ký:** `overlapped_all_reduce(tensor) -> None`; `overlapped_reduce_scatter(shard, full) -> None`; `sync_overlapped_all_reduce() -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:106-108`
- **Tham số:** cùng tensor như các collective thường.
- **Trả về:** `None`. Wrapper mức module quanh singleton `OverlappedAllReducer`.
- **PyTorch:** không có tương đương trực tiếp.

<!-- ex: id=r12-overlapped device=cpu backend=cpu runtime=~15s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        t = lt.Tensor.from_vector([float(rank + 1)] * 2, [2], cpu)
        lt.distributed.overlapped_all_reduce(t)
        lt.distributed.sync_overlapped_all_reduce()
        print("rank", rank, "overlapped all_reduce:", t.to_vector())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29614)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## Song song pipeline

### `distributed.PipelineParallelModule`

- **Chữ ký:** `PipelineParallelModule(sub_module)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:110-118`
- **Tham số:** `sub_module` — một `nn.Module` (một stage của pipeline).
- **Phương thức:** `forward(input)` · `forward_microbatches(microbatches)` · `schedule_1f1b(microbatches)` · `train_step_1f1b(microbatches, loss_fn=None)` · `run_1f1b_with_backward(microbatches, loss_fn)` · `parameters()` · `to(device)`.
- **Trả về:** `forward` chạy stage; `schedule_1f1b` trả về lịch 1F1B (một-forward-một-backward); `train_step_1f1b` chạy một bước huấn luyện đầy đủ trên các micro-batch.
- **Trường hợp biên:** module phải được đặt lên thiết bị của rank trước khi wrap. `loss_fn` mặc định `None` (không tính loss).
- **PyTorch:** `torch.distributed.pipelining` — cùng ý tưởng (stage kiểu GPipe); lịch 1F1B của LiteTorch được tích hợp sẵn.

### `distributed.InterleavedPipelineModule`

- **Chữ ký:** `InterleavedPipelineModule(chunks)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:120-128`
- **Tham số:** `chunks` — danh sách `nn.Module`, các chunk stage xen kẽ trên rank này.
- **Phương thức:** `forward(input)` · `schedule_interleaved_1f1b(microbatches)` · `train_step_interleaved_1f1b(microbatches, loss_fn=None)` · `parameters()` · `to(device)` · `sync()` · `num_chunks()`.
- **Trả về:** giống `PipelineParallelModule`, nhưng stage được chia thành các chunk xen kẽ (song song pipeline ảo); `num_chunks()` trả về số chunk.
- **PyTorch:** `torch.distributed.pipelining` với `num_chunks` — cùng khái niệm.

<!-- ex: id=r12-pipeline device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
stage = lt.nn.Sequential([lt.nn.Linear(4, 4, True), lt.nn.ReLU()])
pp = lt.distributed.PipelineParallelModule(stage)
x = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
print("pipeline forward shape:", pp.forward(x).shape)
chunks = [lt.nn.Linear(4, 4, True), lt.nn.Linear(4, 4, True)]
ip = lt.distributed.InterleavedPipelineModule(chunks)
print("interleaved chunks:", ip.num_chunks())
```

## Wrapper song song dữ liệu

### `distributed.DistributedDataParallel`

- **Chữ ký:** `DistributedDataParallel(module, op=ReduceOp.AVG)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:175-187`
- **Tham số:** `module` — một `nn.Module` cần nhân bản. `op` — một `ReduceOp`, mặc định `AVG` (gradient được trung bình trên các rank).
- **Thuộc tính (đọc/ghi):** `reduce_op`.
- **Phương thức:** `forward(input)` · `backward(loss)` / `backward(loss, grad_output)` · `sync_grads()` · `parameters()` · `to(device)`.
- **Trả về:** `forward` chạy bản sao; `backward` chạy backward và reduce gradient trên các rank; `sync_grads()` ép đồng bộ gradient ngoài `backward`.
- **Trường hợp biên:** process group phải được khởi tạo trước khi khởi tạo đối tượng. Tham số phải giống hệt nhau trên mọi rank trước `forward` đầu tiên — tự broadcast chúng hoặc load cùng một checkpoint ở mọi nơi.
- **Defect đã sửa (đã xác minh 2026-10-26, extension 88a4ba1, torch 2.14.1+cpu; công thức đã chốt 2026-10-27 theo dist-w21; đã sửa bởi dist-w20, merged @39e53ee0; Track 5 PASS 2026-10-31, PIN c195f43c, 2-GPU cell-20 happy + mismatch-throw):** `DDP.backward` nhân đôi một lát cắt của gradient trọng số. Với `Linear(in, out)`, gradient trọng số phẳng bị nhân đôi trên các chỉ số `[numel/4, numel/4 + out)` (tương đương `[numel/4, numel/4 + bias_numel)`) — ví dụ `Linear(4, 4)` cho `[0.5 x4, 1.0 x4, 0.5 x4, 0.5 x4]` thay vì toàn `0.5` (torch: toàn `0.5`). Ảnh hưởng cả `AVG` (hàng thành 1.0) và `SUM` (hàng thành 2.0); gradient bias đúng; `Linear.backward` đơn tiến trình đúng (toàn 0.5). Ví dụ trên in toàn bộ gradient để defect trước khi sửa hiện rõ. Đã sửa bởi dist-w20 (@39e53ee0); header của ví dụ vẫn ghi lại xác minh parity-fail trước khi sửa và đang chờ chạy lại trên tree đã sửa.
- **PyTorch:** `torch.nn.parallel.DistributedDataParallel` — cùng vai trò; DDP của torch cũng trung bình gradient theo mặc định, nhưng DDP của torch hook vào engine autograd tự động trong khi `backward()` của LiteTorch thực hiện phép reduce một cách tường minh.

<!-- ex: id=r12-ddp device=cpu backend=cpu runtime=~15s torch-parity=parity-fail status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
        model.to(cpu)
        ddp = lt.distributed.DistributedDataParallel(model)
        x = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
        loss = ddp.forward(x).sum()
        ddp.backward(loss)
        print("rank", rank, "ddp grad:", model.parameters()[0].grad.to_vector())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29613)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## FSDP

### `distributed.fsdp_fully_shard`

- **Chữ ký:** `fsdp_fully_shard(module) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:255` (`FSDP::fully_shard`, `fsdp_wrapper.cpp:192`)
- **Tham số:** `module` — một `nn.Module` cần shard.
- **Trả về:** `None`. Thay đổi module tại chỗ: đệ quy vào các leaf module có tham số, tạo một `FSDPUnit` cho mỗi leaf, và đăng ký forward hook (gather toàn bộ tham số trước forward, giải phóng sau) và gradient hook (reduce-scatter gradient khi backward). Sau lệnh gọi này, `model(x)` + `loss.backward()` thông thường cứ chạy — gather/giải phóng là tự động. `fsdp_set_prefetch_limit(n)` giới hạn số lần gather đang bay (mặc định 2).
- **Trường hợp biên:** process group phải được khởi tạo trước. Tham số được làm phẳng qua `FlatParameter` (một mảnh cục bộ cho mỗi rank); gradient được reduce-scatter. FSDP KHÔNG shard optimizer state — không có tham chiếu nào tới optimizer trong `fsdp_wrapper.cpp` / `fsdp_tp.cpp`.
- **PyTorch:** `torch.distributed.fsdp.fully_shard` — cùng điểm vào và cùng hợp đồng tại chỗ.

### `distributed.FSDPUnit`

- **Chữ ký:** class với các phương thức `prefetch_gather()` · `ensure_gathered()` · `release_gathered()` · `is_gathered()` — **không gọi được từ Python**
- **Nguồn:** `src/bindings/bind_distributed.cpp:248-252`
- **Trả về:** các phương thức tồn tại trên class C++, nhưng không có `py::init` nào được bind và `FSDP::active_units()` cũng không được bind, nên không có code Python nào lấy được một instance `FSDPUnit`. Khởi tạo một instance raise `TypeError: litetorch.distributed.FSDPUnit: No constructor defined!`. Đây là một khoảng trống API tiềm ẩn, không phải vấn đề tài liệu — dùng `fsdp_fully_shard()` thay thế.
- **PyTorch:** không có class tương đương được expose; các unit FSDP của torch cũng là nội bộ.

### `distributed.fsdp_set_prefetch_limit`

- **Chữ ký:** `fsdp_set_prefetch_limit(n) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:256`
- **Tham số:** `n` — int, số lần gather tham số đang bay tối đa (mặc định 2).
- **Trả về:** `None`.
- **PyTorch:** `torch.distributed.fsdp` không có tương đương trực tiếp; độ sâu prefetch được điều khiển bằng `prefetch_policy`.

### `distributed.FSDPTP`

- **Chữ ký:** class gồm các phương thức static: `init(tp_size, dp_size)` · `is_initialized()` · `tp_size()` · `dp_size()` · `tp_rank()` · `dp_rank()` · `fully_shard_tp(module)` · `state_dict(module)` · `load_state_dict(module, sd)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:259-268`
- **Trả về:** kết hợp FSDP+TP: `init` chia world thành TP group có `tp_size` và DP group có `dp_size`; `fully_shard_tp` shard một module trên cả hai trục; `state_dict` / `load_state_dict` gather / scatter toàn bộ state dict.
- **Trường hợp biên:** phải gọi `init` trước `fully_shard_tp`. Khác với `FSDPUnit`, `FSDPTP` hoàn toàn truy cập được từ Python (mọi static đều được bind).
- **PyTorch:** không có tương đương đơn lẻ; torch kết hợp FSDP2 với TP qua `device_mesh` + `parallelize_module()`.

<!-- ex: id=r12-fsdp device=cpu backend=cpu runtime=~15s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        model = lt.nn.Sequential([
            lt.nn.Linear(8, 8, True),
            lt.nn.GELU(),
            lt.nn.Linear(8, 8, True),
        ])
        model.to(cpu)
        lt.distributed.fsdp_fully_shard(model)
        x = lt.Tensor.from_vector([0.1] * 16, [2, 8], cpu)
        loss = model(x).sum()
        loss.backward()
        print("rank", rank, "fsdp loss:", loss.item())
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(r, 2, 29611)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)



if __name__ == "__main__":
    main()
```

## Cầu nối NCCL

### `distributed.NCCLBridge`

- **Chữ ký:** class, singleton qua `NCCLBridge.get()`
- **Nguồn:** `src/bindings/bind_distributed.cpp:163-173`
- **Phương thức:** `get()` (static) · `is_available()` · `init(rank, world_size)` · `shutdown()` · `sync_comm()`.
- **Trả về:** `is_available()` báo cáo backend NCCL native có dùng được không; `init` nối NCCL communicator cho group; `sync_comm()` đồng bộ communicator.
- **Trường hợp biên:** trên máy chỉ có CPU, `is_available()` là False và `init` là no-op. Thư viện NCCL được tìm bằng dlopen thuần (xem ghi chú ở đầu trang — lựa chọn `LITETORCH_NCCL_LIB` là patch chưa merge); thư viện thiếu báo cáo không khả dụng thay vì fail-closed tại `init`.
- **PyTorch:** backend NCCL của `torch.distributed` — cùng vai trò, nhưng torch chọn NCCL qua đối số `backend="nccl"` trong khi LiteTorch tìm bằng dlopen thuần (override `LITETORCH_NCCL_LIB` là patch chưa merge — xem ghi chú ở đầu trang).

## Checkpoint phân mảnh

### `distributed.TensorShardInfo` / `distributed.CheckpointMetadata`

- **Chữ ký:** class với các thuộc tính đọc/ghi. `TensorShardInfo`: `global_shape`, `dtype_str`, `shard_dim`, `shard_ranges`. `CheckpointMetadata`: `version`, `world_size`, `step`, `timestamp`, `tensors` (dict tên tới `TensorShardInfo`).
- **Nguồn:** `src/bindings/bind_distributed.cpp:200-214`
- **Trả về:** các data holder mô tả một mảnh của một checkpoint phân tán.
- **PyTorch:** metadata của `torch.distributed.checkpoint` — cùng vai trò, hình thức khác.

### `distributed.make_sharded_spec` / `distributed.make_replicated_spec`

- **Chữ ký:** `make_sharded_spec(global_shape, dtype_str, shard_dim, world_size) -> TensorShardInfo`; `make_replicated_spec(global_shape, dtype_str) -> TensorShardInfo`
- **Nguồn:** `src/bindings/bind_distributed.cpp:215-218`
- **Tham số:** `global_shape` — danh sách int, shape đầy đủ của tensor. `dtype_str` — ví dụ `"float32"`. `shard_dim` — chiều được shard trên các rank. `world_size` — số rank.
- **Trả về:** một `TensorShardInfo` mô tả bố cục shard.
- **PyTorch:** không có tương đương trực tiếp; torch suy ra sharding từ placement của `DTensor`.

### `distributed.save_distributed` / `distributed.load_distributed` / `distributed.read_dist_metadata`

- **Chữ ký:** `save_distributed(local_shards, shard_specs, ckpt_dir, step) -> None`; `load_distributed(local_shards, ckpt_dir) -> None`; `read_dist_metadata(ckpt_dir) -> CheckpointMetadata`
- **Nguồn:** `src/bindings/bind_distributed.cpp:219-236`
- **Tham số:** `local_shards` — dict tên tới shard tensor của rank này. `shard_specs` — dict tên tới `TensorShardInfo`. `ckpt_dir` — đường dẫn thư mục. `step` — bước checkpoint int.
- **Trả về:** `save_distributed` ghi một file cho mỗi rank cộng metadata; `load_distributed` điền các shard cục bộ từ đĩa; `read_dist_metadata` trả về metadata đã lưu.
- **Trường hợp biên:** `load_distributed` yêu cầu các shard cục bộ đã có đúng shape. `ckpt_dir` được tạo nếu chưa tồn tại.
- **PyTorch:** `torch.distributed.checkpoint.save` / `load_state_dict` — cùng vai trò.

<!-- ex: id=r12-checkpoint device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import tempfile
import litetorch as lt

cpu = lt.Device("cpu")
t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
spec = lt.distributed.make_sharded_spec([4], "float32", 0, 1)
with tempfile.TemporaryDirectory() as d:
    lt.distributed.save_distributed({"w": t}, {"w": spec}, d, 1)
    meta = lt.distributed.read_dist_metadata(d)
    print("meta step:", meta.step, "world:", meta.world_size)
    t2 = lt.Tensor.zeros([4], cpu)
    lt.distributed.load_distributed({"w": t2}, d)
    print("loaded:", t2.to_vector())
```

### `distributed.FlatParameter` / `distributed.make_flat_param`

- **Chữ ký:** `FlatParameter(params)`; `make_flat_param(params) -> FlatParameter`
- **Nguồn:** `src/bindings/bind_distributed.cpp:238-250`
- **Tham số:** `params` — danh sách tensor cần làm phẳng thành một bộ đệm.
- **Phương thức:** `flat_numel()` · `shard_numel()` · `unpadded_numel()` · `world_size()` · `rank()` · `local_shard()` · `full_buffer()` · `param_views()` · `all_gather_flat()` · `reduce_scatter_flat(full_grad)` · `metadata()` (danh sách dict `{offset, numel, shape}`).
- **Trả về:** một bộ đệm phẳng duy nhất chứa mọi tham số liên tiếp, với các view cho từng tham số. `local_shard()` là lát cắt của rank này; `full_buffer()` gather mọi lát cắt. `param_views()` yêu cầu `all_gather_flat()` trước — nếu không nó raise `[litetorch Error] FlatParameter::param_views: full buffer not allocated; call all_gather_flat() first`. Được FSDP dùng nội bộ; có thể dùng trực tiếp cho sharding tùy chỉnh.
- **Trường hợp biên:** `world_size()` / `rank()` đọc từ process group; không có group đã khởi tạo chúng báo cáo mặc định đơn tiến trình.
- **PyTorch:** `torch.distributed.fsdp.FlatParameter` — cùng khái niệm.

<!-- ex: id=r12-flatparam device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
a = lt.Tensor.from_vector([1.0, 2.0], [2], cpu)
b = lt.Tensor.from_vector([3.0, 4.0, 5.0], [3], cpu)
fp = lt.distributed.make_flat_param([a, b])
fp.all_gather_flat()
print("flat numel:", fp.flat_numel(), "views:", len(fp.param_views()))
print("metadata:", fp.metadata())
```

## Khởi động lại đàn hồi

### `distributed.RendezvousStore`

- **Chữ ký:** `RendezvousStore(bind_addr, port)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:272-280`
- **Tham số:** `bind_addr` — địa chỉ cần bind, ví dụ `"127.0.0.1"`. `port` — cổng cần lắng nghe.
- **Phương thức:** `port()` · `run()` (blocking; giải phóng GIL — chạy trong một thread) · `stop()`.
- **Trả về:** một kho key-value mà các rank dùng để tìm nhau và publish heartbeat. `port()` trả về cổng đã bind (hữu ích khi `port=0` yêu cầu hệ điều hành chọn).
- **Trường hợp biên:** `run()` chặn cho đến khi `stop()` được gọi từ một thread khác. Chỉ một store cho mỗi cặp địa chỉ/cổng. Join thread đang chạy `run()` sau `stop()` trước khi interpreter thoát — nếu không tiến trình có thể abort khi shutdown với `terminate called without an active exception` (đã xác minh: ví dụ abort khi không có join, và pass 3/3 khi có join).
- **PyTorch:** `torch.distributed.elastic.rendezvous` (store `etcd` / `c10d`) — cùng vai trò; store của LiteTorch là một store TCP nhẹ duy nhất.

### `distributed.RendezvousClient`

- **Chữ ký:** `RendezvousClient(addr, port, timeout_sec=30)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:281-290`
- **Tham số:** `addr` — địa chỉ store. `port` — cổng store. `timeout_sec` — mặc định 30.
- **Phương thức:** `set(key, value)` · `get(key, timeout_sec=-1)` · `add(key, delta)` (tăng nguyên tử, trả về giá trị mới) · `heartbeat(rank)` · `get_heartbeats()`.
- **Trả về:** `get` chặn tối đa `timeout_sec` (-1 chờ mãi mãi); `add` là nguyên tử trên các rank — dùng nó để gán rank.
- **PyTorch:** client store của `torch.distributed.elastic.rendezvous` — cùng các phép toán.

### `distributed.rendezvous_free_port`

- **Chữ ký:** `rendezvous_free_port() -> int`
- **Nguồn:** `src/bindings/bind_distributed.cpp:291`
- **Trả về:** một cổng cục bộ hiện đang trống, dùng cho `init_process_group` hoặc một `RendezvousStore`.
- **Trường hợp biên:** cổng có thể bị chiếm giữa lúc gọi và lúc bind (TOCTOU) — thử lại khi bind thất bại.
- **PyTorch:** không có tương đương trực tiếp; người dùng torch bind cổng 0 hoặc dùng `torch.distributed.elastic.utils`.

<!-- ex: id=r12-rendezvous device=cpu backend=cpu runtime=~5s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import threading
import time
import litetorch as lt

port = lt.distributed.rendezvous_free_port()
store = lt.distributed.RendezvousStore("127.0.0.1", port)
t = threading.Thread(target=store.run, daemon=True)
t.start()
time.sleep(0.5)
client = lt.distributed.RendezvousClient("127.0.0.1", port)
client.set("world_size", "2")
print("world_size:", client.get("world_size"))
print("rank counter:", client.add("next_rank", 1))
client.heartbeat(0)
print("heartbeats:", client.get_heartbeats())
store.stop()
t.join(timeout=2.0)
print("store thread joined:", not t.is_alive())
```

### `distributed.HeartbeatAgent`

- **Chữ ký:** `HeartbeatAgent(store_addr, store_port, rank, interval_ms=1000)`
- **Nguồn:** `src/bindings/bind_distributed.cpp:293-300`
- **Tham số:** `store_addr` / `store_port` — rendezvous store. `rank` — rank này. `interval_ms` — khoảng heartbeat, mặc định 1000.
- **Phương thức:** `start()` · `stop()` · `running()`.
- **Trả về:** một agent nền heartbeat tới store mỗi `interval_ms` cho đến `stop()`. `running()` báo cáo thread agent có đang sống không.
- **PyTorch:** heartbeat của `torch.distributed.elastic.agent` — cùng vai trò.

### `distributed.detect_dead_ranks`

- **Chữ ký:** `detect_dead_ranks(store_addr, store_port, timeout_ms) -> list[int]`
- **Nguồn:** `src/bindings/bind_distributed.cpp:301-302`
- **Tham số:** `store_addr` / `store_port` — rendezvous store. `timeout_ms` — một rank được coi là chết nếu heartbeat cuối của nó cũ hơn giá trị này.
- **Trả về:** danh sách các rank đã chết.
- **PyTorch:** phát hiện lỗi của `torch.distributed.elastic` — cùng vai trò.

### `distributed.save_consistent` / `distributed.latest_consistent_step` / `distributed.load_latest`

- **Chữ ký:** `save_consistent(local_shards, shard_specs, ckpt_dir, step) -> None`; `latest_consistent_step(ckpt_dir) -> int`; `load_latest(local_shards, ckpt_dir) -> None`
- **Nguồn:** `src/bindings/bind_distributed.cpp:303-316`
- **Tham số:** cùng đối số shard như `save_distributed`; `step` — bước checkpoint int.
- **Trả về:** `save_consistent` chỉ ghi checkpoint khi mọi rank đã báo cáo (nên một lần khởi động lại không bao giờ load checkpoint viết dở); `latest_consistent_step` trả về bước nhất quán mới nhất, hoặc -1 nếu không có; `load_latest` load nó.
- **Trường hợp biên:** dùng các hàm này (không phải `save_distributed`) cho checkpoint phải sống sót qua một lần crash giữa lúc ghi.
- **PyTorch:** `torch.distributed.checkpoint` với atomic commit — cùng đảm bảo, API khác.
