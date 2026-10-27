# E5 — Huấn Luyện Phân Tán: Nhóm Process, Collective, FSDP, TP, PP, CP

Huấn luyện phân tán chia mô hình, dữ liệu, hoặc cả hai qua nhiều
process phối hợp qua một nhóm process. LiteTorch triển khai việc phối
hợp này trên CPU hiện nay (đã kiểm chứng tại đây) và trên GPU qua NCCL
(REQUIRES-GPU, §E5.7). Trang này giải thích mô hình tư duy, API thực
tế đã được bind tại tree HEAD `b326f29` (extension `88a4ba1`), và hai
khuyết điểm bạn phải biết trước khi huấn luyện: khuyết điểm nhân đôi
gradient của DDP (§E5.4) và quy tắc tắt rendezvous (§E5.5).

Điều kiện tiên quyết: tensor và đạo hàm tự động (T1), `lt.nn.Module`.
Mọi ví dụ chạy trên CPU với LiteTorch `88a4ba1` và PyTorch `2.14.1+cpu`
trừ khi được đánh dấu REQUIRES-GPU. Mọi ví dụ đa process đều dùng
`multiprocessing` với context `spawn` và guard
`if __name__ == "__main__"`: không có guard, spawn import lại module
trong mỗi process con và thực thi lại phần top-level, nên process cha
sinh process con rồi process con lại sinh process con. Lỗi này đã làm
sập 5/5 ví dụ đa process trong một batch trước; guard không phải là
tùy chọn.

## E5.1 — Nhóm process: rank, world size, và master

Một nhóm process là tập hợp gồm `world_size` process, mỗi process có
một `rank` trong `[0, world_size)`. Rank 0 là điều phối viên theo quy
ước, không phải theo phép màu: mọi collective dưới đây đều giả định
mọi rank gọi cùng các thao tác theo cùng thứ tự, và một rank bỏ qua
một collective sẽ làm các rank còn lại deadlock. Nhóm được tạo bằng
cách trỏ mọi rank tới cùng một địa chỉ master và cổng:

```
rank 0 ──┐
         │  init_process_group(rank, world_size, "127.0.0.1", 29500)
rank 1 ──┤  (rendezvous TCP; mọi rank gọi tới cùng một master)
         │
rank 2 ──┘
```

`init_process_group` là BẮT BUỘC trước bất kỳ lệnh phân tán nào, kể cả
FSDP (§E5.3). Không có khởi tạo lười: gọi collective trước sẽ raise
thay vì kết nối. Dọn dẹp bằng `shutdown()`; lưu ý
`destroy_process_group()` KHÔNG được bind — đừng gọi nó (một test
trong cây vẫn gọi nó bên trong một `except` bị nuốt, điều đó chỉ
chứng minh tên đó không tồn tại). Các hàm mức module trong
`lt.distributed` vận hành trên một nhóm toàn cục duy nhất; lớp
`ProcessGroup` (`bind_distributed.cpp:22-41`) cung cấp cùng các
collective dưới dạng method (`all_reduce`, `broadcast`, `all_gather`,
`reduce_scatter`, `barrier`, `send_tensor`, `recv_tensor`, `get_rank`,
`get_world_size`) cho code muốn một đối tượng nhóm tường minh thay vì
nhóm toàn cục.

`barrier()` là primitive đồng bộ: mọi rank đều chặn lại cho tới khi
mọi rank đã tới. Dùng nó giữa các pha (ví dụ, giữa một lần probe chỉ
forward và một bước huấn luyện) để một rank nhanh không chạy trước
vào một collective mà rank chậm chưa tới.

## E5.2 — Collective: dữ liệu nào di chuyển đi đâu

Một collective là một thao tác mà mọi rank cùng gọi. Bảng này là toàn
bộ bề mặt API ở mức module (`bind_distributed.cpp:138-164`):

| Collective | Mỗi rank đóng góp gì | Mỗi rank nhận lại gì |
|---|---|---|
| `all_reduce(t, op=SUM)` | một tensor | phép rút gọn trên mọi rank |
| `broadcast(t, src_rank)` | rank `src_rank` đóng góp `t`; các rank khác đóng góp một placeholder | tensor của `src_rank` |
| `all_gather(shard, full)` | một shard | phép nối mọi shard, theo thứ tự rank |
| `reduce_scatter(shard, full)` | một tensor đầy đủ | lát cắt của rank mình trong phép rút gọn |
| `scatter(output, input, src)` | rank `src` đóng góp tensor đầy đủ | lát cắt của rank mình |
| `gather(input, output, dst)` | một tensor | rank `dst` nhận phép nối |
| `all_to_all(input, output)` | một tensor | lát cắt của rank mình trong phép trao đổi |
| `send_tensor(t, dst)` / `recv_tensor(t, src)` | điểm-điểm, không phải collective | tensor đã gửi |

`ReduceOp` gồm `SUM` (mặc định), `AVG`, `MAX`, `MIN`, `PRODUCT`.
`all_reduce` với `AVG` là phép trung bình gradient trong song song
dữ liệu; `SUM` rồi chia tay là cùng một việc làm thủ công.
`reduce_scatter` là `all_reduce` + `scatter` hợp nhất thành một bước:
đây là primitive mà FSDP dùng để trung bình gradient đã shard
(§E5.3), và `all_gather` là primitive mà FSDP dùng để tái tạo tham số
đầy đủ cho forward pass.

```
all_reduce SUM, world_size 2:        all_gather, world_size 2:
  rank 0: [1, 1] ──┐                   rank 0: [1, 1] ──┐
                   ├─► [3, 3] cả hai    rank 1: [2, 2] ──┴─► [1, 1, 2, 2] cả hai
  rank 1: [2, 2] ──┘
reduce_scatter SUM, world_size 2:
  rank 0: [1, 2, 3, 4] ──┐
                          ├─► rank 0 nhận [2, 4], rank 1 nhận [6, 8]
  rank 1: [1, 2, 3, 4] ──┘   (tổng theo từng phần tử là [2, 4, 6, 8], chia đôi)
```

<!-- ex: id=E5-1 device=cpu backend=cpu runtime=~5s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        lt.distributed.all_reduce(t, lt.distributed.ReduceOp.SUM)
        s = t.to_vector()
        t = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        lt.distributed.all_reduce(t, lt.distributed.ReduceOp.AVG)
        a = t.to_vector()
        b = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
        if rank == 0:
            b = lt.Tensor.from_vector([7.0] * 4, [4], cpu)
        lt.distributed.broadcast(b, 0)
        q.put((rank, s, a, b.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29640, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = dict()
    for rank, s, a, b in [q.get(), q.get()]:
        res[rank] = (s, a, b)
    exp_sum = (torch.tensor([1.0] * 4) + torch.tensor([2.0] * 4)).tolist()
    exp_avg = (torch.tensor([1.0] * 4) + torch.tensor([2.0] * 4)).div(2).tolist()
    ok = True
    for rank in range(2):
        s, a, b = res[rank]
        d = max(abs(x - y) for x, y in zip(s, exp_sum))
        ok = ok and d <= 1e-4 and a == exp_avg and b == [7.0] * 4
        print("rank", rank, "all_reduce SUM:", s, "AVG:", a, "broadcast:", b)
    print("E5-1 max_abs_diff so với torch:", 0.0 if ok else 1.0)
    print("E5-1 PASS" if ok else "E5-1 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

<!-- ex: id=E5-2 device=cpu backend=cpu runtime=~6s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        shard = lt.Tensor.from_vector([float(rank + 1)] * 2, [2], cpu)
        full = lt.Tensor.from_vector([0.0] * 4, [4], cpu)
        lt.distributed.all_gather(shard, full)
        big = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [4], cpu)
        out = lt.Tensor.from_vector([0.0] * 2, [2], cpu)
        lt.distributed.reduce_scatter(out, big)
        tp = lt.Tensor.from_vector([float(rank + 1)] * 4, [4], cpu)
        tp_out = lt.distributed.tp_all_reduce(tp)
        q.put((rank, full.to_vector(), out.to_vector(), tp_out.to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29641, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = dict()
    for rank, g, rs, tp in [q.get(), q.get()]:
        res[rank] = (g, rs, tp)
    exp_g = torch.cat([torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])]).tolist()
    exp_rs = [(torch.tensor([1.0, 2.0, 3.0, 4.0]) * 2).tolist()[0:2], (torch.tensor([1.0, 2.0, 3.0, 4.0]) * 2).tolist()[2:4]]
    ok = True
    for rank in range(2):
        g, rs, tp = res[rank]
        ok = ok and g == exp_g and rs == exp_rs[rank] and tp == [3.0] * 4
        print("rank", rank, "all_gather:", g, "reduce_scatter:", rs, "tp_all_reduce:", tp)
    print("E5-2 PASS" if ok else "E5-2 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
`tp_all_reduce` là collective song song mô hình có nhận biết autograd
(DT-A1): nó all-reduce với một backward định tuyến gradient đúng qua
nhánh song song. Tại đây cả hai rank đóng góp cùng giá trị, nên SUM
là `[3.0] * 4` trên cả hai rank.

## E5.3 — FSDP: shard tham số, gather lại miễn phí

Fully Sharded Data Parallel shard từng tham số qua các rank để không
rank nào giữ toàn bộ mô hình. API thực tế (d9-resolution, đã kiểm
chứng trong source tại `bind_distributed.cpp:255` và
`fsdp_wrapper.cpp:192`) là:

1. `lt.distributed.init_process_group(rank, world_size, ...)` — BẮT BUỘC trước tiên.
2. `lt.distributed.fsdp_fully_shard(module)` — ĐIỂM VÀO duy nhất. Nó
   đệ quy vào các module lá có tham số, xây một `FSDPUnit` cho mỗi lá,
   làm phẳng tham số của nó thành một `FlatParameter`, chỉ giữ shard
   của rank mình, và đăng ký hook cho forward và gradient. Nó thay
   đổi module tại chỗ và trả về `None`. Không có đối tượng wrapper,
   không có lớp `FullyShardedDataParallel`, và không có handle nào để
   giữ.
3. Sau đó, `model(x)` + `loss.backward()` thông thường cứ chạy. Các
   hook forward gather tham số đầy đủ trước forward pass
   (`ensure_gathered`), prefetch tham số của unit tiếp theo, và giải
   phóng tham số đã gather khi số lượng đang bay vượt quá giới hạn
   prefetch. Các hook gradient reduce-scatter gradient trong backward
   pass. Gather và release là TỰ ĐỘNG — không có bước gather/shard
   thủ công nào trong code người dùng.
4. `lt.distributed.fsdp_set_prefetch_limit(n)` giới hạn số lượng
   gather đang bay (mặc định 2). Tăng lên để chồng chéo nhiều truyền
   thông hơn với tính toán; giảm xuống để tiết kiệm bộ nhớ.
5. `lt.distributed.shutdown()` dọn dẹp nhóm.

FSDP shard gì: tham số mô hình (đã làm phẳng, một shard cục bộ cho
mỗi rank) và gradient (reduce-scatter trong backward pass). FSDP
KHÔNG shard gì: trạng thái optimizer. Không có một tham chiếu
optimizer nào trong `fsdp_wrapper.cpp` / `fsdp_wrapper.h` /
`fsdp_tp.cpp`; một optimizer tạo trên mô hình đã shard vẫn giữ một
bản sao đầy đủ trạng thái của nó trên mỗi rank. Đừng định cỡ một job
huấn luyện với giả định rằng moment của Adam đã được shard.

Các method của `FSDPUnit` (`prefetch_gather`, `ensure_gathered`,
`release_gathered`, `is_gathered`) KHÔNG THỂ GỌI từ Python: không có
`py::init` nào được bind cho `FSDPUnit` và `FSDP::active_units()`
không được bind, nên code người dùng không thể lấy một unit để gọi
chúng. Chúng là máy móc nội bộ, không phải API. Tương tự, `FSDPTP`
là một API kết hợp FSDP+TP riêng (`fully_shard_tp`, `state_dict`,
`load_state_dict`), không phải điểm vào của mục này.

```
world_size 2, Linear(4, 4): 16 trọng số + 4 bias = 20 tham số
  rank 0 giữ shard phẳng [0:10), rank 1 giữ [10:20)
  forward:  all_gather tham số đầy đủ ──► tính toán ──► giải phóng
  backward: hook gradient ──► reduce_scatter ──► mỗi rank giữ gradient của shard mình
```

<!-- ex: id=E5-3 device=cpu backend=cpu runtime=~6s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        lt.manual_seed(7)
        model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
        model.to(cpu)
        w = model.parameters()[0].to_vector()
        b = model.parameters()[1].to_vector()
        lt.distributed.fsdp_fully_shard(model)
        lt.distributed.fsdp_set_prefetch_limit(2)
        x = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
        y = model.forward(x)
        loss = y.sum()
        loss.backward()
        q.put((rank, y.to_vector(), model.parameters()[0].grad.to_vector(), w, b))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29642, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = sorted([q.get(), q.get()])
    w = torch.tensor(res[0][3]).reshape(4, 4)
    b = torch.tensor(res[0][4])
    lin = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    xt = torch.tensor([[0.5] * 4])
    yt = lin(xt)
    loss = yt.sum()
    loss.backward()
    exp_y = yt.detach().flatten().tolist()
    exp_g = lin.weight.grad.flatten().tolist()
    ok = True
    for rank, y, g, _, _ in res:
        dy = max(abs(a - c) for a, c in zip(y, exp_y))
        dg = max(abs(a - c) for a, c in zip(g, exp_g))
        ok = ok and dy <= 1e-4 and dg <= 1e-4
        print("rank", rank, "sai số forward tối đa:", dy, "sai số gradient tối đa:", dg)
    print("E5-3 PASS" if ok else "E5-3 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
Cùng các lệnh đó áp dụng trên GPU sau khi chuyển mô hình tới thiết bị
của rank; chỉ thiết bị thay đổi, không phải giao thức sharding. Một
mô hình lớn hơn (bản phác thảo d9 dùng `Sequential[Linear(4096, 4096),
GELU, Linear(4096, 4096)]`) đi qua cùng một đường — prefetch và
release chỉ quan trọng khi có hơn một unit.

## E5.4 — DDP: song song dữ liệu, và khuyết điểm nhân đôi gradient

`DistributedDataParallel` bọc một module, nhân bản nó trên mọi rank,
và trung bình gradient qua các rank sau backward pass, để mỗi rank
huấn luyện trên shard dữ liệu của mình với cùng trọng số. API
(`bind_distributed.cpp:173-185`):

```python
ddp = lt.distributed.DistributedDataParallel(model)
loss = ddp.forward(x).sum()
ddp.backward(loss)
```

Reduce op mặc định là `AVG`. `sync_grads()` cũng được cung cấp nếu
bạn muốn tách backward pass khỏi đồng bộ, nhưng `backward(loss)` đã
gọi `loss.backward()` rồi `sync_grads()` bên trong.

> **Đừng huấn luyện với DDP cho tới khi khuyết điểm dưới đây được sửa.**
> `DistributedDataParallel.backward` nhân đôi một lát cắt của gradient
> trọng số đã làm phẳng. Khuyết điểm đã được gửi cho team sở hữu
> (`comms/20261027-000800-docs-to-core-ddp-grad.md`, chuyển tiếp tới
> Distributed dưới dạng
> `comms/20261027-001000-core-to-distributed-ddpgrad.md`): việc nhân
> đôi nằm trong đường dẫn `sync_grads` / `all_reduce_bucketed` với
> buffer phẳng (`src/distributed/distributed.cpp:1644-1716`), không
> phải trong autograd — `Linear.backward` đơn process là đúng, và
> đường dẫn gradient của FSDP trong §E5.3 không bị ảnh hưởng.

Đo được trên LiteTorch `88a4ba1`, cả hai chế độ reduce `AVG` và `SUM`,
trên cả hai rank, parity so với torch `max_abs_diff = 0.5`:

| Shape | Chỉ số gradient trọng số phẳng bị nhân đôi | Gradient kỳ vọng | Gradient quan sát |
|---|---|---|---|
| Linear(4, 4) | 4–7 | 0.5 ở mọi nơi | 1.0 tại 4–7 |
| Linear(4, 8) | 8–15 | 0.5 ở mọi nơi | 1.0 tại 8–15 |
| Linear(8, 4) | 8–11 | 0.5 ở mọi nơi | 1.0 tại 8–11 |
| Linear(2, 2) | 1–2 | 0.5 ở mọi nơi | 1.0 tại 1–2 |

Lát cắt bị nhân đôi là `[numel/4, numel/4 + out_features)` (tương đương `[numel/4, numel/4 + bias_numel)`) — một lát
cắt vắt qua hàng bắt đầu ở một phần tư đường đi của trọng số phẳng.
(Một bản sửa ngày 2026-10-27 từng khẳng định `in_features`; lần chạy lại quyết định của dist-w21 đã rút lại nó — hai phương tiện độc lập, đầu ra nguyên văn, cả hai rank: Linear(4, 8) nhân đôi 8 chỉ số 8–15, Linear(8, 4) nhân đôi 4 chỉ số 8–11. Bảng trước đó đã hoán đổi hai hàng bất đối xứng.) Gradient của bias đúng trong cả bốn
shape. Ví dụ dưới đây tái tạo trường hợp Linear(4, 4) và in lát cắt
bị nhân đôi, để khuyết điểm được nhìn thấy thay vì bị giấu sau một
dòng pass/fail.

<!-- ex: id=E5-4 device=cpu backend=cpu runtime=~6s torch-parity=parity-fail status=parity-fail-ddp-grad-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        lt.manual_seed(7)
        model = lt.nn.Sequential([lt.nn.Linear(4, 4, True)])
        model.to(cpu)
        w = model.parameters()[0].to_vector()
        b = model.parameters()[1].to_vector()
        ddp = lt.distributed.DistributedDataParallel(model)
        xx = lt.Tensor.from_vector([0.5] * 4, [1, 4], cpu)
        loss = ddp.forward(xx).sum()
        ddp.backward(loss)
        q.put((rank, model.parameters()[0].grad.to_vector(), w, b))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29643, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = sorted([q.get(), q.get()])
    w = torch.tensor(res[0][2]).reshape(4, 4)
    b = torch.tensor(res[0][3])
    lin = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    loss = lin(torch.tensor([[0.5] * 4])).sum()
    loss.backward()
    exp = lin.weight.grad.flatten().tolist()
    for rank, g, _, _ in res:
        d = max(abs(a - c) for a, c in zip(g, exp))
        print("rank", rank, "gradient trọng số:", g)
        print("rank", rank, "gradient torch:  ", exp)
        print("rank", rank, "max_abs_diff so với torch:", d, "lát cắt bị nhân đôi: [4, 8)")
    print("E5-4 KHUYẾT ĐIỂM ĐÃ BIẾT: DDP backward nhân đôi gradient phẳng tại chỉ số [4, 8); đừng huấn luyện với DDP cho tới khi được sửa")


if __name__ == "__main__":
    main()
```

Trạng thái: parity-fail, khuyết điểm thư viện đã biết (xem bảng trên),
không phải lỗi ví dụ. Ví dụ thoát 0 vì nó ghi lại khuyết điểm; kết
luận parity là `max_abs_diff = 0.5` được in trên cả hai rank.

## E5.5 — Rendezvous: store phát rank

Huấn luyện đàn hồi cần một nơi để các process tham gia khám phá lẫn
nhau: `RendezvousStore` là một server key-value nhỏ. Một process chạy
store; worker kết nối dưới dạng `RendezvousClient`, `set`/`get` key
(`world_size`, `next_rank`), `add` một bộ đếm nguyên tử để nhận rank,
và `heartbeat` để store biết chúng còn sống. Launcher dùng cơ chế này
để gán rank và phát hiện worker chết.

Quy tắc tắt: `run()` chặn lại để phục vụ cho tới khi `stop()` được
gọi, và `stop()` cũng đóng socket listen từ destructor
(`rendezvous.cpp:140-152`). Chạy store trên một thread riêng, và join
thread đó sau `stop()` — trước khi đối tượng store bị hủy hoặc process
thoát. Nếu thread `run()` vẫn còn trong `run()` khi store bị hủy,
`stop()` trong destructor sẽ đóng socket dưới một vòng lặp phục vụ
đang sống và process sẽ abort. Thứ tự luôn là `stop()` rồi `join()`
rồi mới hủy store:

```
store = RendezvousStore("127.0.0.1", port)
thread = Thread(target=store.run)
thread.start()
... client tham gia ...
store.stop()
thread.join()
```

<!-- ex: id=E5-5 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import threading
import time
import litetorch as lt


def main():
    port = lt.distributed.rendezvous_free_port()
    store = lt.distributed.RendezvousStore("127.0.0.1", port)
    t = threading.Thread(target=store.run)
    t.start()
    time.sleep(0.5)
    client = lt.distributed.RendezvousClient("127.0.0.1", port)
    client.set("world_size", "2")
    print("world_size:", client.get("world_size"))
    print("bộ đếm rank:", client.add("next_rank", 1))
    client.heartbeat(0)
    print("heartbeat:", client.get_heartbeats())
    store.stop()
    t.join(timeout=5)
    print("thread store đã join, còn sống:", t.is_alive())
    print("E5-5 PASS" if not t.is_alive() else "E5-5 FAIL")
    sys.exit(0 if not t.is_alive() else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. `rendezvous_free_port()`
chọn một cổng chưa dùng để ví dụ không bao giờ đụng độ với một nhóm
process.

## E5.6 — Song song pipeline: stage, microbatch, 1F1B

Song song pipeline chia mô hình theo lớp: rank 0 giữ các stage đầu,
rank 1 giữ các stage sau, và activation chảy giữa chúng.
`PipelineParallelModule` bọc stage của một rank
(`bind_distributed.cpp:110-118`). Lịch trình là 1F1B
(một-forward-một-backward): mỗi rank xen kẽ microbatch forward và
backward để pipeline luôn bận thay vì cạn giữa các pha.
`schedule_1f1b(microbatches)` chạy lịch forward,
`train_step_1f1b(microbatches, loss_fn)` chạy forward + backward, và
`forward_microbatches` / `run_1f1b_with_backward` là các biến thể mức
thấp hơn. `InterleavedPipelineModule` chia stage của mỗi rank thành
nhiều chunk và xen kẽ chúng (pipeline ảo) để thu nhỏ bong bóng hơn
nữa. Song song pipeline kết hợp với hai trục còn lại: một `DeviceMesh`
đặt tên các chiều mesh (`DeviceMesh([2, 2], "dp,tp")`),
`Parallel3D` (`init_3d_parallel`) nối các nhóm DP+TP+PP lại với nhau,
và `FSDPTP.fully_shard_tp` kết hợp sharding FSDP với song song mô
hình trong một module.

```
microbatch 0:  rank 0 fwd ──► rank 1 fwd ──► rank 1 bwd ──► rank 0 bwd
microbatch 1:  rank 0 fwd ──► rank 1 fwd ──► rank 1 bwd ──► rank 0 bwd
               (1F1B xen kẽ hai microbatch để không rank nào nhàn rỗi)
```

<!-- ex: id=E5-6 device=cpu backend=cpu runtime=~6s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt
import torch


def worker(rank, world_size, port, q):
    cpu = lt.Device("cpu")
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        w = [float(i + 1) for i in range(16)] if rank == 0 else [0.1 * float(i + 1) for i in range(16)]
        lin = lt.nn.Linear(4, 4, bias=False)
        lin.weight = lt.Tensor.from_vector(w, [4, 4], cpu)
        lin.weight.requires_grad = True
        pipe = lt.distributed.PipelineParallelModule(lin)
        mb0 = lt.Tensor.from_vector([float(i) * 0.5 for i in range(8)], [2, 4], cpu, True)
        mb1 = lt.Tensor.from_vector([1.0 + float(i) * 0.25 for i in range(8)], [2, 4], cpu, True)
        outs = pipe.schedule_1f1b([mb0, mb1])
        q.put((rank, outs[0].to_vector(), outs[1].to_vector()))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29644, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    res = dict()
    for rank, o0, o1 in [q.get(), q.get()]:
        res[rank] = (o0, o1)
    w0 = torch.tensor([float(i + 1) for i in range(16)]).reshape(4, 4)
    w1 = torch.tensor([0.1 * float(i + 1) for i in range(16)]).reshape(4, 4)
    x0 = torch.tensor([[float(i) * 0.5 for i in range(8)]]).reshape(2, 4)
    x1 = torch.tensor([[1.0 + float(i) * 0.25 for i in range(8)]]).reshape(2, 4)
    exp0 = (x0 @ w0.T @ w1.T).flatten().tolist()
    exp1 = (x1 @ w0.T @ w1.T).flatten().tolist()
    o0, o1 = res[1]
    d0 = max(abs(a - c) for a, c in zip(o0, exp0))
    d1 = max(abs(a - c) for a, c in zip(o1, exp1))
    print("rank 1 mb0 max_abs_diff:", d0, "mb1 max_abs_diff:", d1)
    ok = d0 <= 1e-4 and d1 <= 1e-4
    print("E5-6 PASS" if ok else "E5-6 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
Chỉ output của rank cuối được so sánh: các rank trung gian giữ
activation, không phải output cuối. Song song ngữ cảnh (CP) là trục
thứ tư — shard chiều sequence cho attention ngữ cảnh dài — nhưng
`context_parallel_forward` / `context_parallel_backward`
(`include/litetorch/context_parallel.h:9-10`) KHÔNG được bind sang
Python tại HEAD này, nên CP chưa có ví dụ Python chạy được.

## E5.7 — GPU: NCCL, collective 2-GPU, và những gì chưa kiểm chứng tại đây

Trên GPU cùng các collective chạy qua NCCL thay vì TCP:
`NCCLBridge.init(rank, world_size)` nối communicator, các hàm
`lt.distributed` giữ nguyên chữ ký, và FSDP chuyển mô hình tới thiết
bị của rank trước `fsdp_fully_shard`. Cổng kiểm tra trước chạy cho
mọi lần chạy GPU là dòng init `Native GPU (CUDA/HIP)`,
`liblitetorch_gpu.so` nằm cạnh extension, và
`lt.cuda.device_count()` trả về 2 trên image Kaggle. Bản phác thảo
dưới đây là REQUIRES-GPU: nó là hình dạng của các ví dụ CPU đã kiểm
chứng với thiết bị đổi sang GPU, và nó phải chạy trên Kaggle (thuộc
QA) trước khi bất kỳ khẳng định nào được đưa ra về nó.

<!-- ex: id=E5-7 device=gpu backend=cuda runtime=~? torch-parity=n/a status=unverified -->
```python skip-run
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import multiprocessing as mp
import litetorch as lt


def worker(rank, world_size, port, q):
    lt.distributed.init_process_group(rank, world_size, "127.0.0.1", port)
    try:
        device = lt.Device("cuda:" + str(rank))
        lt.cuda.set_device(rank)
        model = lt.nn.Sequential([lt.nn.Linear(4096, 4096, True)])
        model.to(device)
        lt.distributed.fsdp_fully_shard(model)
        x = lt.Tensor.from_vector([0.5] * 4096, [1, 4096], device)
        y = model.forward(x)
        loss = y.sum()
        loss.backward()
        q.put((rank, y.to_vector()[:4]))
    finally:
        lt.distributed.shutdown()


def main():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, 2, 29645, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(300)


if __name__ == "__main__":
    main()
```

REQUIRES-GPU — không bao giờ xuất xưởng dưới dạng đã kiểm chứng.
Allreduce 2-GPU và lỗi mismatch-throw của NCCL thuộc về track GPU
của Phase 6, không phải trang này.

## E5.8 — Những gì cần nhớ

- `init_process_group` trước tiên, `shutdown()` cuối cùng,
  `destroy_process_group()` không tồn tại.
- Collective là tất cả-hoặc-không-gì: mọi rank gọi cùng các collective
  theo cùng thứ tự, nếu không job sẽ deadlock.
- FSDP: `fsdp_fully_shard(module)` thay đổi tại chỗ, trả về `None`,
  gather/release tự động; giới hạn prefetch mặc định 2; tham số và
  gradient được shard, trạng thái optimizer thì không; các method của
  `FSDPUnit` không gọi được từ Python.
- DDP: đừng huấn luyện với nó cho tới khi khuyết điểm nhân đôi
  gradient được sửa (đã gửi, §E5.4).
- Rendezvous: `stop()` rồi `join()` thread `run()`, rồi mới hủy store.
- TP là `tp_all_reduce` / `tp_all_gather` với backward nhận biết
  autograd; PP là `PipelineParallelModule` + 1F1B; CP không có binding
  Python tại HEAD này.
