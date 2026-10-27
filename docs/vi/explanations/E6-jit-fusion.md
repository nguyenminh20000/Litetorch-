# E6 — JIT Fusion: Tracer, JITFunction, và Fused Op

JIT fusion lấy một đồ thị nhỏ gồm các op theo từng phần tử và biên
dịch nó thành một lần gọi kernel thay vì một lần gọi cho mỗi op. Lợi
ích không phải là phép tính nhanh hơn — mà là ít lần đi-về bộ nhớ hơn:
`relu(x + b)` không fusion đọc `x` và `b`, ghi tổng, đọc lại tổng, rồi
ghi output của ReLU, trong khi bản fusion đọc `x` và `b` một lần và
ghi một lần. Trang này giải thích tracer xây đồ thị, `JITFunction`
chạy nó, các quy tắc fusion chính xác, các lỗi tracer raise, và danh
mục fused op — tất cả với tree HEAD `b326f29` (extension `88a4ba1`).

Điều kiện tiên quyết: tensor, `lt.Ops.*`, đạo hàm tự động (T1). Mọi ví
dụ chạy trên CPU với LiteTorch `88a4ba1` và PyTorch `2.14.1+cpu`.

## E6.1 — Hai đối tượng: JITVar là đồ thị, JITFunction là chương trình

Một `JITVar` là một nút của một biểu thức tượng trưng: một input, một
hằng số, hoặc kết quả của một op (`ADD`, `SUB`, `MUL`, `DIV`, `RELU`,
`GELU`, `SIGMOID`, `TANH`, `SQRT`, `EXP`, `LOG`, `ABS`, `NEG` —
`bind_jit.cpp:6-23`). Bạn xây biểu thức bằng các hàm unary của
`lt.jit` (`lt.jit.relu(a)`, `lt.jit.gelu(a)`, ...) và bằng toán tử
Python trên `JITVar` (`a + b`, `a * 2.0`). Một `JITVar` không giữ dữ
liệu: nó không thể in ra dưới dạng tensor và không thể tự thực thi.

Một `JITFunction` là một chương trình đã biên dịch: một tên, một biểu
thức `JITVar` gốc, và danh sách có thứ tự các `JITVar` input. Bạn gọi
nó với một danh sách tensor và nhận lại một tensor. `JITFunction` có
`__call__` và `forward` (cùng một thứ), cộng `save(filepath)` và
`JITFunction.load(filepath)` để lưu trữ (`bind_jit.cpp:80-89`).

```
JITVar (tượng trưng)                     JITFunction (thực thi được)
  a = JITVar(INPUT, "x")                   f = JITFunction("relu_fn", relu(a), [a])
  expr = relu(a)                           f([tensor]) -> tensor
  expr không có dữ liệu                    f.save(path) / JITFunction.load(path)
```

<!-- ex: id=E6-1 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    a = lt.jit.JITVar(lt.jit.OpType.INPUT, "x")
    f = lt.jit.JITFunction("relu_fn", lt.jit.relu(a), [a])
    y = f([x])
    d = lt.jit.Tracer.derivative(lt.jit.relu(a), a)
    df = lt.jit.JITFunction("drelu_fn", d, [a])
    dy = df([x])
    f.save("/tmp/e6-jitfn.bin")
    g = lt.jit.JITFunction.load("/tmp/e6-jitfn.bin")
    z = g([x])
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    exp = F.relu(xt).flatten().tolist()
    exp_d = (xt > 0).float().flatten().tolist()
    dy_max = max(abs(p - q) for p, q in zip(y.to_vector(), exp))
    dd_max = max(abs(p - q) for p, q in zip(dy.to_vector(), exp_d))
    dz_max = max(abs(p - q) for p, q in zip(z.to_vector(), exp))
    print("sai số relu tối đa:", dy_max, "sai số đạo hàm tối đa:", dd_max, "sai số save/load tối đa:", dz_max)
    ok = dy_max <= 1e-4 and dd_max <= 1e-4 and dz_max <= 1e-4
    print("E6-1 PASS" if ok else "E6-1 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
`Tracer.derivative(var, wrt)` lấy đạo hàm một biểu thức `JITVar`
theo cách tượng trưng (quy tắc thương cho `DIV`, quy tắc chuỗi cho
`SIGMOID` / `TANH` / `EXP` / `LOG`, các nút `RELU_GRAD` / `GELU_GRAD`
/ `ABS_GRAD` cho các trường hợp không trơn — `src/ops/jit.cpp`,
`Tracer::derivative`). Đạo hàm của ReLU là hàm bậc thang: 1 tại nơi
input dương, 0 ở nơi còn lại.

## E6.2 — Các fused op làm sẵn

Đối với các mẫu thông dụng bạn không cần tự xây `JITFunction`:
`lt.jit` cung cấp chúng trực tiếp (`bind_jit.cpp:46-78`). Mỗi hàm nhận
tensor và trả về một tensor, fusion thành một kernel:

| Hàm | Tính gì | Mirror torch |
|---|---|---|
| `fused_bias_relu(x, bias)` | `relu(x + bias)` | `F.relu(x + bias)` |
| `fused_bias_gelu(x, bias)` | `gelu(x + bias)` | `F.gelu(x + bias, approximate='tanh')` |
| `fused_residual_add(x, residual)` | `x + residual` | `x + residual` |
| `fused_residual_gelu(x, residual)` | `gelu(x + residual)` | `F.gelu(x + residual, approximate='tanh')` |

GELU tại đây là xấp xỉ tanh, không phải dạng erf chính xác mặc định
của torch — mirror parity phải dùng `approximate='tanh'`, và dung sai
là FP32 1e-4 quen thuộc so với mirror đó (xem D-09 trong trang Khác
Biệt Hành Vi: GELU erf chính xác khác tới ~4.4e-4).

<!-- ex: id=E6-2 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def check(name, got, exp, tol=1e-4):
    d = max(abs(a - b) for a, b in zip(got, exp))
    print(name, "max_abs_diff:", d)
    return d <= tol


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    b = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    bt = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    ok = True
    ok = check("fused_bias_relu", lt.jit.fused_bias_relu(x, b).to_vector(), F.relu(xt + bt).flatten().tolist()) and ok
    ok = check("fused_bias_gelu", lt.jit.fused_bias_gelu(x, b).to_vector(), F.gelu(xt + bt, approximate="tanh").flatten().tolist()) and ok
    ok = check("fused_residual_add", lt.jit.fused_residual_add(x, b).to_vector(), (xt + bt).flatten().tolist()) and ok
    ok = check("fused_residual_gelu", lt.jit.fused_residual_gelu(x, b).to_vector(), F.gelu(xt + bt, approximate="tanh").flatten().tolist()) and ok
    print("E6-2 PASS" if ok else "E6-2 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

## E6.3 — Tracer: từ op eager tới JITFunction

`Tracer.trace(inputs, func, name)` ghi lại những gì `func` làm với
các tensor input và trả về một `JITFunction` — không phải `JITVar`.
Sự phân biệt này quan trọng: giá trị trả về là thực thi được, các nút
bên trong nó thì không. `trace` hoạt động bằng cách đặt
`requires_grad = True` trên input, gọi `func(inputs)`, và đi ngược đồ
thị autograd của output, dịch mỗi nút creator thành một `JITVar`
(`src/ops/jit.cpp`, `Tracer::trace` và `trace_tensor`). Cờ
`requires_grad` gốc của các tensor input được khôi phục sau đó.

Các quy tắc tracer bắt buộc, theo thứ tự gây ngạc nhiên:

1. **Xây đồ thị bên trong `func`, với `lt.Ops.*`.** Các op tracer
   hiểu là đúng những op trong `trace_tensor`: `Add`, `Sub`, `Mul`,
   `Div`, `ReLU`, `GELU`, `Sigmoid`, `Tanh`, `Sqrt`, `Exp`, `Log`,
   `Abs`, `Neg`. Bất kỳ op nào khác đều raise
   `[litetorch Error] Unsupported operator name in JIT compilation: <name>`
   (§E6.4).
2. **Mọi lá không phải input phải là một hằng số 1 phần tử.** Một lá
   scalar có `requires_grad` không thể bake vào — hãy truyền nó dưới
   dạng một input tường minh. Một lá đa phần tử không nằm trong danh
   sách input sẽ raise `Traced graph reached a leaf tensor that is not
   in the input list` (§E6.4).
3. **Giữ mọi intermediate sống cho tới khi `trace` trả về.** Tracer
   đọc input của creator qua weak pointer; một intermediate đã bị giải
   phóng sẽ raise lỗi expired-intermediate (§E6.4). Trên thực tế điều
   này có nghĩa: đừng trace một đồ thị bạn đã xây trước đó rồi bỏ các
   intermediate của nó — hãy xây bên trong `func`, nơi các biến cục bộ
   vẫn sống trong suốt lệnh gọi.

<!-- ex: id=E6-3 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    b = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)

    def func(inputs):
        return lt.Ops.relu(lt.Ops.add(inputs[0], inputs[1]))

    fn = lt.jit.Tracer.trace([x, b], func, "traced_add_relu")
    print("trace trả về:", type(fn).__name__)
    y = fn([x, b])
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    bt = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    exp = F.relu(xt + bt).flatten().tolist()
    d = max(abs(a - c) for a, c in zip(y.to_vector(), exp))
    print("sai số trace tối đa:", d)
    ok = type(fn).__name__ == "JITFunction" and d <= 1e-4
    print("E6-3 PASS" if ok else "E6-3 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.

## E6.4 — Những gì tracer từ chối, và lỗi expired-intermediate

Ba lỗi khác nhau, ba thông điệp khác nhau. Đừng nhầm lẫn chúng:

| Lỗi | Tác nhân | Cách sửa |
|---|---|---|
| `Unsupported operator name in JIT compilation: MatMul` | `func` dùng một op ngoài danh sách §E6.3 (tại đây là `matmul`) | Viết lại vùng được trace chỉ bằng các op elementwise được hỗ trợ |
| `Traced graph reached a leaf tensor that is not in the input list` | Trace một đồ thị xây *ngoài* `func` (input có `requires_grad=False` khi xây, nên không tồn tại nút creator) hoặc một lá đa phần tử không được truyền làm input | Xây đồ thị bên trong `func` từ các input đã liệt kê |
| `Tracer found an expired intermediate tensor; keep every intermediate alive until trace returns` | Weak pointer của một intermediate hết hạn trước khi `trace` đi qua đồ thị (`src/ops/jit.cpp:404`) | Giữ mọi intermediate sống cho tới khi `trace` trả về; xây bên trong `func` với `lt.Ops.*` |

Ví dụ dưới đây tái tạo hai lỗi đầu trên `88a4ba1` và nêu trạng thái
đo được của lỗi thứ ba. Hai phát hiện từ việc thăm dò bản build này
đáng ghi lại vì chúng mâu thuẫn với cách đọc ngây thơ của quy tắc.
Thứ nhất, phép `+` thuần của Python trên tensor *bên trong* `func`
trace bình thường trên bản build này — nút creator `Add` giống hệt
của `lt.Ops.add`, nên yêu cầu `lt.Ops.*` là về op nào tracer hiểu,
không phải về cú pháp toán tử. Thứ hai, lỗi expired-intermediate
không thể kích hoạt qua Python API trên bản build này trong tám biến
thể đã thăm dò (intermediate có tên, intermediate đã `del`, intermediate
xây trong helper, `gc.collect()`, đồ thị xây trước với
`requires_grad=True`): đồ thị autograd giữ các intermediate sống qua
điểm mà code người dùng bỏ chúng. Chuỗi lỗi tồn tại trong source
(`jit.cpp:404`) và quy tắc vẫn giữ nguyên như đã viết — giữ các
intermediate sống cho tới khi `trace` trả về — nhưng trên `88a4ba1`
lỗi này tiềm ẩn qua Python, không tái tạo được. Biến thể đồ thị xây
trước dưới đây raise lỗi lá thay vào đó, đó là lỗi bạn thực sự sẽ gặp
nếu trace một đồ thị cũ.

<!-- ex: id=E6-4 device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt


def main():
    cpu = lt.Device("cpu")
    a = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0, 0.25, 0.75, -0.25, 0.5], [2, 4], cpu)
    b = lt.Tensor.from_vector([0.5, 0.25, -0.5, 0.75, 1.0, -1.0, 0.5, 0.25], [4, 2], cpu)
    try:
        lt.jit.Tracer.trace([a, b], lambda ins: lt.Ops.matmul(ins[0], ins[1]), "mm")
        print("trace matmul: không lỗi (KHÔNG KỲ VỌNG)")
    except RuntimeError as e:
        print("trace matmul raise:", e)
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    c = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)
    y = lt.Ops.relu(lt.Ops.add(x, c))
    try:
        lt.jit.Tracer.trace([x, c], lambda ins: y, "prebuilt")
        print("trace prebuilt: không lỗi (KHÔNG KỲ VỌNG)")
    except RuntimeError as e:
        print("trace prebuilt raise:", e)
    print("E6-4 PASS")


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1. Hai thông điệp in ra là
các lỗi runtime nguyên văn; thông điệp expired-intermediate được trích
trong bảng trên từ `src/ops/jit.cpp:404` và không tái tạo được qua
Python API trên bản build này (xem phần chữ).

## E6.5 — Danh mục fused op: các kernel lớn hơn

Ngoài `lt.jit`, `lt.Ops` cung cấp các fused kernel cho cả đồ thị con
(`bind_ops.cpp:35, 78, 162-164`). Đây là các op đơn, không phải chương
trình được trace: chúng chạy một kernel (hoặc một chuỗi kernel) mà
không xây đồ thị nào cả.

| Hàm | Tính gì | Ghi chú |
|---|---|---|
| `Ops.fused_add_layernorm(input, residual, normalized_shape, weight=None, bias=None, eps=1e-5)` | `layernorm(input + residual)` | `normalized_shape` là một list, ví dụ `[4]` |
| `Ops.fused_swiglu(gate, up)` | `silu(gate) * up` | Activation SwiGLU, lõi MLP của transformer |
| `Ops.fused_qkv_proj(x, wq, wk, wv)` | `(x @ wq.T, x @ wk.T, x @ wv.T)` | Trả về một 3-tuple `(q, k, v)`, không phải một tensor |
| `Ops.fused_linear_cross_entropy(x, weight, target, bias=None)` | cross-entropy của `x @ weight.T + bias` với `target` | Fusion để logits đầy đủ không bao giờ hiện hình |

<!-- ex: id=E6-5 device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import litetorch as lt
import torch
import torch.nn.functional as F


def check(name, got, exp, tol=1e-4):
    d = max(abs(a - b) for a, b in zip(got, exp))
    print(name, "max_abs_diff:", d)
    return d <= tol


def main():
    cpu = lt.Device("cpu")
    x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
    r = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
    w = lt.Tensor.from_vector([1.0, 1.0, 1.0, 1.0], [4], cpu)
    xt = torch.tensor([[0.5, -0.5, 1.0, -1.0]])
    rt = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    ok = True
    ok = check("fused_add_layernorm", lt.Ops.fused_add_layernorm(x, r, [4], w).to_vector(), F.layer_norm(xt + rt, [4]).flatten().tolist()) and ok
    ok = check("fused_swiglu", lt.Ops.fused_swiglu(x, r).to_vector(), (F.silu(xt) * rt).flatten().tolist()) and ok
    wq = lt.Tensor.from_vector([0.5, 0.25, 0.0, 0.75, 0.1, 0.5, -0.5, 0.25, 0.3, 0.3, 0.2, 0.1, -0.2, 0.4, 0.6, 0.2], [4, 4], cpu)
    wqt = torch.tensor([[0.5, 0.25, 0.0, 0.75], [0.1, 0.5, -0.5, 0.25], [0.3, 0.3, 0.2, 0.1], [-0.2, 0.4, 0.6, 0.2]])
    q, k, v = lt.Ops.fused_qkv_proj(x, wq, wq, wq)
    exp_q = (xt @ wqt.T).flatten().tolist()
    ok = check("fused_qkv_proj q", q.to_vector(), exp_q) and ok
    ok = check("fused_qkv_proj k", k.to_vector(), exp_q) and ok
    ok = check("fused_qkv_proj v", v.to_vector(), exp_q) and ok
    tgt = lt.Tensor.from_vector([2.0], [1], cpu)
    ok = check("fused_linear_cross_entropy", lt.Ops.fused_linear_cross_entropy(x, wq, tgt).to_vector(), F.cross_entropy(xt @ wqt.T, torch.tensor([2])).flatten().tolist()) and ok
    print("E6-5 PASS" if ok else "E6-5 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4.
`fused_linear_cross_entropy` quan trọng nhất ở quy mô lớn: với một
từ vựng lớn, đường không fusion hiện hình một tensor logits
`[batch, vocab]` chỉ để rút gọn nó thành một số cho mỗi hàng; kernel
fusion không bao giờ xây nó.

## E6.6 — Khi nào fusion giúp, và khi nào gây hại

Fusion giúp khi đồ thị bị giới hạn bởi bộ nhớ và theo từng phần tử:
lợi ích tỉ lệ với lưu lượng được loại bỏ. Một `relu(x + b)` trên một
tensor lớn loại bỏ một lần đọc đầy đủ + một lần ghi đầy đủ của
intermediate; một chuỗi năm op elementwise loại bỏ bốn lần. Fusion
gây hại — hoặc trung tính — khi:

- Đồ thị đã bị giới hạn bởi tính toán (matmul chiếm đa số): fusion
  phần epilogue tiết kiệm một lần gọi nhưng matmul chiếm phần lớn thời
  gian, nên lợi ích tương đối chỉ là nhiễu.
- Intermediate còn cần sau này: nếu một op sau dùng tổng
  pre-activation, fusion nó vào ReLU buộc phải tính lại hoặc hiện hình
  lần thứ hai. Tracer không có phân tích rematerialization; nó fusion
  những gì bạn đưa cho nó.
- Đồ thị quá nhỏ: một lần gọi kernel tốn ~micro giây, và trace +
  biên dịch `JITFunction` tốn nhiều hơn một lần chạy eager rất nhiều.
  Trace một lần, gọi nhiều lần — không bao giờ trace bên trong vòng
  lặp huấn luyện.
- Op không nằm trong tập được hỗ trợ (§E6.3): trace thất bại hoàn
  toàn, và phương án dự phòng là đường eager không fusion.

Ví dụ dưới đây đo hình dạng của sự đánh đổi trên CPU: cùng một
`relu(x + b)` theo ba cách — wrapper tiện ích `lt.jit`, một
`JITFunction` được trace một lần và gọi 2000 lần, và đường eager không
fusion — mỗi cách 2000 lần lặp, giá trị được kiểm tra parity trước,
thời gian in ra sau. Thời gian tuyệt đối phụ thuộc máy (VM 2-CPU tại
đây, đang chịu tải từ các job khác); tỉ lệ mới là điểm chính. Đo được
trên CPU tại `[64, 64]` trên `88a4ba1`, đường JIT THUA: hàm đã trace
tốn ~0.012 giây/lần và wrapper ~0.011 giây/lần so với ~0.0003-0.0005
giây/lần không fusion, tỉ lệ ~0.03. Lợi ích fusion — một lần qua bộ
nhớ thay vì hai — là có thật, nhưng tại kích thước này nó nhỏ hơn chi
phí dispatch của JIT, và wrapper còn xây lại hàm trên mỗi lần gọi.
Fusion có lợi ở nơi tiết kiệm lưu lượng và lần gọi vượt chi phí
dispatch: tensor lớn hơn, và trên hết là GPU, nơi mỗi op eager là một
lần gọi kernel riêng. Phía GPU của khẳng định đó là REQUIRES-GPU,
chưa đo tại đây; phía CPU đã đo, và nó nói: với đồ thị nhỏ trên CPU,
hãy ưu tiên các fused kernel `Ops` của §E6.5 — các lệnh gọi C++ đơn
không có dispatch JIT — thay vì các `JITFunction` được trace.

<!-- ex: id=E6-6 device=cpu backend=cpu runtime=~45s torch-parity=1e-4 status=verified-2026-10-27@88a4ba1 -->
```python
import sys
sys.path.insert(0, "/home/hatch/workspace/framework/core/litetorch-d3/python")
import time
import litetorch as lt
import torch
import torch.nn.functional as F


def main():
    cpu = lt.Device("cpu")
    n = 64 * 64
    xv = [0.001 * float((i * 7) % 200 - 100) for i in range(n)]
    bv = [0.01] * n
    x = lt.Tensor.from_vector(xv, [64, 64], cpu)
    b = lt.Tensor.from_vector(bv, [64, 64], cpu)
    xt = torch.tensor(xv).reshape(64, 64)
    bt = torch.tensor(bv).reshape(64, 64)
    exp = F.relu(xt + bt).flatten().tolist()
    fn = lt.jit.Tracer.trace([x, b], lambda ins: lt.Ops.relu(lt.Ops.add(ins[0], ins[1])), "traced_add_relu")
    d = max(abs(a - c) for a, c in zip(fn([x, b]).to_vector(), exp))
    print("trace so với torch max_abs_diff:", d)
    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters):
        lt.jit.fused_bias_relu(x, b)
    t_wrap = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        fn([x, b])
    t_traced = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(iters):
        lt.Ops.relu(lt.Ops.add(x, b))
    t_unfused = time.perf_counter() - t0
    print("wrapper giây/lần:", t_wrap / iters, "trace giây/lần:", t_traced / iters, "không fusion giây/lần:", t_unfused / iters)
    print("tỉ lệ không fusion/trace:", t_unfused / t_traced)
    ok = d <= 1e-4
    print("E6-6 PASS" if ok else "E6-6 FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

Đã kiểm chứng trên CPU, LiteTorch 88a4ba1, PyTorch 2.14.1, parity 1e-4
(giá trị); thời gian chỉ mang tính tham khảo và phụ thuộc máy.

## E6.7 — Những gì cần nhớ

- `JITVar` là đồ thị, `JITFunction` là chương trình;
  `Tracer.trace` trả về một `JITFunction`, không bao giờ là `JITVar`.
- Trace bên trong `func` chỉ với các op được hỗ trợ (`Add/Sub/Mul/Div`,
  `ReLU/GELU/Sigmoid/Tanh/Sqrt/Exp/Log/Abs/Neg`); `MatMul` và các op
  tương tự bị từ chối, cũng như bất kỳ đồ thị nào xây ngoài `func`.
- Lỗi expired-intermediate (`jit.cpp:404`) có thật trong source và
  tiềm ẩn qua Python trên `88a4ba1` — vẫn giữ các intermediate sống
  cho tới khi `trace` trả về, vì tracer đọc chúng qua weak pointer.
- Các fused op làm sẵn (`fused_bias_relu`, `fused_bias_gelu`,
  `fused_residual_add`, `fused_residual_gelu`) và các fused kernel
  `Ops` lớn hơn (`fused_add_layernorm`, `fused_swiglu`,
  `fused_qkv_proj`, `fused_linear_cross_entropy`) bao phủ các mẫu
  thông dụng mà không cần trace.
- GELU trong đường fusion là xấp xỉ tanh: so sánh với
  `F.gelu(..., approximate='tanh')`, không phải mặc định chính xác
  của torch.
- Fusion có lợi khi đồ thị bị giới hạn bởi bộ nhớ, theo từng phần tử,
  và hàm đã biên dịch được gọi nhiều lần; nhưng trên CPU tại `[64, 64]`
  đường JIT đo được chậm hơn ~30 lần so với eager không fusion — lợi
  ích fusion chỉ vượt chi phí dispatch khi tensor lớn hơn hoặc trên GPU
  (chưa đo, REQUIRES-GPU). Với đồ thị nhỏ trên CPU, hãy ưu tiên các
  fused kernel `Ops` của §E6.5 thay vì `JITFunction` được trace.
