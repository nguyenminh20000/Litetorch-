# Lượng tử hóa, JIT, ONNX, Hugging Face, Suy luận, Pruning, Profiling, Hub — Tài liệu tham khảo API

Trang này bao phủ bề mặt tối ưu mô hình và trao đổi mô hình: `litetorch.quantization`, `litetorch.jit`, `litetorch.onnx`, `litetorch.hf`, `litetorch.inference`, `litetorch.prune`, `litetorch.profiler`, `litetorch.hub`.

- **Nguồn chân lý:** `src/bindings/bind_quant.cpp`, `bind_jit.cpp`, `bind_onnx.cpp`, `bind_hf.cpp`, `bind_inference.cpp`, `bind_prune.cpp` (binding Python); `include/litetorch/quantization.h`, `jit.h`, `jit_detail.h`, `onnx.h`, `hf.h`, `inference.h`, `prune.h`, `profiler.h`, `hub.h`.
- **Cây mã nguồn:** agent-v1@02cc271a / HEAD cục bộ 89afd67 (baseline tài liệu là extension 88a4ba1; B5 xác minh lại tại head).
- **Quy ước:** `cpu = lt.Device("cpu")` trong mọi ví dụ. Mọi ví dụ đều tự chứa và chạy được trên CPU trừ khi được đánh dấu `REQUIRES-GPU` / `REQUIRES-TPU` / `REQUIRES-NETWORK`. Ngang bằng torch là FP32 1e-4 trừ khi tiêu đề ghi khác.
- **Sai khác liên quan:** D-01 (`Tensor.data_ptr()` throw trên tensor không phải FP32 — xem `cuda-tpu-autograd.md`); D-09 (GELU dùng xấp xỉ tanh — xem `plan/differences.md`).

## Lượng tử hóa

### `quantization.quantize_linear`

- **Chữ ký:** `quantize_linear(linear, bits=8) -> Tensor` (w8a8 — đầu vào FP32 được lượng tử hóa theo từng tensor; xem trường hợp biên)
- **Nguồn:** `src/bindings/bind_quant.cpp:5`
- **Tham số:** `linear` — một `nn.Linear`. `bits` — int, mặc định 8.
- **Trả về:** tensor trọng số đã lượng tử hóa (đóng gói INT8). Tensor trả về là trọng số, không phải module.
- **Trường hợp biên:** đường w8a8 nhận đầu vào FP32 và lượng tử hóa chúng theo từng tensor tại lúc gọi — truyền tensor FP32, không phải INT8 đã lượng tử hóa trước (xem ghi chú how-to H8.3). `bits` khác 8 vẫn được chấp nhận nhưng kernel được tối ưu cho 8.
- **PyTorch:** `torch.ao.quantization.quantize_dynamic` — cùng vai trò (lượng tử hóa động theo từng tensor); API của torch lượng tử hóa toàn bộ module, LiteTorch lượng tử hóa từng `Linear` một.

### `quantization.quantize_dynamic_linear` / `quantize_dynamic_lstm` / `quantize_dynamic_gru`

- **Chữ ký:** `quantize_dynamic_linear(linear) -> nn.Module`; `quantize_dynamic_lstm(lstm) -> nn.Module`; `quantize_dynamic_gru(gru) -> nn.Module`
- **Nguồn:** `src/bindings/bind_quant.cpp:40-49`
- **Tham số:** một module `Linear` / `LSTM` / `GRU`.
- **Trả về:** một module mới mà forward của nó lượng tử hóa activation một cách động (trọng số INT8, activation FP32). Kiểu trả về là `litetorch.nn.Module` (một module được wrap, không phải class gốc).
- **Trường hợp biên:** module gốc không bị thay đổi. `forward` của module đã lượng tử hóa có cùng hợp đồng shape như bản gốc.
- **PyTorch:** `torch.ao.nn.quantized.dynamic.Linear` / `LSTM` / `GRU` — cùng vai trò.

### `quantization.Calibrator`

- **Chữ ký:** class `Calibrator()`; các phương thức `collect(name, tensor)` · `get_scale(name) -> float` · `get_asymmetric_params(name) -> (float, float)`
- **Nguồn:** `src/bindings/bind_quant.cpp:7-11`
- **Tham số:** `collect(name, tensor)` — ghi một mẫu hiệu chuẩn dưới `name`. `get_scale(name)` — scale đối xứng cho `name`. `get_asymmetric_params(name)` — `(scale, zero_point)` cho lượng tử hóa bất đối xứng.
- **Trả về:** `get_scale` trả về float; `get_asymmetric_params` trả về tuple `(scale, zero_point)`.
- **Trường hợp biên:** gọi `get_scale` trước bất kỳ `collect` nào cho tên đó sẽ throw. Scale là theo từng tensor (một scale cho mỗi tên), không phải theo từng kênh.
- **PyTorch:** `torch.ao.quantization.observer` — cùng vai trò (thu thập thống kê activation, suy ra scale).

<!-- ex: id=r13-calibrator device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
cal = lt.quantization.Calibrator()
w = lt.Tensor.from_vector([0.01, -0.02, 0.03, -0.04], [4], cpu)
cal.collect("w", w)
print("scale:", cal.get_scale("w"))
print("asymmetric:", cal.get_asymmetric_params("w"))
```

### `quantization.PTQConfig`

- **Chữ ký:** class `PTQConfig()` với các trường đọc/ghi `bits` (int), `per_channel` (bool), `symmetric` (bool)
- **Nguồn:** `src/bindings/bind_quant.cpp:28-32`
- **Trả về:** một đối tượng config thuần cho lượng tử hóa sau huấn luyện (PTQ). Mặc định: `bits=8`, `per_channel=False`, `symmetric=True` (xác minh mặc định bằng cách in chúng — chúng là các trường thuần, không phải đối số constructor).
- **PyTorch:** `torch.ao.quantization.QConfig` — cùng vai trò; config của torch mang các observer class, LiteTorch mang ba flag.

### `quantization.PreparedModel` / `quantization.prepare` / `quantization.convert`

- **Chữ ký:** `prepare(model, config) -> PreparedModel`; `convert(prepared) -> nn.Module`; `PreparedModel.calibrate(data_fn, num_batches=10) -> None`; `PreparedModel.model` (đọc/ghi)
- **Nguồn:** `src/bindings/bind_quant.cpp:34-39`
- **Tham số:** `prepare(model, config)` — một module và một `PTQConfig`. `calibrate(data_fn, num_batches=10)` — `data_fn` là callable không đối số trả về một tensor hiệu chuẩn mỗi lần gọi; `num_batches` mặc định 10.
- **Trả về:** `prepare` trả về một `PreparedModel` wrap module (đã gắn hook hiệu chuẩn); `convert` trả về module đã lượng tử hóa (một `nn.Linear` cho đầu vào `Linear`).
- **Trường hợp biên:** `convert` trước `calibrate` dùng scale mặc định (1.0) — hãy hiệu chuẩn trước để có scale thật. `PreparedModel.model` là module bên dưới; chỉ thay đổi nó qua `calibrate`/`convert`.
- **PyTorch:** `torch.ao.quantization.prepare` / `convert` — cùng luồng PTQ hai bước.

<!-- ex: id=r13-ptq device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
lin = lt.nn.Linear(4, 4, True)
lin.to(cpu)
cfg = lt.quantization.PTQConfig()
cfg.bits = 8
pm = lt.quantization.prepare(lin, cfg)
pm.calibrate(lambda: lt.Tensor.from_vector([0.1] * 4, [1, 4], cpu))
q = lt.quantization.convert(pm)
x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
print("quantized forward shape:", q.forward(x).shape)
```

### `quantization.QATLinear`

- **Chữ ký:** `QATLinear(linear, weight_scale, weight_zero_point, act_scale, act_zero_point, bits)`
- **Nguồn:** `src/bindings/bind_quant.cpp:13-26`
- **Tham số:** `linear` — một `nn.Linear`. `weight_scale` / `weight_zero_point` — tham số fake-quant cho trọng số. `act_scale` / `act_zero_point` — tham số fake-quant cho activation. `bits` — int.
- **Thuộc tính (đọc/ghi):** `weight`, `bias`, `weight_scale`, `weight_zero_point`, `act_scale`, `act_zero_point`, `bits`, `qat_enabled`.
- **Phương thức:** `forward(input)` · `parameters()` · `to(device)`.
- **Trả về:** một module chạy fake quantization trong forward (lượng tử hóa rồi giải lượng tử hóa) để huấn luyện thấy được sai số lượng tử hóa. Đặt `qat_enabled = False` để chạy nó như một linear thường.
- **Trường hợp biên:** scale bằng 0.0 chỉ làm fake-quant thành no-op khi `qat_enabled` là False; với `qat_enabled` True, scale bằng 0 chia cho 0 — hãy đặt scale thật từ một `Calibrator` trước.
- **PyTorch:** `torch.ao.nn.qat.Linear` — cùng vai trò (huấn luyện fake-quant).

<!-- ex: id=r13-qat device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
lin = lt.nn.Linear(4, 4, True)
lin.to(cpu)
qat = lt.quantization.QATLinear(lin, 0.01, 0.0, 0.01, 0.0, 8)
x = lt.Tensor.from_vector([0.1, 0.2, 0.3, 0.4], [1, 4], cpu)
print("qat forward shape:", qat.forward(x).shape)
print("qat bits:", qat.bits, "enabled:", qat.qat_enabled)
```

## JIT

### `jit.OpType`

- **Chữ ký:** enum với `INPUT`, `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `RELU`, `GELU`, `SIGMOID`, `TANH`, `SQRT`, `EXP`, `LOG`, `ABS`, `CONST`
- **Nguồn:** `src/bindings/bind_jit.cpp:6-23`
- **Trả về:** loại phép toán của một node `JITVar`.
- **PyTorch:** `torch.jit` không có enum op cho người dùng thấy; đây là từ vựng IR riêng của LiteTorch.

### `jit.JITVar`

- **Chữ ký:** `JITVar(op, name)`; `JITVar(val)`; `op`, `name`, `val` đọc/ghi
- **Nguồn:** `src/bindings/bind_jit.cpp:24-41`
- **Tham số:** `JITVar(op, name)` — một node ký hiệu loại `op` tên `name`. `JITVar(val)` — một node hằng giữ float `val`.
- **Toán tử:** `+` / `-` / `*` / `/` (var-var và var-float, cả hai thứ tự) và `-` một ngôi. Mỗi phép toán tạo một node `JITVar` mới.
- **Trả về:** một node biểu thức ký hiệu. Nó không giữ dữ liệu — nó là một đồ thị, không phải tensor.
- **PyTorch:** không có tương đương trực tiếp; gần nhất là symbolic tracing của `torch.fx`, ghi lại các phép toán trên đối tượng `Proxy`.

### `jit.relu` / `jit.gelu` / `jit.sigmoid` / `jit.tanh` / `jit.sqrt` / `jit.exp` / `jit.log` / `jit.abs`

- **Chữ ký:** `relu(a) -> JITVar` (và tương tự cho bảy hàm còn lại)
- **Nguồn:** `src/bindings/bind_jit.cpp:43-50`
- **Tham số:** `a` — một `JITVar`.
- **Trả về:** một node `JITVar` mới áp dụng phép toán một ngôi. GELU ở đây là xấp xỉ tanh (D-09).
- **PyTorch:** `torch.ops.aten.relu` v.v. — cùng phép toán, áp dụng trên tensor thay vì node ký hiệu.

### `jit.fused_bias_relu` / `jit.fused_bias_gelu` / `jit.fused_residual_add` / `jit.fused_residual_gelu`

- **Chữ ký:** `fused_bias_relu(x, bias) -> Tensor`; `fused_bias_gelu(x, bias) -> Tensor`; `fused_residual_add(x, residual) -> Tensor`; `fused_residual_gelu(x, residual) -> Tensor`
- **Nguồn:** `src/bindings/bind_jit.cpp:51-79`
- **Tham số:** `x` — tensor đầu vào; `bias` / `residual` — toán hạng thứ hai.
- **Trả về:** kết quả đã fusion: `relu(x + bias)`, `gelu(x + bias)`, `x + residual`, `gelu(x + residual)`.
- **Trường hợp biên:** `fused_bias_relu` / `fused_bias_gelu` yêu cầu `x` và `bias` có **shape giống hệt nhau** — bias shape `[4]` với `x` shape `[1, 4]` raise `RuntimeError: JITFunction requires all inputs to have identical shapes`. Broadcast bias về shape của `x` trước (ví dụ `bias.reshape([1, 4])`). Đây là một giới hạn thật, không phải lỗi tài liệu.
- **PyTorch:** không có tương đương đơn lẻ; torch fusion các phép này bên trong `torch.compile` / Inductor, không bao giờ dưới dạng hàm có tên.

<!-- ex: id=r13-fused device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
b = lt.Tensor.from_vector([0.1, 0.1, 0.1, 0.1], [1, 4], cpu)
print("fused_bias_relu:", lt.jit.fused_bias_relu(x, b).to_vector())
print("fused_residual_gelu:", lt.jit.fused_residual_gelu(x, b).to_vector())
```

### `jit.JITFunction`

- **Chữ ký:** `JITFunction(name, expr, inputs)`; `__call__(args) -> Tensor`; `forward(args) -> Tensor`; `save(filepath) -> None`; `load(filepath) -> JITFunction` (static)
- **Nguồn:** `src/bindings/bind_jit.cpp:80-86`
- **Tham số:** `name` — str, tên hàm. `expr` — `JITVar` đầu ra. `inputs` — danh sách `JITVar` đầu vào, theo thứ tự gọi.
- **Trả về:** gọi hàm với một danh sách tensor (cùng thứ tự như `inputs`) trả về tensor đã tính. `save` ghi hàm đã biên dịch ra đĩa; `load` đọc lại.
- **Trường hợp biên:** số đối số khi gọi phải khớp `len(inputs)`. Mọi đầu vào phải có shape giống hệt nhau (cùng quy tắc như `fused_bias_relu`).
- **PyTorch:** `torch.jit.ScriptFunction` — cùng vai trò (một hàm đã biên dịch, có thể serialize); `torch.jit.script` của torch biên dịch mã nguồn Python, `JITFunction` của LiteTorch biên dịch một đồ thị `JITVar`.

<!-- ex: id=r13-jitfn device=cpu backend=cpu runtime=~1s torch-parity=1e-4 status=verified-2026-10-26@88a4ba1 -->
```python
import tempfile
import litetorch as lt

cpu = lt.Device("cpu")
a = lt.jit.JITVar(lt.jit.OpType.INPUT, "x")
y = lt.jit.relu(a)
f = lt.jit.JITFunction("relu_fn", y, [a])
x = lt.Tensor.from_vector([0.5, -0.5, 1.0, -1.0], [1, 4], cpu)
print("jit relu:", f([x]).to_vector())
with tempfile.TemporaryDirectory() as d:
    f.save(d + "/relu_fn.bin")
    f2 = lt.jit.JITFunction.load(d + "/relu_fn.bin")
    print("loaded jit relu:", f2([x]).to_vector())
```

### `jit.Tracer`

- **Chữ ký:** các phương thức static `trace(inputs, func) -> JITFunction`; `derivative(var, wrt) -> JITVar`; `serialize_var(var) -> str`; `deserialize_var(str) -> JITVar`
- **Nguồn:** `src/bindings/bind_jit.cpp:88-92`
- **Tham số:** `trace(inputs, func)` — `inputs` là một dãy tensor; `func` là callable nhận một danh sách tensor và trả về một tensor. `derivative(var, wrt)` — đạo hàm ký hiệu của `var` theo `wrt`. `serialize_var` / `deserialize_var` — chuyển đổi văn bản hai chiều của một `JITVar`.
- **Trả về:** `trace` trả về một `JITFunction` wrap đồ thị đã trace — gọi nó với một danh sách tensor để tính nó; `derivative` trả về `JITVar` đạo hàm.
- **Trường hợp biên (đã xác minh):** `func` phải dùng `lt.Ops.*` trên các tensor được truyền — `trace([x], lambda ts: lt.Ops.relu(ts[0]))` chạy được. Truyền một `JITVar` cho `trace` raise `TypeError` (nó nhận tensor, không nhận var). Xây một intermediate bằng `+` thường bên trong `func` rồi trả về nó raise `RuntimeError: Tracer found an expired intermediate tensor; keep every intermediate alive until trace returns` — ngay cả giữ intermediate trong dict cũng không giúp; hãy dùng các lệnh gọi `lt.Ops.*` mà kết quả chảy trực tiếp vào giá trị trả về.
- **PyTorch:** `torch.jit.trace` — cùng ý tưởng (ghi lại đồ thị bằng cách chạy hàm); tracer của torch chấp nhận Python tùy ý, của LiteTorch chỉ chấp nhận các lệnh gọi `lt.Ops.*` trên đầu vào.

<!-- ex: id=r13-tracer device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.5, -0.5], [2], cpu)
fn = lt.jit.Tracer.trace([x], lambda ts: lt.Ops.relu(ts[0]))
print("traced:", fn([x]).to_vector())
v = lt.jit.JITVar(lt.jit.OpType.INPUT, "v")
print("d/dv v:", lt.jit.Tracer.serialize_var(lt.jit.Tracer.derivative(v, v)))
```

## ONNX

### `onnx.ValueInfo` / `onnx.Node` / `onnx.Graph`

- **Chữ ký:** class với các trường đọc/ghi. `ValueInfo`: `name`, `shape`, `dtype`, `dim_params`. `Node`: `op_type`, `name`, `inputs`, `outputs`, `attrs`, `int_attrs`, `float_attrs`, `ints_attrs`, `domain`. `Graph`: `name`; các phương thức `add_input(name, shape, dtype)` · `add_output(name, shape, dtype)` · `add_node(op_type, inputs, outputs)` · `add_dynamic_input(name, dim_params, dtype)` · `add_dynamic_output(name, dim_params, dtype)` · `add_node_typed(op_type, inputs, outputs, attrs)` · `add_initializer(name, tensor)` · `num_nodes()` · `node_op_types()`.
- **Nguồn:** `src/bindings/bind_onnx.cpp:15-63`
- **Trả về:** ba khối xây dựng của một đồ thị ONNX. `Graph.num_nodes()` trả về số node; `node_op_types()` trả về danh sách loại op theo thứ tự.
- **Trường hợp biên:** `add_node_typed` nhận một dict attrs; `add_node` thuần không nhận attrs. `dim_params` giữ tên chiều ký hiệu (ví dụ `"batch"`) cho shape động.
- **PyTorch:** `torch.onnx` xây cùng ba cấu trúc này nội bộ; LiteTorch expose chúng trực tiếp.

### `onnx.export_onnx_text` / `onnx.export_onnx_binary` / `onnx.export_onnx`

- **Chữ ký:** `export_onnx_text(graph) -> str`; `export_onnx_binary(graph) -> bytes`; `export_onnx(graph, path) -> None`
- **Nguồn:** `src/bindings/bind_onnx.cpp:64-70`
- **Tham số:** `graph` — một `onnx.Graph`. `path` — đường dẫn file đầu ra (cho `export_onnx`).
- **Trả về:** dạng văn bản bắt đầu `ir_version: 8 | producer_name: "litetorch" | graph { name: "litetorch" ...`; dạng binary là mã protobuf; `export_onnx` ghi dạng binary ra `path`.
- **PyTorch:** `torch.onnx.export` — cùng vai trò; torch ghi một file mô hình đầy đủ, LiteTorch ghi đồ thị bạn đã xây.

### `onnx.trace_module` / `onnx.trace_module_dynamic`

- **Chữ ký:** `trace_module(module, example_input) -> Graph`; `trace_module_dynamic(module, example_input, dynamic_axes) -> Graph`
- **Nguồn:** `src/bindings/bind_onnx.cpp:72-74`
- **Tham số:** `module` — một `nn.Module`. `example_input` — một tensor để trace. `dynamic_axes` — dict ánh xạ tên đầu vào tới tên chiều động.
- **Trả về:** `Graph` đã trace.
- **PyTorch:** `torch.onnx.export` với `dynamo=True` — cùng vai trò (trace một module thành đồ thị).

### `onnx.export` / `onnx.export_dynamic`

- **Chữ ký:** `export(module, example_input, path) -> None`; `export_dynamic(module, example_input, dynamic_axes, path) -> None`
- **Nguồn:** `src/bindings/bind_onnx.cpp:76-78`
- **Tham số:** `module`, `example_input`, `dynamic_axes` như trên; `path` — file đầu ra.
- **Trả về:** `None`. Ghi mô hình đã trace ra `path` trong một lệnh gọi.
- **PyTorch:** `torch.onnx.export(model, args, path)` — cùng hình thức một lệnh gọi.

<!-- ex: id=r13-onnx device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import tempfile
import litetorch as lt

cpu = lt.Device("cpu")
lin = lt.nn.Linear(4, 4, True)
lin.to(cpu)
x = lt.Tensor.from_vector([0.1] * 4, [1, 4], cpu)
g = lt.onnx.trace_module(lin, x)
print("traced nodes:", g.num_nodes(), g.node_op_types())
print(lt.onnx.export_onnx_text(g)[:120])
with tempfile.TemporaryDirectory() as d:
    lt.onnx.export(lin, x, d + "/linear.onnx")
    print("exported ok")
```

## Hugging Face

### `hf.ModelConfig`

- **Chữ ký:** class `ModelConfig()` với các trường đọc/ghi `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `hidden_act`, `rms_norm_eps`, `layer_norm_eps`, `n_layer`, `n_head`, `n_embd`, `n_positions`
- **Nguồn:** `src/bindings/bind_hf.cpp:7-24`
- **Trả về:** một đối tượng config thuần. Các trường kiểu GPT (`n_layer`, `n_head`, `n_embd`, `n_positions`) và các trường kiểu Llama (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `rms_norm_eps`) nằm cạnh nhau — đặt những trường mà loại mô hình của bạn dùng.
- **PyTorch:** `transformers.PretrainedConfig` — cùng vai trò; của LiteTorch là một struct phẳng, không phải hệ phân cấp class.

### `hf.PretrainedBundle`

- **Chữ ký:** class `PretrainedBundle()` với `config` (một `ModelConfig`) và `state_dict` (dict tên tới tensor) đọc/ghi
- **Nguồn:** `src/bindings/bind_hf.cpp:26-29`
- **Trả về:** kết quả của `from_pretrained`: config + trọng số cùng nhau.
- **PyTorch:** giá trị trả về của `transformers.AutoModel.from_pretrained` — cùng cặp đôi.

### `hf.load_config`

- **Chữ ký:** `load_config(repo_or_dir, revision="main") -> ModelConfig`
- **Nguồn:** `src/bindings/bind_hf.cpp:31`
- **Tham số:** `repo_or_dir` — một repo id Hugging Face (ví dụ `"gpt2"`) hoặc một thư mục cục bộ chứa `config.json`. `revision` — mặc định `"main"`.
- **Trả về:** `ModelConfig` đã parse.
- **Trường hợp biên:** repo id cần mạng; thư mục cục bộ cần một `config.json` trong đó.
- **PyTorch:** `transformers.AutoConfig.from_pretrained` — cùng vai trò.

### `hf.from_pretrained`

- **Chữ ký:** `from_pretrained(repo_id, revision="main", cache_dir=None) -> PretrainedBundle`
- **Nguồn:** `src/bindings/bind_hf.cpp:33`
- **Tham số:** `repo_id` — một repo id Hugging Face. `revision` — mặc định `"main"`. `cache_dir` — thư mục cache cục bộ tùy chọn.
- **Trả về:** một `PretrainedBundle` (config + state dict).
- **Trường hợp biên:** tải trọng số từ Hub — cần mạng và có thể mất vài phút đối với mô hình lớn. Ví dụ này là một fragment, không phải xác minh chạy được (không có mạng trong harness tài liệu).
- **PyTorch:** `transformers.AutoModel.from_pretrained` — cùng vai trò.

<!-- ex: id=r13-hf-pretrained device=cpu backend=cpu runtime=n/a torch-parity=n/a status=fragment -->
```python skip-run
import litetorch as lt

bundle = lt.hf.from_pretrained("gpt2")
print("model type:", bundle.config.model_type)
print("weights:", len(bundle.state_dict))
```

### `hf.weight_name_map`

- **Chữ ký:** `weight_name_map(model_type) -> dict[str, str]`
- **Nguồn:** `src/bindings/bind_hf.cpp:35`
- **Tham số:** `model_type` — ví dụ `"gpt2"`, `"llama"`, `"bert"`.
- **Trả về:** một dict ánh xạ tên trọng số Hugging Face sang tên trọng số LiteTorch. Đối với `"gpt2"`: `transformer.wte.weight` → `embed.weight`, `transformer.ln_f.bias` → `final_norm.bias`, `lm_head.weight` → `lm_head.weight`.
- **Trường hợp biên:** một `model_type` không biết trả về dict rỗng (không throw).
- **PyTorch:** không có tương đương trực tiếp; đây là bảng đổi tên phía LiteTorch.

### `hf.remap_state_dict`

- **Chữ ký:** `remap_state_dict(state_dict, model_type) -> dict[str, Tensor]`
- **Nguồn:** `src/bindings/bind_hf.cpp:36`
- **Tham số:** `state_dict` — dict tên tới `Tensor` (giá trị phải là tensor — dict int raise `TypeError`). `model_type` — ví dụ `"gpt2"`.
- **Trả về:** một dict mới với key đã đổi tên qua `weight_name_map`. Tên không có mapping được giữ nguyên.
- **PyTorch:** không có tương đương trực tiếp; các mô hình `transformers` đổi tên nội bộ khi load.

<!-- ex: id=r13-hf-remap device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.1] * 4, [1, 4], cpu)
sd = {"transformer.wte.weight": x, "custom.weight": x}
print("gpt2 map:", lt.hf.weight_name_map("gpt2")["transformer.wte.weight"])
print("remapped keys:", list(lt.hf.remap_state_dict(sd, "gpt2").keys()))
```

## Suy luận

### `inference.CudaGraph`

- **Chữ ký:** class `CudaGraph()`; các phương thức `capture(fn)` · `replay()` · `captured() -> bool`
- **Nguồn:** `src/bindings/bind_inference.cpp:7-13`
- **Tham số:** `capture(fn)` — một callable không đối số mà công việc CUDA của nó được ghi vào đồ thị.
- **Trả về:** `replay()` chạy lại đồ thị đã ghi; `captured()` báo cáo có đồ thị đang giữ không.
- **Trường hợp biên:** capture cần một thiết bị CUDA — trên máy chỉ có CPU nó throw. Đây là mục `REQUIRES-GPU`; ví dụ dưới đây là một fragment.
- **PyTorch:** `torch.cuda.CUDAGraph` — cùng vai trò (`capture_begin` / `replay`).

### `inference.capture_cuda_graph`

- **Chữ ký:** `capture_cuda_graph(fn) -> CudaGraph`
- **Nguồn:** `src/bindings/bind_inference.cpp:15-17`
- **Tham số:** `fn` — một callable không đối số.
- **Trả về:** một `CudaGraph` đã capture.
- **Trường hợp biên:** `REQUIRES-GPU`.
- **PyTorch:** `torch.cuda.CUDAGraph` + `g.capture_begin()` — cùng vai trò.

<!-- ex: id=r13-cudagraph device=gpu backend=cuda runtime=n/a torch-parity=n/a status=fragment -->
```python skip-run
import litetorch as lt

g = lt.inference.capture_cuda_graph(lambda: None)
print("captured:", g.captured())
g.replay()
```

### `inference.FusionReport`

- **Chữ ký:** class với các trường chỉ đọc `conv_bn_folded`, `layernorm_detected`, `linear_bias_kernel_fused`, `notes`
- **Nguồn:** `src/bindings/bind_inference.cpp:18-22`
- **Trả về:** kết quả của một lệnh gọi phát hiện fusion: số lượng/flag cộng ghi chú đọc được `notes`.
- **PyTorch:** không có tương đương trực tiếp; đây là báo cáo fusion riêng của LiteTorch.

### `inference.fold_conv_bn` / `inference.fuse_conv_bn`

- **Chữ ký:** `fold_conv_bn(conv, bn) -> nn.Module`; `fuse_conv_bn(seq) -> FusionReport`
- **Nguồn:** `src/bindings/bind_inference.cpp:24-26`
- **Tham số:** `fold_conv_bn(conv, bn)` — một `Conv2d` và một `BatchNorm2d`. `fuse_conv_bn(seq)` — một `Sequential` (hoặc danh sách module) để quét tìm cặp Conv+BN.
- **Trả về:** `fold_conv_bn` trả về một module mới với BN đã fold vào trọng số convolution; `fuse_conv_bn` trả về một `FusionReport`.
- **Trường hợp biên:** `fold_conv_bn` yêu cầu BatchNorm ở **chế độ eval** — fold thống kê batch ở chế độ huấn luyện là sai về mặt toán học, nên nó raise `RuntimeError: fold_conv_bn: BatchNorm2d must be in eval mode ... (call eval() first)`. Gọi `bn.eval()` trước khi fold.
- **PyTorch:** `torch.ao.quantization.fuse_modules` — cùng vai trò (fold BN vào conv trước nó).

### `inference.detect_layernorm_candidates` / `inference.detect_layernorm_fusion`

- **Chữ ký:** `detect_layernorm_candidates(model) -> FusionReport`; `detect_layernorm_fusion(model) -> FusionReport`
- **Nguồn:** `src/bindings/bind_inference.cpp:27-30`
- **Tham số:** `model` — một `nn.Module` cần quét.
- **Trả về:** một `FusionReport` với `layernorm_detected` được đặt và `notes` liệt kê các vị trí có thể fusion.
- **PyTorch:** không có tương đương trực tiếp; Inductor thực hiện phát hiện này nội bộ.

### `inference.verify_linear_bias_fusion`

- **Chữ ký:** `verify_linear_bias_fusion() -> bool`
- **Nguồn:** `src/bindings/bind_inference.cpp:31`
- **Trả về:** `True` khi kernel linear+bias đã fusion khớp kết quả chưa fusion trong dung sai trên build này (đã xác minh: trả về `True` trên extension CPU 88a4ba1).
- **PyTorch:** không có tương đương trực tiếp.

<!-- ex: id=r13-inference device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
conv = lt.nn.Conv2d(3, 3, 3, 1, 1)
bn = lt.nn.BatchNorm2d(3)
bn.eval()
folded = lt.inference.fold_conv_bn(conv, bn)
print("folded:", type(folded).__name__)
seq = lt.nn.Sequential([lt.nn.Conv2d(3, 3, 3, 1, 1), lt.nn.BatchNorm2d(3)])
seq.eval()
print("fuse report:", lt.inference.fuse_conv_bn(seq).conv_bn_folded)
print("layernorm:", lt.inference.detect_layernorm_candidates(seq).layernorm_detected)
print("linear+bias fusion ok:", lt.inference.verify_linear_bias_fusion())
```

## Pruning

### `prune.l1_unstructured`

- **Chữ ký:** `l1_unstructured(module, name, amount) -> None`; `l1_unstructured(module, name, amount, dim) -> None`
- **Nguồn:** `src/bindings/bind_prune.cpp:6-15`
- **Tham số:** `module` — một `nn.Module`. `name` — tên tham số, ví dụ `"weight"`. `amount` — tỉ lệ (float trong (0, 1)) hoặc số lượng (int) phần tử cần prune. `dim` — int tùy chọn, chiều cần prune dọc theo.
- **Trả về:** `None`. Đặt 0 cho các phần tử có độ lớn nhỏ nhất (theo L1) và đăng ký một mask pruning trên module.
- **Trường hợp biên:** `amount` dạng float phải nằm trong (0, 1); dạng int phải nhỏ hơn kích thước tham số. Prune hai lần sẽ kết hợp các mask.
- **PyTorch:** `torch.nn.utils.prune.l1_unstructured` — cùng chữ ký và ngữ nghĩa.

### `prune.random_unstructured`

- **Chữ ký:** `random_unstructured(module, name, amount) -> None`; `random_unstructured(module, name, amount, dim) -> None`
- **Nguồn:** `src/bindings/bind_prune.cpp:16-23`
- **Tham số:** giống `l1_unstructured`.
- **Trả về:** `None`. Đặt 0 cho một tập con phần tử ngẫu nhiên đều.
- **Trường hợp biên:** lựa chọn ngẫu nhiên không được seed qua `lt.manual_seed` — đừng dựa vào tính tái lập giữa các lần chạy.
- **PyTorch:** `torch.nn.utils.prune.random_unstructured` — cùng chữ ký và ngữ nghĩa.

### `prune.remove`

- **Chữ ký:** `remove(module, name) -> None`
- **Nguồn:** `src/bindings/bind_prune.cpp:24-28`
- **Tham số:** `module`, `name` — tham số đã prune.
- **Trả về:** `None`. Làm việc pruning thành vĩnh viễn (nướng mask vào tham số) và gỡ mask.
- **PyTorch:** `torch.nn.utils.prune.remove` — cùng vai trò.

### `prune.is_pruned`

- **Chữ ký:** `is_pruned(module, name) -> bool`
- **Nguồn:** `src/bindings/bind_prune.cpp:29-32`
- **Tham số:** `module`, `name`.
- **Trả về:** `True` khi một mask pruning đã đăng ký trên tham số đó.
- **PyTorch:** `torch.nn.utils.prune.is_pruned` — cùng vai trò.

<!-- ex: id=r13-prune device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
lin = lt.nn.Linear(4, 4, True)
lin.to(cpu)
print("pruned before:", lt.prune.is_pruned(lin, "weight"))
lt.prune.l1_unstructured(lin, "weight", 0.5)
print("pruned after:", lt.prune.is_pruned(lin, "weight"))
lt.prune.remove(lin, "weight")
print("pruned after remove:", lt.prune.is_pruned(lin, "weight"))
```

## Profiler

### `profiler.enable` / `profiler.disable` / `profiler.is_enabled`

- **Chữ ký:** `enable() -> None`; `disable() -> None`; `is_enabled() -> bool`
- **Nguồn:** `src/bindings/bind_onnx.cpp:82-84`
- **Trả về:** profiler CPU là một công tắc toàn cục; `is_enabled` báo cáo trạng thái của nó.
- **PyTorch:** context manager `torch.profiler.profile` — cùng vai trò, có phạm vi thay vì toàn cục.

### `profiler.reset` / `profiler.report`

- **Chữ ký:** `reset() -> None`; `report() -> str`
- **Nguồn:** `src/bindings/bind_onnx.cpp:85-86`
- **Trả về:** `reset` xóa mọi event đã ghi; `report` trả về một chuỗi tóm tắt đọc được.
- **PyTorch:** `torch.profiler.profile` + `key_averages().table()` — cùng vai trò.

### `profiler.record` / `profiler.profile`

- **Chữ ký:** `record(name, elapsed_ms) -> None`; `profile(name) -> list[dict]` (mỗi dict: `calls`, `total_ms`, `avg_ms`, `max_ms`)
- **Nguồn:** `src/bindings/bind_onnx.cpp:87-92`
- **Tham số:** `record(name, elapsed_ms)` — ghi một event đã tính giờ thủ công. `profile(name)` — trả về các event đã ghi cho `name` (hoặc bắt đầu một khối profiling có tên — xem trường hợp biên).
- **Trả về:** `profile` trả về một danh sách dict event.
- **Trường hợp biên:** `profile("blk")` khi chưa có `record` nào trước đó trả về danh sách rỗng. `record` là đường thủ công; không có dạng context-manager.
- **PyTorch:** `torch.profiler.record_function` — cùng ý tưởng ghi thủ công.

<!-- ex: id=r13-profiler device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import time
import litetorch as lt

lt.profiler.enable()
lt.profiler.reset()
t0 = time.perf_counter()
x = lt.Tensor.from_vector([1.0] * 4, [4], lt.Device("cpu"))
lt.profiler.record("add", (time.perf_counter() - t0) * 1000.0)
print("enabled:", lt.profiler.is_enabled())
print("events:", lt.profiler.profile("add"))
print(lt.profiler.report())
lt.profiler.disable()
```

## Hub

### `hub.download_url_to_file` / `hub.load_state_dict_from_url`

- **Chữ ký:** `download_url_to_file(url, dst, hash_prefix=None, progress=True) -> None`; `load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False) -> dict`
- **Nguồn:** `src/bindings/bind_onnx.cpp:110-112` (binding hub nằm cùng file với profiler)
- **Tham số:** `url` — URL của file. `dst` / `model_dir` — đích cục bộ. `hash_prefix` — tiền tố hash kỳ vọng tùy chọn. `check_hash` — xác minh hash sau khi tải.
- **Trả về:** `download_url_to_file` ghi file; `load_state_dict_from_url` tải (nếu chưa cache) và trả về state dict.
- **Trường hợp biên:** cả hai đều cần mạng — các ví dụ là fragment, không phải xác minh chạy được.
- **PyTorch:** `torch.hub.download_url_to_file` / `load_state_dict_from_url` — cùng chữ ký, cùng ngữ nghĩa.

### `hub.get_dir` / `hub.set_dir` / `hub.is_url`

- **Chữ ký:** `get_dir() -> str`; `set_dir(d) -> None`; `is_url(s) -> bool`
- **Nguồn:** `src/bindings/bind_onnx.cpp:107-109`
- **Trả về:** `get_dir` trả về thư mục cache hub (mặc định `~/.cache/litetorch/hub`); `set_dir` thay đổi nó; `is_url` báo cáo một chuỗi có giống URL không.
- **PyTorch:** `torch.hub.get_dir` / `set_dir` — cùng vai trò.

<!-- ex: id=r13-hub device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

print("hub dir:", lt.hub.get_dir())
print("is_url:", lt.hub.is_url("https://example.com/w.pt"), lt.hub.is_url("local.pt"))
lt.hub.set_dir("/tmp/lt-hub-test")
print("hub dir after set:", lt.hub.get_dir())
```

<!-- ex: id=r13-hub-download device=cpu backend=cpu runtime=n/a torch-parity=n/a status=fragment -->
```python skip-run
import litetorch as lt

lt.hub.download_url_to_file("https://example.com/weights.pt", "/tmp/weights.pt")
sd = lt.hub.load_state_dict_from_url("https://example.com/weights.pt")
print("keys:", list(sd.keys()))
```
