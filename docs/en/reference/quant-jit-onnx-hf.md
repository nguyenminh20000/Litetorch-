# Quantization, JIT, ONNX, Hugging Face, Inference, Pruning, Profiling, Hub — API Reference

This page covers the model-optimization and model-interchange surface: `litetorch.quantization`, `litetorch.jit`, `litetorch.onnx`, `litetorch.hf`, `litetorch.inference`, `litetorch.prune`, `litetorch.profiler`, `litetorch.hub`.

- **Source of truth:** `src/bindings/bind_quant.cpp`, `bind_jit.cpp`, `bind_onnx.cpp`, `bind_hf.cpp`, `bind_inference.cpp`, `bind_prune.cpp` (Python bindings); `include/litetorch/quantization.h`, `jit.h`, `jit_detail.h`, `onnx.h`, `hf.h`, `inference.h`, `prune.h`, `profiler.h`, `hub.h`.
- **Tree:** agent-v1@02cc271a / local HEAD 89afd67 (docs baseline is the 88a4ba1 extension; B5 re-verifies at head).
- **Conventions:** `cpu = lt.Device("cpu")` in every example. Every example is self-contained and runnable on CPU unless marked `REQUIRES-GPU` / `REQUIRES-TPU` / `REQUIRES-NETWORK`. Torch parity is FP32 1e-4 unless the header says otherwise.
- **Related deviations:** D-01 (`Tensor.data_ptr()` throws on non-FP32 — see `cuda-tpu-autograd.md`); D-09 (GELU uses the tanh approximation — see `plan/differences.md`).

## Quantization

### `quantization.quantize_linear`

- **Signature:** `quantize_linear(linear, bits=8) -> Tensor` (w8a8 — FP32 inputs are quantized per-tensor; see the edge case)
- **Source:** `src/bindings/bind_quant.cpp:5`
- **Parameters:** `linear` — an `nn.Linear`. `bits` — int, default 8.
- **Returns:** the quantized weight tensor (INT8-packed). The returned tensor is the weight, not a module.
- **Edge cases:** the w8a8 path takes FP32 inputs and quantizes them per-tensor at call time — pass FP32 tensors, not pre-quantized INT8 (see the how-to H8.3 note). `bits` other than 8 is accepted but the kernel is tuned for 8.
- **PyTorch:** `torch.ao.quantization.quantize_dynamic` — same role (dynamic per-tensor quantization); torch's API quantizes whole modules, LiteTorch's quantizes one `Linear` at a time.

### `quantization.quantize_dynamic_linear` / `quantize_dynamic_lstm` / `quantize_dynamic_gru`

- **Signature:** `quantize_dynamic_linear(linear) -> nn.Module`; `quantize_dynamic_lstm(lstm) -> nn.Module`; `quantize_dynamic_gru(gru) -> nn.Module`
- **Source:** `src/bindings/bind_quant.cpp:40-49`
- **Parameters:** a `Linear` / `LSTM` / `GRU` module.
- **Returns:** a new module whose forward quantizes activations dynamically (INT8 weights, FP32 activations). The return type is `litetorch.nn.Module` (a wrapped module, not the original class).
- **Edge cases:** the original module is not modified. The quantized module's `forward` has the same shape contract as the original.
- **PyTorch:** `torch.ao.nn.quantized.dynamic.Linear` / `LSTM` / `GRU` — same role.

### `quantization.Calibrator`

- **Signature:** class `Calibrator()`; methods `collect(name, tensor)` · `get_scale(name) -> float` · `get_asymmetric_params(name) -> (float, float)`
- **Source:** `src/bindings/bind_quant.cpp:7-11`
- **Parameters:** `collect(name, tensor)` — records one calibration sample under `name`. `get_scale(name)` — the symmetric scale for `name`. `get_asymmetric_params(name)` — `(scale, zero_point)` for asymmetric quantization.
- **Returns:** `get_scale` returns a float; `get_asymmetric_params` returns a `(scale, zero_point)` tuple.
- **Edge cases:** calling `get_scale` before any `collect` for that name throws. Scales are per-tensor (one scale per name), not per-channel.
- **PyTorch:** `torch.ao.quantization.observer` — same role (collect activation statistics, derive scales).

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

- **Signature:** class `PTQConfig()` with read/write fields `bits` (int), `per_channel` (bool), `symmetric` (bool)
- **Source:** `src/bindings/bind_quant.cpp:28-32`
- **Returns:** a plain config object for post-training quantization. Defaults: `bits=8`, `per_channel=False`, `symmetric=True` (verify the defaults by printing them — they are plain fields, not constructor arguments).
- **PyTorch:** `torch.ao.quantization.QConfig` — same role; torch's config carries observer classes, LiteTorch's carries three flags.

### `quantization.PreparedModel` / `quantization.prepare` / `quantization.convert`

- **Signature:** `prepare(model, config) -> PreparedModel`; `convert(prepared) -> nn.Module`; `PreparedModel.calibrate(data_fn, num_batches=10) -> None`; `PreparedModel.model` (read/write)
- **Source:** `src/bindings/bind_quant.cpp:34-39`
- **Parameters:** `prepare(model, config)` — a module and a `PTQConfig`. `calibrate(data_fn, num_batches=10)` — `data_fn` is a zero-argument callable returning one calibration tensor per call; `num_batches` defaults to 10.
- **Returns:** `prepare` returns a `PreparedModel` wrapping the module (calibration hooks attached); `convert` returns the quantized module (an `nn.Linear` for a `Linear` input).
- **Edge cases:** `convert` before `calibrate` uses the default scale (1.0) — calibrate first for a real scale. `PreparedModel.model` is the underlying module; mutate it only through `calibrate`/`convert`.
- **PyTorch:** `torch.ao.quantization.prepare` / `convert` — same two-step PTQ flow.

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

- **Signature:** `QATLinear(linear, weight_scale, weight_zero_point, act_scale, act_zero_point, bits)`
- **Source:** `src/bindings/bind_quant.cpp:13-26`
- **Parameters:** `linear` — an `nn.Linear`. `weight_scale` / `weight_zero_point` — fake-quant parameters for the weight. `act_scale` / `act_zero_point` — fake-quant parameters for the activation. `bits` — int.
- **Attributes (read/write):** `weight`, `bias`, `weight_scale`, `weight_zero_point`, `act_scale`, `act_zero_point`, `bits`, `qat_enabled`.
- **Methods:** `forward(input)` · `parameters()` · `to(device)`.
- **Returns:** a module that runs fake quantization in the forward pass (quantize then dequantize) so training sees quantization error. Set `qat_enabled = False` to run it as a plain linear.
- **Edge cases:** scales of 0.0 make the fake-quant a no-op only if `qat_enabled` is False; with `qat_enabled` True a zero scale divides by zero — set real scales from a `Calibrator` first.
- **PyTorch:** `torch.ao.nn.qat.Linear` — same role (fake-quant training).

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

- **Signature:** enum with `INPUT`, `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `RELU`, `GELU`, `SIGMOID`, `TANH`, `SQRT`, `EXP`, `LOG`, `ABS`, `CONST`
- **Source:** `src/bindings/bind_jit.cpp:6-23`
- **Returns:** the operation kind of a `JITVar` node.
- **PyTorch:** `torch.jit` has no user-visible op enum; this is LiteTorch's own IR vocabulary.

### `jit.JITVar`

- **Signature:** `JITVar(op, name)`; `JITVar(val)`; read/write `op`, `name`, `val`
- **Source:** `src/bindings/bind_jit.cpp:24-41`
- **Parameters:** `JITVar(op, name)` — a symbolic node of kind `op` named `name`. `JITVar(val)` — a constant node holding float `val`.
- **Operators:** `+` / `-` / `*` / `/` (var-var and var-float, both orders) and unary `-`. Each builds a new `JITVar` node.
- **Returns:** a symbolic expression node. It holds no data — it is a graph, not a tensor.
- **PyTorch:** no direct equivalent; closest is `torch.fx` symbolic tracing, which records operations on `Proxy` objects.

### `jit.relu` / `jit.gelu` / `jit.sigmoid` / `jit.tanh` / `jit.sqrt` / `jit.exp` / `jit.log` / `jit.abs`

- **Signature:** `relu(a) -> JITVar` (and the same for the other seven)
- **Source:** `src/bindings/bind_jit.cpp:43-50`
- **Parameters:** `a` — a `JITVar`.
- **Returns:** a new `JITVar` node applying the unary op. GELU here is the tanh approximation (D-09).
- **PyTorch:** `torch.ops.aten.relu` etc. — same operations, applied to tensors rather than symbolic nodes.

### `jit.fused_bias_relu` / `jit.fused_bias_gelu` / `jit.fused_residual_add` / `jit.fused_residual_gelu`

- **Signature:** `fused_bias_relu(x, bias) -> Tensor`; `fused_bias_gelu(x, bias) -> Tensor`; `fused_residual_add(x, residual) -> Tensor`; `fused_residual_gelu(x, residual) -> Tensor`
- **Source:** `src/bindings/bind_jit.cpp:51-79`
- **Parameters:** `x` — input tensor; `bias` / `residual` — the second operand.
- **Returns:** the fused result: `relu(x + bias)`, `gelu(x + bias)`, `x + residual`, `gelu(x + residual)`.
- **Edge cases:** `fused_bias_relu` / `fused_bias_gelu` require `x` and `bias` to have **identical shapes** — a bias of shape `[4]` against an `x` of shape `[1, 4]` raises `RuntimeError: JITFunction requires all inputs to have identical shapes`. Broadcast the bias to `x`'s shape first (e.g. `bias.reshape([1, 4])`). This is a real limitation, not a docs error.
- **PyTorch:** no single equivalent; torch fuses these inside `torch.compile` / Inductor, never as named functions.

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

- **Signature:** `JITFunction(name, expr, inputs)`; `__call__(args) -> Tensor`; `forward(args) -> Tensor`; `save(filepath) -> None`; `load(filepath) -> JITFunction` (static)
- **Source:** `src/bindings/bind_jit.cpp:80-86`
- **Parameters:** `name` — str, the function name. `expr` — the output `JITVar`. `inputs` — list of input `JITVar`s, in call order.
- **Returns:** calling the function with a list of tensors (same order as `inputs`) returns the evaluated tensor. `save` writes the compiled function to disk; `load` reads it back.
- **Edge cases:** the number of call arguments must match `len(inputs)`. All inputs must have identical shapes (same rule as `fused_bias_relu`).
- **PyTorch:** `torch.jit.ScriptFunction` — same role (a compiled, serializable function); torch's `torch.jit.script` compiles Python source, LiteTorch's `JITFunction` compiles a `JITVar` graph.

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

- **Signature:** static methods `trace(inputs, func) -> JITFunction`; `derivative(var, wrt) -> JITVar`; `serialize_var(var) -> str`; `deserialize_var(str) -> JITVar`
- **Source:** `src/bindings/bind_jit.cpp:88-92`
- **Parameters:** `trace(inputs, func)` — `inputs` is a sequence of tensors; `func` is a callable taking a list of tensors and returning a tensor. `derivative(var, wrt)` — the symbolic derivative of `var` with respect to `wrt`. `serialize_var` / `deserialize_var` — text round-trip of a `JITVar`.
- **Returns:** `trace` returns a `JITFunction` wrapping the traced graph — call it with a list of tensors to evaluate it; `derivative` returns the derivative `JITVar`.
- **Edge cases (verified):** `func` must use `lt.Ops.*` on the passed tensors — `trace([x], lambda ts: lt.Ops.relu(ts[0]))` works. Passing a `JITVar` to `trace` raises `TypeError` (it takes tensors, not vars). Building an intermediate with plain `+` inside `func` and returning it raises `RuntimeError: Tracer found an expired intermediate tensor; keep every intermediate alive until trace returns` — even holding the intermediate in a dict does not help; use `lt.Ops.*` calls whose results flow directly into the return value.
- **PyTorch:** `torch.jit.trace` — same idea (record a graph by running the function); torch's tracer accepts arbitrary Python, LiteTorch's accepts only `lt.Ops.*` calls on the inputs.

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

- **Signature:** classes with read/write fields. `ValueInfo`: `name`, `shape`, `dtype`, `dim_params`. `Node`: `op_type`, `name`, `inputs`, `outputs`, `attrs`, `int_attrs`, `float_attrs`, `ints_attrs`, `domain`. `Graph`: `name`; methods `add_input(name, shape, dtype)` · `add_output(name, shape, dtype)` · `add_node(op_type, inputs, outputs)` · `add_dynamic_input(name, dim_params, dtype)` · `add_dynamic_output(name, dim_params, dtype)` · `add_node_typed(op_type, inputs, outputs, attrs)` · `add_initializer(name, tensor)` · `num_nodes()` · `node_op_types()`.
- **Source:** `src/bindings/bind_onnx.cpp:15-63`
- **Returns:** the three building blocks of an ONNX graph. `Graph.num_nodes()` returns the node count; `node_op_types()` returns the list of op types in order.
- **Edge cases:** `add_node_typed` takes an attrs dict; plain `add_node` takes none. `dim_params` holds symbolic dimension names (e.g. `"batch"`) for dynamic shapes.
- **PyTorch:** `torch.onnx` builds the same three structures internally; LiteTorch exposes them directly.

### `onnx.export_onnx_text` / `onnx.export_onnx_binary` / `onnx.export_onnx`

- **Signature:** `export_onnx_text(graph) -> str`; `export_onnx_binary(graph) -> bytes`; `export_onnx(graph, path) -> None`
- **Source:** `src/bindings/bind_onnx.cpp:64-70`
- **Parameters:** `graph` — an `onnx.Graph`. `path` — output file path (for `export_onnx`).
- **Returns:** the text form starts `ir_version: 8 | producer_name: "litetorch" | graph { name: "litetorch" ...`; the binary form is the protobuf encoding; `export_onnx` writes the binary form to `path`.
- **PyTorch:** `torch.onnx.export` — same role; torch writes a full model file, LiteTorch writes the graph you built.

### `onnx.trace_module` / `onnx.trace_module_dynamic`

- **Signature:** `trace_module(module, example_input) -> Graph`; `trace_module_dynamic(module, example_input, dynamic_axes) -> Graph`
- **Source:** `src/bindings/bind_onnx.cpp:72-74`
- **Parameters:** `module` — an `nn.Module`. `example_input` — a tensor to trace with. `dynamic_axes` — dict mapping input names to dynamic dimension names.
- **Returns:** the traced `Graph`.
- **PyTorch:** `torch.onnx.export` with `dynamo=True` — same role (trace a module into a graph).

### `onnx.export` / `onnx.export_dynamic`

- **Signature:** `export(module, example_input, path) -> None`; `export_dynamic(module, example_input, dynamic_axes, path) -> None`
- **Source:** `src/bindings/bind_onnx.cpp:76-78`
- **Parameters:** `module`, `example_input`, `dynamic_axes` as above; `path` — output file.
- **Returns:** `None`. Writes the traced model to `path` in one call.
- **PyTorch:** `torch.onnx.export(model, args, path)` — same one-call shape.

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

- **Signature:** class `ModelConfig()` with read/write fields `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `hidden_act`, `rms_norm_eps`, `layer_norm_eps`, `n_layer`, `n_head`, `n_embd`, `n_positions`
- **Source:** `src/bindings/bind_hf.cpp:7-24`
- **Returns:** a plain config object. The GPT-style fields (`n_layer`, `n_head`, `n_embd`, `n_positions`) and the Llama-style fields (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `rms_norm_eps`) live side by side — set the ones your model type uses.
- **PyTorch:** `transformers.PretrainedConfig` — same role; LiteTorch's is a flat struct, not a class hierarchy.

### `hf.PretrainedBundle`

- **Signature:** class `PretrainedBundle()` with read/write `config` (a `ModelConfig`) and `state_dict` (dict of name to tensor)
- **Source:** `src/bindings/bind_hf.cpp:26-29`
- **Returns:** the result of `from_pretrained`: config + weights together.
- **PyTorch:** `transformers.AutoModel.from_pretrained` return — same pairing.

### `hf.load_config`

- **Signature:** `load_config(repo_or_dir, revision="main") -> ModelConfig`
- **Source:** `src/bindings/bind_hf.cpp:31`
- **Parameters:** `repo_or_dir` — a Hugging Face repo id (e.g. `"gpt2"`) or a local directory holding `config.json`. `revision` — default `"main"`.
- **Returns:** the parsed `ModelConfig`.
- **Edge cases:** a repo id needs network; a local directory needs a `config.json` in it.
- **PyTorch:** `transformers.AutoConfig.from_pretrained` — same role.

### `hf.from_pretrained`

- **Signature:** `from_pretrained(repo_id, revision="main", cache_dir=None) -> PretrainedBundle`
- **Source:** `src/bindings/bind_hf.cpp:33`
- **Parameters:** `repo_id` — a Hugging Face repo id. `revision` — default `"main"`. `cache_dir` — optional local cache directory.
- **Returns:** a `PretrainedBundle` (config + state dict).
- **Edge cases:** downloads weights from the Hub — needs network and can take minutes for large models. This example is a fragment, not a runnable verification (no network in the docs harness).
- **PyTorch:** `transformers.AutoModel.from_pretrained` — same role.

<!-- ex: id=r13-hf-pretrained device=cpu backend=cpu runtime=n/a torch-parity=n/a status=fragment -->
```python skip-run
import litetorch as lt

bundle = lt.hf.from_pretrained("gpt2")
print("model type:", bundle.config.model_type)
print("weights:", len(bundle.state_dict))
```

### `hf.weight_name_map`

- **Signature:** `weight_name_map(model_type) -> dict[str, str]`
- **Source:** `src/bindings/bind_hf.cpp:35`
- **Parameters:** `model_type` — e.g. `"gpt2"`, `"llama"`, `"bert"`.
- **Returns:** a dict mapping Hugging Face weight names to LiteTorch weight names. For `"gpt2"`: `transformer.wte.weight` → `embed.weight`, `transformer.ln_f.bias` → `final_norm.bias`, `lm_head.weight` → `lm_head.weight`.
- **Edge cases:** an unknown `model_type` returns an empty dict (it does not throw).
- **PyTorch:** no direct equivalent; this is a LiteTorch-side rename table.

### `hf.remap_state_dict`

- **Signature:** `remap_state_dict(state_dict, model_type) -> dict[str, Tensor]`
- **Source:** `src/bindings/bind_hf.cpp:36`
- **Parameters:** `state_dict` — dict of name to `Tensor` (values must be tensors — a dict of ints raises `TypeError`). `model_type` — e.g. `"gpt2"`.
- **Returns:** a new dict with keys renamed via `weight_name_map`. Names with no mapping pass through unchanged.
- **PyTorch:** no direct equivalent; `transformers` models rename internally on load.

<!-- ex: id=r13-hf-remap device=cpu backend=cpu runtime=~1s torch-parity=n/a status=verified-2026-10-26@88a4ba1 -->
```python
import litetorch as lt

cpu = lt.Device("cpu")
x = lt.Tensor.from_vector([0.1] * 4, [1, 4], cpu)
sd = {"transformer.wte.weight": x, "custom.weight": x}
print("gpt2 map:", lt.hf.weight_name_map("gpt2")["transformer.wte.weight"])
print("remapped keys:", list(lt.hf.remap_state_dict(sd, "gpt2").keys()))
```

## Inference

### `inference.CudaGraph`

- **Signature:** class `CudaGraph()`; methods `capture(fn)` · `replay()` · `captured() -> bool`
- **Source:** `src/bindings/bind_inference.cpp:7-13`
- **Parameters:** `capture(fn)` — a zero-argument callable whose CUDA work is recorded into the graph.
- **Returns:** `replay()` re-runs the recorded graph; `captured()` reports whether a graph is held.
- **Edge cases:** capture requires a CUDA device — on a CPU-only box it throws. This is a `REQUIRES-GPU` entry; the example below is a fragment.
- **PyTorch:** `torch.cuda.CUDAGraph` — same role (`capture_begin` / `replay`).

### `inference.capture_cuda_graph`

- **Signature:** `capture_cuda_graph(fn) -> CudaGraph`
- **Source:** `src/bindings/bind_inference.cpp:15-17`
- **Parameters:** `fn` — a zero-argument callable.
- **Returns:** a captured `CudaGraph`.
- **Edge cases:** `REQUIRES-GPU`.
- **PyTorch:** `torch.cuda.CUDAGraph` + `g.capture_begin()` — same role.

<!-- ex: id=r13-cudagraph device=gpu backend=cuda runtime=n/a torch-parity=n/a status=fragment -->
```python skip-run
import litetorch as lt

g = lt.inference.capture_cuda_graph(lambda: None)
print("captured:", g.captured())
g.replay()
```

### `inference.FusionReport`

- **Signature:** class with read-only fields `conv_bn_folded`, `layernorm_detected`, `linear_bias_kernel_fused`, `notes`
- **Source:** `src/bindings/bind_inference.cpp:18-22`
- **Returns:** the result of a fusion-detection call: counts/flags plus human-readable `notes`.
- **PyTorch:** no direct equivalent; this is LiteTorch's own fusion report.

### `inference.fold_conv_bn` / `inference.fuse_conv_bn`

- **Signature:** `fold_conv_bn(conv, bn) -> nn.Module`; `fuse_conv_bn(seq) -> FusionReport`
- **Source:** `src/bindings/bind_inference.cpp:24-26`
- **Parameters:** `fold_conv_bn(conv, bn)` — a `Conv2d` and a `BatchNorm2d`. `fuse_conv_bn(seq)` — a `Sequential` (or list of modules) to scan for Conv+BN pairs.
- **Returns:** `fold_conv_bn` returns a new module with the BN folded into the convolution weights; `fuse_conv_bn` returns a `FusionReport`.
- **Edge cases:** `fold_conv_bn` requires the BatchNorm to be in **eval mode** — folding training-mode batch statistics is mathematically wrong, so it raises `RuntimeError: fold_conv_bn: BatchNorm2d must be in eval mode ... (call eval() first)`. Call `bn.eval()` before folding.
- **PyTorch:** `torch.ao.quantization.fuse_modules` — same role (fold BN into the preceding conv).

### `inference.detect_layernorm_candidates` / `inference.detect_layernorm_fusion`

- **Signature:** `detect_layernorm_candidates(model) -> FusionReport`; `detect_layernorm_fusion(model) -> FusionReport`
- **Source:** `src/bindings/bind_inference.cpp:27-30`
- **Parameters:** `model` — an `nn.Module` to scan.
- **Returns:** a `FusionReport` with `layernorm_detected` set and `notes` listing the fusable sites.
- **PyTorch:** no direct equivalent; Inductor does this detection internally.

### `inference.verify_linear_bias_fusion`

- **Signature:** `verify_linear_bias_fusion() -> bool`
- **Source:** `src/bindings/bind_inference.cpp:31`
- **Returns:** `True` when the fused linear+bias kernel matches the unfused result within tolerance on this build (verified: returns `True` on the 88a4ba1 CPU extension).
- **PyTorch:** no direct equivalent.

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

- **Signature:** `l1_unstructured(module, name, amount) -> None`; `l1_unstructured(module, name, amount, dim) -> None`
- **Source:** `src/bindings/bind_prune.cpp:6-15`
- **Parameters:** `module` — an `nn.Module`. `name` — parameter name, e.g. `"weight"`. `amount` — fraction (float in (0, 1)) or count (int) of entries to prune. `dim` — optional int, the dimension to prune along.
- **Returns:** `None`. Zeroes the smallest-magnitude entries (by L1) and registers a pruning mask on the module.
- **Edge cases:** `amount` as a float must be in (0, 1); as an int it must be less than the parameter size. Pruning twice composes the masks.
- **PyTorch:** `torch.nn.utils.prune.l1_unstructured` — same signature and semantics.

### `prune.random_unstructured`

- **Signature:** `random_unstructured(module, name, amount) -> None`; `random_unstructured(module, name, amount, dim) -> None`
- **Source:** `src/bindings/bind_prune.cpp:16-23`
- **Parameters:** same as `l1_unstructured`.
- **Returns:** `None`. Zeroes a uniformly random subset of entries.
- **Edge cases:** the random choice is not seeded through `lt.manual_seed` — do not rely on reproducibility across runs.
- **PyTorch:** `torch.nn.utils.prune.random_unstructured` — same signature and semantics.

### `prune.remove`

- **Signature:** `remove(module, name) -> None`
- **Source:** `src/bindings/bind_prune.cpp:24-28`
- **Parameters:** `module`, `name` — the pruned parameter.
- **Returns:** `None`. Makes the pruning permanent (bakes the mask into the parameter) and removes the mask.
- **PyTorch:** `torch.nn.utils.prune.remove` — same role.

### `prune.is_pruned`

- **Signature:** `is_pruned(module, name) -> bool`
- **Source:** `src/bindings/bind_prune.cpp:29-32`
- **Parameters:** `module`, `name`.
- **Returns:** `True` when a pruning mask is registered on that parameter.
- **PyTorch:** `torch.nn.utils.prune.is_pruned` — same role.

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

- **Signature:** `enable() -> None`; `disable() -> None`; `is_enabled() -> bool`
- **Source:** `src/bindings/bind_onnx.cpp:82-84`
- **Returns:** the CPU profiler is a global switch; `is_enabled` reports its state.
- **PyTorch:** `torch.profiler.profile` context manager — same role, scoped instead of global.

### `profiler.reset` / `profiler.report`

- **Signature:** `reset() -> None`; `report() -> str`
- **Source:** `src/bindings/bind_onnx.cpp:85-86`
- **Returns:** `reset` clears all recorded events; `report` returns a human-readable summary string.
- **PyTorch:** `torch.profiler.profile` + `key_averages().table()` — same role.

### `profiler.record` / `profiler.profile`

- **Signature:** `record(name, elapsed_ms) -> None`; `profile(name) -> list[dict]` (each dict: `calls`, `total_ms`, `avg_ms`, `max_ms`)
- **Source:** `src/bindings/bind_onnx.cpp:87-92`
- **Parameters:** `record(name, elapsed_ms)` — logs one timed event manually. `profile(name)` — returns the recorded events for `name` (or starts a named profiling block — see the edge case).
- **Returns:** `profile` returns a list of event dicts.
- **Edge cases:** `profile("blk")` with no prior `record` returns an empty list. `record` is the manual path; there is no context-manager form.
- **PyTorch:** `torch.profiler.record_function` — same manual-recording idea.

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

- **Signature:** `download_url_to_file(url, dst, hash_prefix=None, progress=True) -> None`; `load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False) -> dict`
- **Source:** `src/bindings/bind_onnx.cpp:110-112` (hub bindings live in the same file as the profiler)
- **Parameters:** `url` — the file URL. `dst` / `model_dir` — local destination. `hash_prefix` — optional expected hash prefix. `check_hash` — verify the hash after download.
- **Returns:** `download_url_to_file` writes the file; `load_state_dict_from_url` downloads (if not cached) and returns the state dict.
- **Edge cases:** both need network — the examples are fragments, not runnable verifications.
- **PyTorch:** `torch.hub.download_url_to_file` / `load_state_dict_from_url` — same signatures, same semantics.

### `hub.get_dir` / `hub.set_dir` / `hub.is_url`

- **Signature:** `get_dir() -> str`; `set_dir(d) -> None`; `is_url(s) -> bool`
- **Source:** `src/bindings/bind_onnx.cpp:107-109`
- **Returns:** `get_dir` returns the hub cache directory (default `~/.cache/litetorch/hub`); `set_dir` changes it; `is_url` reports whether a string looks like a URL.
- **PyTorch:** `torch.hub.get_dir` / `set_dir` — same role.

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
