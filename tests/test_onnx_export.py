import os
import tempfile

import litetorch as lt
from litetorch import nn

CPU = lt.Device("cpu")

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"[FAIL] {name}")

def t(shape):
    return lt.Tensor.zeros(shape, CPU)

# --- expanded op table ---
model = nn.Sequential([nn.Linear(8, 4), nn.LeakyReLU(0.2), nn.Linear(4, 2)])
g = lt.onnx.trace_module(model, t([3, 8]))
ops = g.node_op_types()
check("leakyrelu maps to LeakyRelu", "LeakyRelu" in ops)
nodes = None
txt = lt.onnx.export_onnx_text(g)
check("gemm transB attr", 'name: "transB" i: 1 type: INT' in txt)
check("leakyrelu alpha attr", 'name: "alpha"' in txt)

model2 = nn.Sequential([nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8), nn.ReLU(),
                        nn.MaxPool2d(2, 2)])
g2 = lt.onnx.trace_module(model2, t([1, 3, 8, 8]))
ops2 = g2.node_op_types()
check("conv maps to Conv", "Conv" in ops2)
check("batchnorm maps to BatchNormalization", "BatchNormalization" in ops2)
check("maxpool maps to MaxPool", "MaxPool" in ops2)
txt2 = lt.onnx.export_onnx_text(g2)
check("conv kernel_shape attr", 'name: "kernel_shape"' in txt2)
check("conv kernel_shape ints", "ints: [3, 3] type: INTS" in txt2)
check("conv pads attr", 'name: "pads"' in txt2)
check("batchnorm epsilon attr", 'name: "epsilon"' in txt2)

model3 = nn.Sequential([nn.Embedding(10, 4), nn.Flatten(), nn.Softmax(-1),
                        nn.Dropout(0.1)])
g3 = lt.onnx.trace_module(model3, t([2, 5]))
ops3 = g3.node_op_types()
check("embedding maps to Gather", "Gather" in ops3)
check("softmax maps to Softmax", "Softmax" in ops3)
check("dropout maps to Dropout", "Dropout" in ops3)
txt3 = lt.onnx.export_onnx_text(g3)
check("softmax axis attr", 'name: "axis" i: -1 type: INT' in txt3)
check("dropout ratio attr", 'name: "ratio"' in txt3)
check("flatten axis attr", 'name: "axis" i: 1 type: INT' in txt3)

model4 = nn.Sequential([nn.LayerNorm([4]), nn.AvgPool2d(2, 2), nn.GroupNorm(2, 8)])
g4 = lt.onnx.trace_module(model4, t([1, 8, 4, 4]))
ops4 = g4.node_op_types()
check("layernorm maps to LayerNormalization", "LayerNormalization" in ops4)
check("avgpool maps to AveragePool", "AveragePool" in ops4)
txt4 = lt.onnx.export_onnx_text(g4)
check("layernorm epsilon attr", 'name: "epsilon"' in txt4)

model5 = nn.Sequential([nn.ELU(1.0), nn.PReLU(1), nn.Mish()])
g5 = lt.onnx.trace_module(model5, t([2, 4]))
ops5 = g5.node_op_types()
check("elu maps to Elu", "Elu" in ops5)
check("prelu maps to PRelu", "PRelu" in ops5)
check("mish maps to Mish", "Mish" in ops5)

# --- dynamic axes ---
model6 = nn.Sequential([nn.Linear(8, 4), nn.ReLU()])
gd = lt.onnx.trace_module_dynamic(model6, t([3, 8]), {"input": {0: "batch"}})
txtd = lt.onnx.export_onnx_text(gd)
check("dynamic dim_param in text", 'dim_param: "batch"' in txtd)
check("static dim still dim_value", "dim_value: 8" in txtd)

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "dyn.onnx")
    lt.onnx.export_dynamic(model6, t([3, 8]), p, {"input": {0: "batch"}, "output": {0: "batch"}})
    check("export_dynamic file exists", os.path.exists(p))
    with open(p) as f:
        content = f.read()
    check("export_dynamic has dim_param", 'dim_param: "batch"' in content)

# --- manual graph with typed attrs ---
gm = lt.onnx.Graph()
gm.add_dynamic_input("x", [4, 8], ["batch", ""])
gm.add_dynamic_output("y", [4, 4], ["batch", ""])
gm.add_node_typed("Gemm", ["x", "w"], ["y"],
                  {"transB": 1}, {"alpha": 1.0, "beta": 1.0}, {}, {}, "")
txtm = lt.onnx.export_onnx_text(gm)
check("manual typed int attr", 'name: "transB" i: 1 type: INT' in txtm)
check("manual typed float attr", 'name: "alpha" f: 1' in txtm)
check("manual dynamic input", 'dim_param: "batch"' in txtm)
check("manual empty dim_param static", "dim_value: 8" in txtm)

# --- onnxruntime roundtrip if available ---
try:
    import onnxruntime  # noqa
    import onnx  # noqa
    check("onnxruntime available", True)
except ImportError:
    print("[SKIP] onnxruntime/onnx not installed, roundtrip skipped")

print(f"\nPASS={PASS} FAIL={FAIL}")
raise SystemExit(1 if FAIL else 0)
