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

model = nn.Sequential([nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2)])
x = t([3, 8])
g = lt.onnx.trace_module(model, x)
check("trace num_nodes>=2", g.num_nodes() >= 2)
ops = g.node_op_types()
check("trace has Gemm", "Gemm" in ops)
check("trace has Relu", "Relu" in ops)

g2 = lt.onnx.Graph()
g2.add_input("x", [2, 3])
g2.add_output("y", [2, 3])
g2.add_node("Relu", ["x"], ["y"])
check("manual graph nodes", g2.num_nodes() == 1)
txt = lt.onnx.export_onnx_text(g2)
check("text has ir_version", "ir_version" in txt)
check("text has Relu", "Relu" in txt)
check("text has dims", "dim_value: 2" in txt)

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "m.onnx")
    lt.onnx.export(model, x, p)
    check("export file exists", os.path.exists(p))
    with open(p) as f:
        content = f.read()
    check("export has Gemm", "Gemm" in content)
    check("export has initializer", "initializer" in content)
    check("export has opset", "opset_import" in content)

lt.profiler.reset()
lt.profiler.enable()
check("profiler enabled", lt.profiler.is_enabled())
with lt.profiler.profile("mm_block"):
    a = t([32, 32])
    b = t([32, 32])
    c = lt.Ops.matmul(a, b)
lt.profiler.disable()
check("profiler disabled", not lt.profiler.is_enabled())
rep = lt.profiler.report()
check("report has mm_block", "mm_block" in rep)
evs = lt.profiler.events()
check("events nonempty", len(evs) > 0)
lt.profiler.reset()
check("reset clears", len(lt.profiler.events()) == 0)

check("is_url http", lt.hub.is_url("https://example.com/m.safetensors"))
check("is_url local", not lt.hub.is_url("/tmp/m.safetensors"))
with tempfile.TemporaryDirectory() as d:
    lt.hub.set_dir(d)
    check("hub set_dir", lt.hub.get_dir() == d)
    src = os.path.join(d, "src.bin")
    with open(src, "wb") as f:
        f.write(b"hub-download-test")
    dst = os.path.join(d, "dst.bin")
    got = lt.hub.download_url_to_file("file://" + src, dst, show_progress=False)
    check("hub download path", got == dst)
    with open(dst, "rb") as f:
        check("hub download content", f.read() == b"hub-download-test")

print(f"\nPASS={PASS} FAIL={FAIL}")
raise SystemExit(1 if FAIL else 0)
