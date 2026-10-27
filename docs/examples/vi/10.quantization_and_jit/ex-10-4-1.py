import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

# Cach 1: Xay dung bieu thuc JIT truc tiep
v1 = lt.jit.JITVar(lt.jit.OpType.INPUT, "x")
v2 = lt.jit.JITVar(lt.jit.OpType.INPUT, "w")
expr = lt.jit.relu(v1 * 2.0 + v2)

jit_fn = lt.jit.JITFunction("fused_relu_add", expr, [v1, v2])

arg1 = lt.Tensor.from_vector([1.0, 2.0], [2], lt.auto_device(), False)
arg2 = lt.Tensor.from_vector([-3.0, 1.0], [2], lt.auto_device(), False)

res = jit_fn([arg1, arg2])
print("JIT output:", res.to_vector())

jit_fn.save("fused_relu.jit")
loaded_fn = lt.jit.JITFunction.load("fused_relu.jit")

# Cach 2: Tu dong truy vet ham Python bang Tracer
def custom_activation(args):
    return lt.Ops.gelu(lt.Ops.add(args[0], args[1]))

traced_fn = lt.jit.Tracer.trace([arg1, arg2], custom_activation, "traced_gelu")
traced_res = traced_fn([arg1, arg2])
print("Traced GELU output:", traced_res.to_vector())
