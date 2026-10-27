import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

# Tensor can tinh gradient
x = lt.Tensor.from_vector([1.0, -2.0, 3.0], [3], device, True)
bias = lt.Tensor.from_vector([0.5, 0.5, 0.5], [3], device, True)

# Forward qua toan tu fused
out = lt.jit.fused_bias_gelu(x, bias)
loss = lt.Ops.sum(out)

# Lan truyen gradient qua JIT Kernel
loss.backward()

print("x.grad:", x.grad.to_vector())
print("bias.grad:", bias.grad.to_vector())
