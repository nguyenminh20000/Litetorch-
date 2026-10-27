import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([1.0, 2.0], [2], device, True)

def custom_hook(grad):
    scale = lt.Tensor.from_vector([2.0, 2.0], [2], device, False)
    return lt.Ops.mul(grad, scale)

x.register_hook(custom_hook)

y = lt.Ops.sum(x)
y.backward()

print("Dao ham sau hook:", x.grad.to_vector())
