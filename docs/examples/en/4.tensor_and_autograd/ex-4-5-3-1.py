import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([3.0], [1], device, True)
y = lt.Ops.mul(x, x)

y.backward(create_graph=True)
print("First derivative dy/dx:", x.grad.to_vector())
