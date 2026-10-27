import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([2.0, 3.0], [2], device, True)
w = lt.Tensor.from_vector([4.0, 5.0], [2], device, True)

y = lt.Ops.mul(x, w)
loss = lt.Ops.sum(y)

loss.backward()

print("Gradient x.grad:", x.grad.to_vector())
print("Gradient w.grad:", w.grad.to_vector())
