import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
b = lt.Tensor.from_vector([0.5, 1.5, -1.0, 2.0], [2, 2], device, True)

c = lt.Ops.matmul(a, b)
loss = lt.Ops.sum(c)

loss.backward()

print("Tich ma tran c:", c.to_vector())
print("Gia tri loss:", loss.item())
print("Dao ham da/dc:", a.grad.to_vector())
