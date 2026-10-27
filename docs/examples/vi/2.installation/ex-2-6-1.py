import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

is_gpu = lt.is_gpu_available()
backend = lt.get_backend_name()
device = lt.auto_device()

print(f"GPU Kha dung: {is_gpu}")
print(f"Backend dang su dung: {backend}")
print(f"Thiet bi mac dinh: {device}")

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
loss = lt.Ops.sum(y)
loss.backward()

print("Kich thuoc ma tran ket qua:", y.shape)
print("Gia tri loss:", loss.item())
print("Gia tri dao ham cua x:", x.grad.to_vector())
