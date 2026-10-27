import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

is_gpu = lt.is_gpu_available()
backend = lt.get_backend_name()
device = lt.auto_device()

print(f"GPU Available: {is_gpu}")
print(f"Backend Name: {backend}")
print(f"Active Device: {device}")

x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, True)
y = lt.Ops.matmul(x, x)
loss = lt.Ops.sum(y)
loss.backward()

print("Result shape:", y.shape)
print("Loss value:", loss.item())
print("Gradient:", x.grad.to_vector())
