import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

x = lt.Tensor.from_vector([float(i) for i in range(100 * 32)], [100, 32], device, False)
y = lt.Tensor.from_vector([float(i % 2) for i in range(100)], [100, 1], device, False)

dataset = lt.data.TensorDataset(x, y)
print("Tong so mau:", dataset.size())
