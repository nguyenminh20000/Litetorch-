import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

x_data = [float(i % 10) for i in range(100 * 64)]
y_data = [float(i % 2) for i in range(100 * 1)]

x_tensor = lt.Tensor.from_vector(x_data, [100, 64], device, False)
y_tensor = lt.Tensor.from_vector(y_data, [100, 1], device, False)

dataset = lt.data.TensorDataset(x_tensor, y_tensor)
loader = lt.data.DataLoader(dataset, batch_size=16, shuffle=True, device=device)
