import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

cpu_dev = lt.Device(lt.DeviceType.CPU, 0)
gpu_dev = lt.Device(lt.DeviceType.GPU, 0)
tpu_dev = lt.Device(lt.DeviceType.TPU, 0)

x_cpu = lt.Tensor.from_vector([1.0, 2.0], [2], cpu_dev, False)
x_gpu = x_cpu.to(gpu_dev)
x_tpu = x_cpu.to(tpu_dev)
