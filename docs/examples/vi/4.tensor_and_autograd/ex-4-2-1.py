import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, False, lt.DataType.FP32)
x_half = x.cast(lt.DataType.FP16)
print("Kieu du lieu moi:", x_half.dtype)
