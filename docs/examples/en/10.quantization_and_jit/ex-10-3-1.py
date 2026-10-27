import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

x_int8 = lt.Tensor.from_vector([10.0, 20.0, -15.0, 30.0], [2, 2], device, False, lt.DataType.INT8)
w_int8 = lt.Tensor.from_vector([5.0, -2.0, 1.0, 4.0], [2, 2], device, False, lt.DataType.INT8)

w8a8_out = lt.Ops.w8a8_matmul(x_int8, w_int8, x_scale=0.01, w_scale=0.02)
print("W8A8 result:", w8a8_out.to_vector())
