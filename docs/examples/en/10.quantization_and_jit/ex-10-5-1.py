import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

w_fp32 = lt.Tensor.from_vector([0.1, -0.5, 1.2, -2.4], [2, 2], device, False, lt.DataType.FP32)
w_fp4 = w_fp32.cast(lt.DataType.FP4_E2M1)
w_nf4 = w_fp32.cast(lt.DataType.NF4)

print("FP4 dtype:", w_fp4.dtype)
print("NF4 dtype:", w_nf4.dtype)
