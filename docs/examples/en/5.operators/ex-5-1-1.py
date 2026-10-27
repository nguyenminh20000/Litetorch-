import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()
a = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], device, False)
b = lt.Tensor.from_vector([5.0, 6.0, 7.0, 8.0], [2, 2], device, False)

add_res = lt.Ops.add(a, b)
mul_res = lt.Ops.mul(a, b)
matmul_res = lt.Ops.matmul(a, b)
