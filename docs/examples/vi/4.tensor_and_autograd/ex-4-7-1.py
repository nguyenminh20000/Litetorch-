import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt
import numpy as np

t = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0], [2, 2], lt.Device(lt.DeviceType.CPU, 0))

arr = t.numpy()
arr[0, 0] = 99.0
print(t.tolist()[0][0])

print(t.dim(), t.ndim)
print(t.half().dtype, t.double().dtype)

a = lt.Tensor.from_vector([1.0, 2.0], [2], lt.Device(lt.DeviceType.CPU, 0))
b = a.int()
print(lt.equal(a, b))
print(b.type_as(a).dtype)
