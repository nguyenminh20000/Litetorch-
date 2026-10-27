import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], device, False)

print("Original shape:", x.shape)
print("Original strides:", x.strides)

x_transposed = x.transpose(0, 1)
print("Transposed shape:", x_transposed.shape)
print("Transposed strides:", x_transposed.strides)

x_contiguous = x_transposed.contiguous()
print("Contiguous strides:", x_contiguous.strides)

x_view = x.view([3, 2])
print("View shape:", x_view.shape)
