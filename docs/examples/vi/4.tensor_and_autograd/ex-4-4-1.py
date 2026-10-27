import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()
x = lt.Tensor.from_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], device, False)

print("Kich thuoc ban dau:", x.shape)
print("Strides ban dau:", x.strides)

x_transposed = x.transpose(0, 1)
print("Kich thuoc sau transpose:", x_transposed.shape)
print("Strides sau transpose:", x_transposed.strides)

x_contiguous = x_transposed.contiguous()
print("Strides lien tuc:", x_contiguous.strides)

x_view = x.view([3, 2])
print("Kich thuoc view:", x_view.shape)
