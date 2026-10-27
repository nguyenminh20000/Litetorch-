import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

p1 = lt.Tensor.from_vector([10.0, 20.0], [2], device, True)
p2 = lt.Tensor.from_vector([30.0, 40.0], [2], device, True)

total_norm = lt.Ops.clip_grad_norm_([p1, p2], max_norm=1.0, norm_type=2.0)
print("Chuan gradient truoc khi cat:", total_norm)
