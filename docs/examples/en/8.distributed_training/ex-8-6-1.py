import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()
pg = lt.distributed.ProcessGroup.get()

q = lt.Tensor.from_vector([0.1] * 512, [1, 4, 128, 1], device, True)
k = lt.Tensor.from_vector([0.2] * 512, [1, 4, 128, 1], device, True)
v = lt.Tensor.from_vector([0.3] * 512, [1, 4, 128, 1], device, True)

attn_out = lt.Ops.ring_attention(q, k, v, pg)
