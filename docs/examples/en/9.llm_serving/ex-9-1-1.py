import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

q = lt.Tensor.from_vector([0.1] * 128, [1, 1, 128], device, False)
k_cache = lt.Tensor.from_vector([0.2] * (16 * 16 * 128), [16, 16, 128], device, False)
v_cache = lt.Tensor.from_vector([0.3] * (16 * 16 * 128), [16, 16, 128], device, False)
block_tables = lt.Tensor.from_vector([0.0, 1.0], [1, 2], device, False)
context_lens = lt.Tensor.from_vector([32.0], [1], device, False)

out = lt.Ops.paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size=16)
print("PagedAttention output shape:", out.shape)
