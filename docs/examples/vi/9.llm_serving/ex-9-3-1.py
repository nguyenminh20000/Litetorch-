import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

cache = lt.nn.PrefixCache.get()
system_tokens = [1, 105, 200, 350]

k_data = lt.Tensor.from_vector([0.1] * 512, [1, 4, 128], lt.auto_device(), False)
v_data = lt.Tensor.from_vector([0.2] * 512, [1, 4, 128], lt.auto_device(), False)

cache.put(system_tokens, k_data, v_data)
cached_pair = cache.get_cache(system_tokens)
