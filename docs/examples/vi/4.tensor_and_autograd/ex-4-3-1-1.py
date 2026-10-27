import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

print("GPUs:", lt.cuda.device_count())
lt.cuda.set_device(0)
lt.cuda.manual_seed_all(42)
print("Reserved bytes:", lt.cuda.memory_reserved())
