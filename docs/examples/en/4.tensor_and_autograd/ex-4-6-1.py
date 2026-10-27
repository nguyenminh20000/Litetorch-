import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

# Inspect current cached memory
print("CPU Cache Bytes:", lt.get_cached_cpu_bytes())

# Configure max cache threshold to 256MB
lt.set_max_cpu_cache_size(256 * 1024 * 1024)

# Release all cached pages back to OS
lt.empty_cache()
