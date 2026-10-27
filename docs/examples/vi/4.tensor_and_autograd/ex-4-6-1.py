import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

# Kiem tra dung luong cache hien tai
print("CPU Cache Bytes:", lt.get_cached_cpu_bytes())

# Gioi han muc cache CPU toi da 256MB
lt.set_max_cpu_cache_size(256 * 1024 * 1024)

# Giai phong toan bo bo nho cache ve he dieu hanh
lt.empty_cache()
