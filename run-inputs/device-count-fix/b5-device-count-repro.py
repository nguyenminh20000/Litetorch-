import os
import subprocess
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
import litetorch as lt

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], capture_output=True, text=True, timeout=30)
    print(r.stdout.strip())
except Exception as e:
    print("nvidia-smi failed", e)
try:
    import torch
    print("torch cuda device_count", torch.cuda.device_count())
except Exception as e:
    print("torch unavailable", e)
try:
    import ctypes
    cudart = None
    for name in ("libcudart.so", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    count = ctypes.c_int(0)
    rc = cudart.cudaGetDeviceCount(ctypes.byref(count))
    print("ctypes cudaGetDeviceCount rc", rc, "count", count.value)
except Exception as e:
    print("ctypes cudart failed", e)
print("lt is_available", lt.cuda.is_available())
print("lt device_count", lt.cuda.device_count())
print("lt current_device", lt.cuda.current_device())
print("lt memory before", lt.cuda.memory_allocated())
t = lt.randn([1024, 1024]).to(lt.Device("cuda"))
print("str(t.device)", str(t.device))
print("lt memory after", lt.cuda.memory_allocated())
assert lt.cuda.device_count() == 2, "expected 2 GPUs"
print("B5-DEVICE-COUNT REPRO PASS")
