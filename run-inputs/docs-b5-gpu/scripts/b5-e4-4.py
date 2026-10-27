import os
import sys
TREE = os.environ.get("LITETORCH_TREE") or (sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/litetorch")
sys.path.insert(0, os.path.join(TREE, "python"))
def _hook(t, v, tb):
    print("B5 e4-4 FAIL: %s: %s" % (t.__name__, v))
    sys.exit(1)
sys.excepthook = _hook
import sys
import os
import litetorch as lt
print("cuda available", lt.cuda.is_available())
print("device count", lt.cuda.device_count())
print("backend name", lt.get_backend_name())
print("native lib next to extension", os.path.exists(os.path.join(os.path.dirname(lt.__file__), "liblitetorch_gpu.so")))
lt.cuda.synchronize()
import subprocess
assert lt.cuda.is_available(), "no CUDA backend"
assert lt.cuda.device_count() >= 1, "no GPU device"
assert lt.get_backend_name() in ("CUDA", "ROCm"), "backend is not native GPU"
assert os.path.exists(os.path.join(os.path.dirname(lt.__file__), "liblitetorch_gpu.so")), "liblitetorch_gpu.so missing"
probe_code = "\n".join([
    "import os, sys",
    "sys.path.insert(0, " + repr(os.path.join(TREE, "python")) + ")",
    "import litetorch as lt",
    "print(lt.get_backend_name())",
    "print(lt.cuda.is_available())",
])
probe = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True, timeout=600)
print(probe.stdout)
assert "Native GPU (CUDA/HIP) successfully initialized." in probe.stdout, "init line missing from import log"
assert "OpenCL" not in probe.stdout, "OpenCL backend initialized instead of native GPU"
print("B5 e4-4 PASS")
