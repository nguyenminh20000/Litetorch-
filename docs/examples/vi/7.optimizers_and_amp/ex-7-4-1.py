import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

if lt.cuda.is_available():
    lt.cuda.set_tf32_enabled(True)
    print("TF32 status:", lt.cuda.is_tf32_enabled())
