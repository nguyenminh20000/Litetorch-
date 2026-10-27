import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

params = model.parameters()
opt_8bit = lt.optim.AdamW8bit(params, lr=2e-4, weight_decay=0.01)
opt_fp8 = lt.optim.AdamWFP8(params, lr=2e-4, weight_decay=0.01)
