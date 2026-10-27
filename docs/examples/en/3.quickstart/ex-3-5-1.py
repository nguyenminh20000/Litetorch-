import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

model_path = "model_checkpoint.lt"
optim_path = "optim_checkpoint.lt"

lt.save(model, model_path)
lt.save(optimizer, optim_path)

lt.load(model, model_path)
lt.load(optimizer, optim_path)
