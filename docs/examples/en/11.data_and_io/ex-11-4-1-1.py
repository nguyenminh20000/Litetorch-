import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

model = lt.nn.Sequential([
    lt.nn.Linear(32, 64, True),
    lt.nn.ReLU(),
    lt.nn.Linear(64, 2, True)
])

lt.save(model, "model.lt")
lt.load(model, "model.lt")
