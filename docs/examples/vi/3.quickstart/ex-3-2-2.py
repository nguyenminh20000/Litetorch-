import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

model = lt.nn.Sequential([
    lt.nn.Linear(64, 128, True),
    lt.nn.ReLU(),
    lt.nn.Dropout(0.1),
    lt.nn.Linear(128, 10, True)
])

model.to(device)
