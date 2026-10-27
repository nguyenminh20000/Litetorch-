import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

classifier = lt.nn.Sequential([
    lt.nn.Linear(784, 256, True),
    lt.nn.GELU(),
    lt.nn.Dropout(0.2),
    lt.nn.Linear(256, 10, True)
])

classifier.to(device)
