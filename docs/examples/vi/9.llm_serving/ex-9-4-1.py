import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

draft_model = lt.nn.Sequential([
    lt.nn.Linear(256, 256, True),
    lt.nn.GELU(),
    lt.nn.Linear(256, 32000, False)
])
draft_model.to(device)

target_model = lt.nn.Sequential([
    lt.nn.Linear(256, 1024, True),
    lt.nn.GELU(),
    lt.nn.Linear(1024, 32000, False)
])
target_model.to(device)

engine = lt.nn.SpeculativeEngine(draft_model, target_model)
generated = engine.generate(prompt=[1, 45, 99], max_new_tokens=50, lookahead=4)
