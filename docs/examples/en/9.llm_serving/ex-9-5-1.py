import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

base_model = lt.nn.Sequential([
    lt.nn.Linear(512, 512, True),
    lt.nn.GELU(),
    lt.nn.Linear(512, 32000, False)
])
base_model.to(device)

head1 = lt.nn.MedusaHead(512, 32000)
head2 = lt.nn.MedusaHead(512, 32000)
head1.to(device)
head2.to(device)

medusa = lt.nn.MedusaEngine(base_model, [head1, head2])
tokens = medusa.generate(prompt=[10, 20, 30], max_new_tokens=40, lookahead=2)
