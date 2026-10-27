import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

class CustomBlock(lt.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = lt.nn.Linear(dim, dim, True)
        self.norm = lt.nn.LayerNorm([dim], 1e-5)
        self.act = lt.nn.GELU()

    def forward(self, x):
        h = self.fc.forward(x)
        h = self.act.forward(h)
        return self.norm.forward(h)

    def parameters(self):
        params = []
        params.extend(self.fc.parameters())
        params.extend(self.norm.parameters())
        return params

    def to(self, device):
        self.fc.to(device)
        self.norm.to(device)
