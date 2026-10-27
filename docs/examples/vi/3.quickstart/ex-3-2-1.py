import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

class MultiLayerPerceptron(lt.nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = lt.nn.Linear(in_features, hidden_features, True)
        self.relu = lt.nn.ReLU()
        self.fc2 = lt.nn.Linear(hidden_features, out_features, True)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.relu.forward(h)
        out = self.fc2.forward(h)
        return out

    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def to(self, device):
        self.fc1.to(device)
        self.fc2.to(device)
