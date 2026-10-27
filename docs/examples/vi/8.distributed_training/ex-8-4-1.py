import sys
sys.path.insert(0, '/home/hatch/workspace/framework/integration/python')
import litetorch as lt

device = lt.auto_device()

class TensorParallelMLP(lt.nn.Module):
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.col_proj = lt.nn.ColumnParallelLinear(hidden_dim, ffn_dim, True)
        self.gelu = lt.nn.GELU()
        self.row_proj = lt.nn.RowParallelLinear(ffn_dim, hidden_dim, True)

    def forward(self, x):
        h = self.col_proj.forward(x)
        h = self.gelu.forward(h)
        out = self.row_proj.forward(h)
        return out
