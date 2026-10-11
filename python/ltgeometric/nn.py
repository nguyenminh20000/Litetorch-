import numpy as np

import litetorch as lt
from litetorch import nn


def _edge_list(edge_index):
    ev = [int(v) for v in edge_index.to_vector()]
    ne = len(ev) // 2
    return [(ev[e], ev[ne + e]) for e in range(ne)]


def _adjacency(edge_index, num_nodes, self_loops=True, dtype=np.float32):
    a = np.zeros((num_nodes, num_nodes), dtype=dtype)
    for u, v in _edge_list(edge_index):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            a[u, v] = 1.0
    if self_loops:
        np.fill_diagonal(a, 1.0)
    return a


def _to_tensor(mat):
    n = mat.shape[0]
    return lt.Tensor.from_vector(mat.reshape(-1).tolist(), [n, n])


def _gcn_norm(a):
    deg = a.sum(axis=1)
    d = np.power(deg, -0.5, where=deg > 0)
    d[~np.isfinite(d)] = 0.0
    return (d[:, None] * a * d[None, :])


def _mean_norm(a):
    deg = a.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return a / deg


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index):
        n = x.shape[0]
        ah = _to_tensor(_gcn_norm(_adjacency(edge_index, n, True)).astype(np.float32))
        return self.lin.forward(lt.Ops.matmul(ah, x))

    def parameters(self):
        return self.lin.parameters()

    def to(self, device):
        self.lin.to(device)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_root = nn.Linear(in_channels, out_channels, bias)
        self.lin_msg = nn.Linear(in_channels, out_channels, False)

    def forward(self, x, edge_index):
        n = x.shape[0]
        an = _to_tensor(_mean_norm(_adjacency(edge_index, n, True)).astype(np.float32))
        agg = lt.Ops.matmul(an, x)
        return lt.Ops.add(self.lin_root.forward(x), self.lin_msg.forward(agg))

    def parameters(self):
        return self.lin_root.parameters() + self.lin_msg.parameters()

    def to(self, device):
        self.lin_root.to(device)
        self.lin_msg.to(device)


class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias)
        self.lin_r = nn.Linear(in_channels, out_channels, False)

    def forward(self, x, edge_index):
        n = x.shape[0]
        an = _to_tensor(_mean_norm(_adjacency(edge_index, n, True)).astype(np.float32))
        agg = lt.Ops.matmul(an, x)
        return lt.Ops.add(self.lin_l.forward(x), self.lin_r.forward(agg))

    def parameters(self):
        return self.lin_l.parameters() + self.lin_r.parameters()

    def to(self, device):
        self.lin_l.to(device)
        self.lin_r.to(device)


class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.lins = [nn.Linear(in_channels, out_channels, False) for _ in range(heads)]
        self.scorers = [nn.Linear(2 * out_channels, 1, False) for _ in range(heads)]
        self.leaky = nn.LeakyReLU()

    def forward(self, x, edge_index):
        n = x.shape[0]
        edges = _edge_list(edge_index)
        outs = []
        for h in range(self.heads):
            wh = self.lins[h].forward(x)
            outs.append(self._head_forward(wh, edges, n, h))
        if self.concat:
            return lt.Ops.cat(outs, 1)
        avg = outs[0]
        for o in outs[1:]:
            avg = lt.Ops.add(avg, o)
        return lt.Ops.mul(avg, lt.Tensor.from_vector([1.0 / self.heads], [1]))

    def _head_forward(self, wh, edges, n, h):
        a = _adjacency_from_edges(edges, n, True)
        big = lt.Tensor.from_vector([-1e9] * (n * n), [n, n])
        mask = _to_tensor((1.0 - a).astype(np.float32))
        s = lt.Tensor.zeros([n, n])
        for u, v in edges:
            mu = lt.Tensor.from_vector([1.0 if i == u else 0.0 for i in range(n)], [1, n])
            mv = lt.Tensor.from_vector([1.0 if i == v else 0.0 for i in range(n)], [1, n])
            hu = lt.Ops.matmul(mu, wh).reshape([self.out_channels])
            hv = lt.Ops.matmul(mv, wh).reshape([self.out_channels])
            pair = lt.Ops.cat([hu.reshape([1, self.out_channels]),
                               hv.reshape([1, self.out_channels])], 1)
            score = self.leaky.forward(self.scorers[h].forward(pair))
            me = lt.Tensor.from_vector(
                [1.0 if (i == u and j == v) else 0.0 for i in range(n) for j in range(n)],
                [n, n])
            s = lt.Ops.add(s, lt.Ops.mul(me, score))
        s = lt.Ops.add(s, lt.Ops.mul(mask, big))
        alpha = lt.Ops.softmax(s, 1)
        return lt.Ops.matmul(alpha, wh)

    def parameters(self):
        ps = []
        for m in self.lins:
            ps = ps + m.parameters()
        for m in self.scorers:
            ps = ps + m.parameters()
        return ps

    def to(self, device):
        for m in self.lins:
            m.to(device)
        for m in self.scorers:
            m.to(device)


def _adjacency_from_edges(edges, num_nodes, self_loops=True):
    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            a[u, v] = 1.0
    if self_loops:
        np.fill_diagonal(a, 1.0)
    return a
