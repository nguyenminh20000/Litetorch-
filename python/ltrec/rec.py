import random

import litetorch as lt
from litetorch import nn


def _full(shape, value):
    n = 1
    for s in shape:
        n *= s
    return lt.Tensor.from_vector([float(value)] * n, list(shape))


def _rowsum(x):
    n = x.shape[0]
    d = x.shape[1]
    ones = lt.Tensor.ones([n, 1])
    return lt.Ops.matmul(ones.transpose(0, 1), x).reshape([d])


def _colsum(x):
    b = x.shape[0]
    k = x.shape[1]
    ones = lt.Tensor.ones([k, 1])
    return lt.Ops.matmul(x, ones).reshape([b])


def _index_tensor(bag):
    data = [float(i) for i in bag]
    dt = getattr(lt.DataType, "INT64", None)
    if dt is not None:
        return lt.Tensor.from_vector(data, [len(bag)], lt.Device("cpu"), False, dt)
    return lt.Tensor.from_vector(data, [len(bag)])


class EmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mode="sum"):
        super().__init__()
        assert mode in ("sum", "mean", "max"), mode
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.emb = nn.Embedding(num_embeddings, embedding_dim)

    def _pool(self, x):
        n = x.shape[0]
        d = x.shape[1]
        if n == 0:
            return lt.Tensor.zeros([d])
        ones = lt.Tensor.ones([n, 1])
        s = lt.Ops.matmul(ones.transpose(0, 1), x).reshape([d])
        if self.mode == "sum":
            return s
        if self.mode == "mean":
            return lt.Ops.mul(s, _full([d], 1.0 / n))
        v = x.to_vector()
        arg = [max(range(d), key=lambda j: v[i * d + j]) for i in range(n)]
        mask = [0.0] * (n * d)
        for i, j in enumerate(arg):
            mask[i * d + j] = 1.0
        m = lt.Tensor.from_vector(mask, [n, d])
        return _rowsum(lt.Ops.mul(m, x))

    def forward(self, input, offsets):
        idx = [int(v) for v in input.to_vector()]
        off = [int(v) for v in offsets.to_vector()] + [len(idx)]
        outs = []
        for b in range(len(off) - 1):
            bag = idx[off[b]:off[b + 1]]
            if len(bag) == 0:
                outs.append(lt.Tensor.zeros([self.embedding_dim]))
                continue
            t = _index_tensor(bag)
            outs.append(self._pool(self.emb.forward(t)))
        return lt.Ops.cat([o.reshape([1, self.embedding_dim]) for o in outs], 0)

    def parameters(self):
        return self.emb.parameters()

    def to(self, device):
        self.emb.to(device)


class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, factor_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.v = nn.Linear(input_dim, factor_dim, False)

    def forward(self, x):
        b = x.shape[0]
        lin = self.linear.forward(x)
        xv = self.v.forward(x)
        s1 = _colsum(xv)
        s1 = lt.Ops.mul(s1, s1)
        x2 = lt.Ops.pow(x, 2.0)
        s2 = _colsum(self.v.forward(x2))
        inter = lt.Ops.mul(lt.Ops.sub(s1, s2), _full([b], 0.5))
        return lt.Ops.add(lin, inter.reshape([b, 1]))

    def parameters(self):
        return self.linear.parameters() + self.v.parameters()

    def to(self, device):
        self.linear.to(device)
        self.v.to(device)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for m in self.layers:
            x = m.forward(x)
        return x

    def parameters(self):
        ps = []
        for m in self.layers:
            ps = ps + m.parameters()
        return ps

    def to(self, device):
        for m in self.layers:
            m.to(device)


class DLRM(nn.Module):
    def __init__(self, num_embeddings_list, embedding_dim, dense_dim,
                 bottom_dims=(), top_dims=()):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.embs = [nn.Embedding(n, embedding_dim) for n in num_embeddings_list]
        bottom = [dense_dim] + list(bottom_dims) + [embedding_dim]
        self.bottom_mlp = MLP(bottom) if len(bottom) > 1 else None
        n_feat = 1 + len(num_embeddings_list)
        n_pair = n_feat * (n_feat - 1) // 2
        top = [embedding_dim + n_pair] + list(top_dims) + [1]
        self.top_mlp = MLP(top)

    def _pairwise_dots(self, feats):
        dots = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                dots.append(_colsum(lt.Ops.mul(feats[i], feats[j])).reshape([feats[i].shape[0], 1]))
        return lt.Ops.cat(dots, 1)

    def forward(self, dense_x, sparse_inputs):
        feats = []
        if self.bottom_mlp is not None:
            d = self.bottom_mlp.forward(dense_x)
        else:
            if dense_x.shape[1] != self.embedding_dim:
                raise ValueError(
                    "DLRM: bottom_mlp is None so dense_x dim (%d) must equal embedding_dim (%d)"
                    % (dense_x.shape[1], self.embedding_dim))
            d = dense_x
        feats.append(d)
        for emb, inp in zip(self.embs, sparse_inputs):
            feats.append(emb.forward(inp))
        inter = self._pairwise_dots(feats)
        x = lt.Ops.cat([d] + [inter], 1)
        return self.top_mlp.forward(x)

    def parameters(self):
        ps = []
        for e in self.embs:
            ps = ps + e.parameters()
        if self.bottom_mlp is not None:
            ps = ps + self.bottom_mlp.parameters()
        return ps + self.top_mlp.parameters()

    def to(self, device):
        for e in self.embs:
            e.to(device)
        if self.bottom_mlp is not None:
            self.bottom_mlp.to(device)
        self.top_mlp.to(device)


def negative_sampling(edge_index, num_nodes, num_neg_samples=None, force_undirected=False):
    ev = [int(v) for v in edge_index.to_vector()]
    n_edge = len(ev) // 2
    existing = set()
    for e in range(n_edge):
        u, v = ev[e], ev[n_edge + e]
        existing.add((u, v))
        if force_undirected:
            existing.add((v, u))
    if num_neg_samples is None:
        num_neg_samples = n_edge
    neg = []
    tries = 0
    while len(neg) < num_neg_samples and tries < num_neg_samples * 20 + 100:
        tries += 1
        u = random.randrange(num_nodes)
        v = random.randrange(num_nodes)
        if u == v or (u, v) in existing:
            continue
        existing.add((u, v))
        neg.append((u, v))
    flat = []
    for u, v in neg:
        flat.append(float(u))
    for u, v in neg:
        flat.append(float(v))
    return lt.Tensor.from_vector(flat, [2, len(neg)])
