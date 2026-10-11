import litetorch as lt


class Data:
    def __init__(self, x=None, edge_index=None, y=None, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.edge_attr = edge_attr

    @property
    def num_nodes(self):
        return self.x.shape[0] if self.x is not None else 0

    @property
    def num_edges(self):
        return self.edge_index.shape[1] if self.edge_index is not None else 0

    def to(self, device):
        if self.x is not None:
            self.x = self.x.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self


class Batch(Data):
    def __init__(self, x=None, edge_index=None, y=None, batch=None):
        super().__init__(x=x, edge_index=edge_index, y=y)
        self.batch = batch

    @property
    def num_graphs(self):
        if self.batch is None:
            return 1
        return int(max(self.batch.to_vector())) + 1

    @staticmethod
    def from_data_list(data_list):
        xs = [d.x for d in data_list]
        x = lt.Ops.cat(xs, 0)
        ei_flat = []
        batch_vec = []
        ys = []
        off = 0
        total_e = 0
        for i, d in enumerate(data_list):
            ev = [int(v) for v in d.edge_index.to_vector()]
            ne = d.num_edges
            ei_flat.extend([v + off for v in ev])
            total_e += ne
            batch_vec.extend([float(i)] * d.num_nodes)
            if d.y is not None:
                ys.append(d.y)
            off += d.num_nodes
        n2 = len(ei_flat) // 2
        edge_index = lt.Tensor.from_vector([float(v) for v in ei_flat], [2, n2])
        batch = lt.Tensor.from_vector(batch_vec, [len(batch_vec)])
        y = lt.Ops.cat(ys, 0) if ys else None
        return Batch(x=x, edge_index=edge_index, y=y, batch=batch)

    def to_data_list(self):
        bv = [int(v) for v in self.batch.to_vector()]
        ev = [int(v) for v in self.edge_index.to_vector()]
        ne = self.num_edges
        out = []
        for g in range(self.num_graphs):
            nodes = [i for i, b in enumerate(bv) if b == g]
            idx = {n: k for k, n in enumerate(nodes)}
            xs = [self.x.to_vector()[n * self.x.shape[1]:(n + 1) * self.x.shape[1]] for n in nodes]
            flat = []
            for row in xs:
                flat.extend(row)
            x = lt.Tensor.from_vector(flat, [len(nodes), self.x.shape[1]])
            eu, ew = [], []
            for e in range(ne):
                u, v = ev[e], ev[ne + e]
                if bv[u] == g and bv[v] == g:
                    eu.append(float(idx[u]))
                    ew.append(float(idx[v]))
            edge_index = lt.Tensor.from_vector(eu + ew, [2, len(eu)])
            out.append(Data(x=x, edge_index=edge_index))
        return out
