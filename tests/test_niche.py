import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import litetorch as lt
from litetorch import nn

lt.manual_seed(0)

print("=== ltrec ===")
from ltrec import EmbeddingBag, FactorizationMachine, DLRM, negative_sampling

for mode in ("sum", "mean", "max"):
    eb = EmbeddingBag(10, 8, mode=mode)
    idx = lt.Tensor.from_vector([1, 2, 3, 4, 5], [5])
    off = lt.Tensor.from_vector([0, 2, 5], [3])
    out = eb.forward(idx, off)
    assert list(out.shape) == [3, 8], (mode, out.shape)
    lt.Ops.sum(out).backward()
    assert eb.parameters()[0].grad is not None, mode
    print("EmbeddingBag", mode, "OK", list(out.shape))

fm = FactorizationMachine(6, 4)
x = lt.randn([5, 6])
y = fm.forward(x)
assert list(y.shape) == [5, 1], y.shape
lt.Ops.sum(y).backward()
assert all(p.grad is not None for p in fm.parameters())
print("FactorizationMachine OK", list(y.shape))

dlrm = DLRM([8, 6], 4, 5, bottom_dims=[8], top_dims=[8])
dense = lt.randn([3, 5])
s1 = lt.Tensor.from_vector([1, 3, 5], [3])
s2 = lt.Tensor.from_vector([2, 4, 0], [3])
logit = dlrm.forward(dense, [s1, s2])
assert list(logit.shape) == [3, 1], logit.shape
lt.Ops.sum(logit).backward()
print("DLRM OK", list(logit.shape))

ei = lt.Tensor.from_vector([0, 1, 2, 1, 2, 0], [2, 3])
neg = negative_sampling(ei, 4, num_neg_samples=4)
assert list(neg.shape) == [2, 4], neg.shape
nv = [int(v) for v in neg.to_vector()]
pos = {(0, 1), (1, 2), (2, 0)}
for e in range(4):
    assert (nv[e], nv[4 + e]) not in pos, "negative collided with positive"
print("negative_sampling OK", list(neg.shape))

print("=== ltgeometric ===")
from ltgeometric import Data, Batch, GCNConv, GraphConv, SAGEConv, GATConv

ev = [0, 0, 1, 2, 1, 2, 2, 3, 3, 0]
edge_index = lt.Tensor.from_vector([float(v) for v in ev], [2, 5])
data = Data(x=lt.randn([4, 6]), edge_index=edge_index)
assert data.num_nodes == 4 and data.num_edges == 5
print("Data OK")

gcn = GCNConv(6, 8)
o = gcn.forward(data.x, data.edge_index)
assert list(o.shape) == [4, 8], o.shape
lt.Ops.sum(o).backward()
assert all(p.grad is not None for p in gcn.parameters())
print("GCNConv OK", list(o.shape))

gc = GraphConv(6, 8)
o = gc.forward(data.x, data.edge_index)
assert list(o.shape) == [4, 8], o.shape
lt.Ops.sum(o).backward()
print("GraphConv OK", list(o.shape))

sage = SAGEConv(6, 8)
o = sage.forward(data.x, data.edge_index)
assert list(o.shape) == [4, 8], o.shape
lt.Ops.sum(o).backward()
assert all(p.grad is not None for p in sage.parameters())
print("SAGEConv OK", list(o.shape))

gat = GATConv(6, 8, heads=2)
o = gat.forward(data.x, data.edge_index)
assert list(o.shape) == [4, 16], o.shape
lt.Ops.sum(o).backward()
assert all(p.grad is not None for p in gat.parameters())
print("GATConv OK", list(o.shape))

gat1 = GATConv(6, 8, heads=1, concat=False)
o = gat1.forward(data.x, data.edge_index)
assert list(o.shape) == [4, 8], o.shape
print("GATConv mean-heads OK", list(o.shape))

d2 = Data(x=lt.randn([3, 6]), edge_index=lt.Tensor.from_vector([0.0, 1.0, 1.0, 2.0], [2, 2]))
b = Batch.from_data_list([data, d2])
assert b.num_nodes == 7 and b.num_graphs == 2, (b.num_nodes, b.num_graphs)
assert list(b.batch.to_vector()) == [0.0] * 4 + [1.0] * 3
back = b.to_data_list()
assert len(back) == 2 and back[0].num_nodes == 4 and back[1].num_nodes == 3
print("Batch OK", b.num_nodes, "nodes", b.num_graphs, "graphs")

print("=== ltrrl ===")
from ltrrl import BanditEnv, GridWorldEnv, ReplayBuffer, DQN, PPO, REINFORCE, compute_returns

env = BanditEnv([0.2, 0.8])
assert env.n_actions == 2 and env.observation_dim == 1
o, r, d, _ = env.step(1)
assert d and r in (0.0, 1.0)
gw = GridWorldEnv(3)
assert gw.n_actions == 4 and gw.observation_dim == 9
o = gw.reset()
assert o[0] == 1.0
o, r, d, _ = gw.step(1)
assert o[3] == 1.0 and not d
print("Envs OK")

rb = ReplayBuffer(100)
for i in range(20):
    rb.push([0.1 * i], i % 2, 1.0, [0.1 * (i + 1)], False)
assert len(rb) == 20
s = rb.sample(8)
assert list(s["obs"].shape) == [8, 1]
print("ReplayBuffer OK")

dqn = DQN(1, 2, hidden=16, lr=1e-2)
a = dqn.act([0.5], epsilon=0.0)
assert a in (0, 1)
losses = []
for _ in range(5):
    losses.append(dqn.update(rb.sample(8)))
assert all(v == v and abs(v) < 1e9 for v in losses), losses
dqn.sync_target()
print("DQN OK losses:", ["%.4f" % v for v in losses])

ppo = PPO(9, 4, hidden=16, lr=1e-2)
a, lp = ppo.act(gw.reset())
assert 0 <= a < 4 and lp < 0
obs_l = [gw.reset() for _ in range(6)]
acts = [ppo.act(o)[0] for o in obs_l]
old = [-0.5] * 6
adv = [0.3, -0.2, 0.1, 0.5, -0.1, 0.2]
rets = [1.0, 0.0, 0.5, 1.0, 0.2, 0.8]
pl = ppo.update(obs_l, acts, old, adv, rets)
assert pl == pl and abs(pl) < 1e9, pl
print("PPO OK loss: %.4f" % pl)

re = REINFORCE(9, 4, hidden=16, lr=1e-2)
a = re.act(gw.reset())
assert 0 <= a < 4
rl = re.update(obs_l, acts, compute_returns([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
assert rl == rl and abs(rl) < 1e9, rl
print("REINFORCE OK loss: %.4f" % rl)

print("=== ltmobile ===")
from ltmobile import (
    QuantizedLinear, LiteSequential, quantize_linear, quantize_module,
    quantization_error, export_litemodel, LiteModule,
)

lin = nn.Linear(8, 4)
x = lt.randn([3, 8])
q = quantize_linear(lin)
err = quantization_error(lin, q, x)
assert err < 0.05, err
print("QuantizedLinear OK max_err=%.5f storage=%dB (fp32 would be %dB)"
      % (err, q.storage_bytes(), 8 * 4 * 4 + 16))

seq = LiteSequential([nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)])
xq = quantize_module(seq)
assert isinstance(xq.layers[0], QuantizedLinear)
assert isinstance(xq.layers[1], nn.ReLU)
o1 = seq.forward(x).to_vector()
o2 = xq.forward(x).to_vector()
assert max(abs(a - b) for a, b in zip(o1, o2)) < 0.1
print("quantize_module OK")

tmp = tempfile.mkdtemp()
export_litemodel(xq, os.path.join(tmp, "m"))
lm = LiteModule(os.path.join(tmp, "m"))
assert lm.num_layers() == 3
o3 = lm.forward(x).to_vector()
assert max(abs(a - b) for a, b in zip(o2, o3)) < 1e-5, "export roundtrip mismatch"
bench = lm.benchmark(x, iters=20, warmup=2)
assert bench["ms_per_iter"] > 0
print("export/LiteModule OK storage=%dB bench=%.3fms/iter"
      % (lm.storage_bytes(), bench["ms_per_iter"]))

print()
print("=== ALL NICHE TESTS PASSED ===")
