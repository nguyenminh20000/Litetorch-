import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import litetorch as lt
import torch

cpu = lt.Device("cpu")
fails = 0
checks = 0


def check(cond, label):
    global fails, checks
    checks += 1
    if not cond:
        fails += 1
        print("FAIL: %s" % label)


def f32(vals, shape):
    n = 1
    for s in shape:
        n *= s
    vals = (list(vals) * ((n // len(vals)) + 1))[:n]
    return lt.Tensor.from_vector([float(v) for v in vals], shape, cpu)


def i32(vals, shape=None):
    if shape is None:
        shape = [len(vals)]
    t = lt.tensor(list(vals), dtype=lt.DataType.INT32)
    return t.reshape(shape) if hasattr(t, "reshape") else t


def i64(vals, shape=None):
    return lt.tensor(list(vals), dtype=lt.DataType.INT64)


def vals(t):
    return t.to_vector()


def maxdiff(a, b):
    va, vb = vals(a), vals(b)
    assert len(va) == len(vb), "shape mismatch %d vs %d" % (len(va), len(vb))
    return max(abs(x - y) for x, y in zip(va, vb)) if va else 0.0


def same(a, b):
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return maxdiff(a, b) == 0.0


def throws(exc, fn):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def case(name, make, args, kwargs=None, one_arg=None):
    m = make()
    check(same(m(*args), m.forward(*args)), "%s call==forward" % name)
    if kwargs is not None:
        check(same(m(**kwargs), m.forward(*args)), "%s kwargs==forward" % name)
    if one_arg is not None:
        check(same(m(one_arg), m.forward(one_arg)), "%s 1-arg call==forward" % name)


def torch_parity(name, lt_val, torch_val, tol=1e-5):
    v = vals(lt_val)[0]
    check(abs(v - torch_val) <= tol, "%s torch parity lt=%r torch=%r" % (name, v, torch_val))


V = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2]
A22 = lambda: f32(V, [2, 2])
B22 = lambda: f32([0.2, 0.1, -0.3, 0.4], [2, 2])

case("MSELoss", lt.nn.MSELoss, [A22(), B22()], {"input": A22(), "target": B22()})
torch_parity("MSELoss", lt.nn.MSELoss()(A22(), B22()),
             torch.nn.MSELoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]])).item())
case("BCELoss", lt.nn.BCELoss, [f32([0.2, 0.8, 0.4, 0.6], [2, 2]), f32([0.0, 1.0, 1.0, 0.0], [2, 2])],
     {"input": f32([0.2, 0.8, 0.4, 0.6], [2, 2]), "target": f32([0.0, 1.0, 1.0, 0.0], [2, 2])})
torch_parity("BCELoss", lt.nn.BCELoss()(f32([0.2, 0.8, 0.4, 0.6], [2, 2]), f32([0.0, 1.0, 1.0, 0.0], [2, 2])),
             torch.nn.BCELoss()(torch.tensor([[0.2, 0.8], [0.4, 0.6]]), torch.tensor([[0.0, 1.0], [1.0, 0.0]])).item())
case("L1Loss", lt.nn.L1Loss, [A22(), B22()], {"input": A22(), "target": B22()})
torch_parity("L1Loss", lt.nn.L1Loss()(A22(), B22()),
             torch.nn.L1Loss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]])).item())
case("HingeEmbeddingLoss", lt.nn.HingeEmbeddingLoss, [A22(), f32([1.0, -1.0, 1.0, -1.0], [2, 2])],
     {"input": A22(), "target": f32([1.0, -1.0, 1.0, -1.0], [2, 2])})
case("CosineEmbeddingLoss", lt.nn.CosineEmbeddingLoss, [A22(), B22(), f32([1.0, -1.0], [2])],
     {"input1": A22(), "input2": B22(), "target": f32([1.0, -1.0], [2])})
torch_parity("CosineEmbeddingLoss", lt.nn.CosineEmbeddingLoss()(A22(), B22(), f32([1.0, -1.0], [2])),
             torch.nn.CosineEmbeddingLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]]), torch.tensor([1.0, -1.0])).item())
case("MarginRankingLoss", lt.nn.MarginRankingLoss, [A22(), B22(), f32([1.0, -1.0, -1.0, 1.0], [2, 2])],
     {"input1": A22(), "input2": B22(), "target": f32([1.0, -1.0, -1.0, 1.0], [2, 2])})
torch_parity("MarginRankingLoss", lt.nn.MarginRankingLoss()(A22(), B22(), f32([1.0, -1.0, -1.0, 1.0], [2, 2])),
             torch.nn.MarginRankingLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]]), torch.tensor([[1.0, -1.0], [-1.0, 1.0]])).item())
case("MultiMarginLoss", lt.nn.MultiMarginLoss, [f32(V, [2, 4]), i32([1, 3], [2])],
     {"input": f32(V, [2, 4]), "target": i32([1, 3], [2])})
case("MultiLabelMarginLoss", lt.nn.MultiLabelMarginLoss, [f32(V, [2, 4]), i32([1, 0, -1, -1, 2, 1, -1, -1], [2, 4])],
     {"input": f32(V, [2, 4]), "target": i32([1, 0, -1, -1, 2, 1, -1, -1], [2, 4])})
case("SoftMarginLoss", lt.nn.SoftMarginLoss, [A22(), f32([1.0, -1.0, -1.0, 1.0], [2, 2])],
     {"input": A22(), "target": f32([1.0, -1.0, -1.0, 1.0], [2, 2])})
torch_parity("SoftMarginLoss", lt.nn.SoftMarginLoss()(A22(), f32([1.0, -1.0, -1.0, 1.0], [2, 2])),
             torch.nn.SoftMarginLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[1.0, -1.0], [-1.0, 1.0]])).item())
case("MultiLabelSoftMarginLoss", lt.nn.MultiLabelSoftMarginLoss, [A22(), f32([1.0, 0.0, 0.0, 1.0], [2, 2])],
     {"input": A22(), "target": f32([1.0, 0.0, 0.0, 1.0], [2, 2])})
torch_parity("MultiLabelSoftMarginLoss", lt.nn.MultiLabelSoftMarginLoss()(A22(), f32([1.0, 0.0, 0.0, 1.0], [2, 2])),
             torch.nn.MultiLabelSoftMarginLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])).item())
case("TripletMarginLoss", lt.nn.TripletMarginLoss, [A22(), B22(), f32([0.3, 0.2, -0.1, 0.5], [2, 2])],
     {"anchor": A22(), "positive": B22(), "negative": f32([0.3, 0.2, -0.1, 0.5], [2, 2])})
torch_parity("TripletMarginLoss", lt.nn.TripletMarginLoss()(A22(), B22(), f32([0.3, 0.2, -0.1, 0.5], [2, 2])),
             torch.nn.TripletMarginLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]]), torch.tensor([[0.3, 0.2], [-0.1, 0.5]])).item())
case("TripletMarginWithDistanceLoss", lt.nn.TripletMarginWithDistanceLoss, [A22(), B22(), f32([0.3, 0.2, -0.1, 0.5], [2, 2])],
     {"anchor": A22(), "positive": B22(), "negative": f32([0.3, 0.2, -0.1, 0.5], [2, 2])})
case("PoissonNLLLoss", lt.nn.PoissonNLLLoss, [f32([0.1, 0.2, 0.3, 0.4], [2, 2]), f32([1.0, 2.0, 1.0, 0.0], [2, 2])],
     {"input": f32([0.1, 0.2, 0.3, 0.4], [2, 2]), "target": f32([1.0, 2.0, 1.0, 0.0], [2, 2])})
torch_parity("PoissonNLLLoss", lt.nn.PoissonNLLLoss()(f32([0.1, 0.2, 0.3, 0.4], [2, 2]), f32([1.0, 2.0, 1.0, 0.0], [2, 2])),
             torch.nn.PoissonNLLLoss()(torch.tensor([[0.1, 0.2], [0.3, 0.4]]), torch.tensor([[1.0, 2.0], [1.0, 0.0]])).item())
case("GaussianNLLLoss", lt.nn.GaussianNLLLoss, [A22(), B22(), f32([0.5, 0.5, 0.5, 0.5], [2, 2])],
     {"input": A22(), "target": B22(), "var": f32([0.5, 0.5, 0.5, 0.5], [2, 2])})
torch_parity("GaussianNLLLoss", lt.nn.GaussianNLLLoss()(A22(), B22(), f32([0.5, 0.5, 0.5, 0.5], [2, 2])),
             torch.nn.GaussianNLLLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]]), torch.tensor([[0.5, 0.5], [0.5, 0.5]])).item())
case("CrossEntropyLoss", lt.nn.CrossEntropyLoss, [f32(V, [2, 4]), i32([1, 3], [2])],
     {"input": f32(V, [2, 4]), "target": i32([1, 3], [2])})
torch_parity("CrossEntropyLoss", lt.nn.CrossEntropyLoss()(f32(V, [2, 4]), i32([1, 3], [2])),
             torch.nn.CrossEntropyLoss()(torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]]), torch.tensor([1, 3])).item())
case("NLLLoss", lt.nn.NLLLoss, [f32([-0.5, -1.5, -0.2, -2.0, -0.1, -0.9, -1.1, -0.4], [2, 4]), i32([1, 3], [2])],
     {"input": f32([-0.5, -1.5, -0.2, -2.0, -0.1, -0.9, -1.1, -0.4], [2, 4]), "target": i32([1, 3], [2])})
torch_parity("NLLLoss", lt.nn.NLLLoss()(f32([-0.5, -1.5, -0.2, -2.0, -0.1, -0.9, -1.1, -0.4], [2, 4]), i32([1, 3], [2])),
             torch.nn.NLLLoss()(torch.tensor([[-0.5, -1.5, -0.2, -2.0], [-0.1, -0.9, -1.1, -0.4]]), torch.tensor([1, 3])).item())
case("BCEWithLogitsLoss", lt.nn.BCEWithLogitsLoss, [A22(), f32([1.0, 0.0, 0.0, 1.0], [2, 2])],
     {"input": A22(), "target": f32([1.0, 0.0, 0.0, 1.0], [2, 2])})
torch_parity("BCEWithLogitsLoss", lt.nn.BCEWithLogitsLoss()(A22(), f32([1.0, 0.0, 0.0, 1.0], [2, 2])),
             torch.nn.BCEWithLogitsLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]])).item())
case("KLDivLoss", lt.nn.KLDivLoss, [f32([-0.5, -1.0, -0.2, -0.7], [2, 2]), f32([0.3, 0.2, 0.4, 0.1], [2, 2])],
     {"input": f32([-0.5, -1.0, -0.2, -0.7], [2, 2]), "target": f32([0.3, 0.2, 0.4, 0.1], [2, 2])})
torch_parity("KLDivLoss", lt.nn.KLDivLoss()(f32([-0.5, -1.0, -0.2, -0.7], [2, 2]), f32([0.3, 0.2, 0.4, 0.1], [2, 2])),
             torch.nn.KLDivLoss()(torch.tensor([[-0.5, -1.0], [-0.2, -0.7]]), torch.tensor([[0.3, 0.2], [0.4, 0.1]])).item())
case("SmoothL1Loss", lt.nn.SmoothL1Loss, [A22(), B22()], {"input": A22(), "target": B22()})
torch_parity("SmoothL1Loss", lt.nn.SmoothL1Loss()(A22(), B22()),
             torch.nn.SmoothL1Loss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]])).item())
case("HuberLoss", lt.nn.HuberLoss, [A22(), B22()], {"input": A22(), "target": B22()})
torch_parity("HuberLoss", lt.nn.HuberLoss()(A22(), B22()),
             torch.nn.HuberLoss()(torch.tensor([[0.1, -0.2], [0.3, -0.4]]), torch.tensor([[0.2, 0.1], [-0.3, 0.4]])).item())
lp = f32(V, [5, 2, 3])
case("CTCLoss", lt.nn.CTCLoss, [lp, i32([1, 2, 1, 2, 1, 2], [2, 3]), i32([5, 5], [2]), i32([3, 3], [2])],
     {"log_probs": lp, "targets": i32([1, 2, 1, 2, 1, 2], [2, 3]), "input_lengths": i32([5, 5], [2]), "target_lengths": i32([3, 3], [2])})
case("DPOLoss", lt.nn.DPOLoss, [f32([0.5, 0.6], [2]), f32([0.1, 0.2], [2]), f32([0.4, 0.5], [2]), f32([0.2, 0.3], [2])],
     {"policy_chosen_logps": f32([0.5, 0.6], [2]), "policy_rejected_logps": f32([0.1, 0.2], [2]), "ref_chosen_logps": f32([0.4, 0.5], [2]), "ref_rejected_logps": f32([0.2, 0.3], [2])})
case("Bilinear", lambda: lt.nn.Bilinear(4, 4, 4), [f32(V, [2, 4]), f32(V, [2, 4])],
     {"input1": f32(V, [2, 4]), "input2": f32(V, [2, 4])})
Q = lambda: f32(V, [2, 2, 8])
case("MultiHeadAttention", lambda: lt.nn.MultiHeadAttention(8, 2), [Q(), Q(), Q()],
     {"query": Q(), "key": Q(), "value": Q()}, one_arg=Q())
mha = lt.nn.MultiHeadAttention(8, 2)
am = f32([0.0, 0.0, 0.0, 0.0], [2, 2])
check(same(mha(Q(), Q(), Q(), am, None, True), mha.forward(Q(), Q(), Q(), am, None, True)), "MultiHeadAttention 6-arg call==forward")
case("TransformerDecoderLayer", lambda: lt.nn.TransformerDecoderLayer(8, 2), [Q(), Q()],
     {"tgt": Q(), "memory": Q()}, one_arg=Q())
case("TransformerEncoderLayer", lambda: lt.nn.TransformerEncoderLayer(8, 2, 2048, 0.0), [Q(), None, None],
     {"src": Q()}, one_arg=Q())
enc_layer = lt.nn.TransformerEncoderLayer(8, 2, 2048, 0.0)
case("TransformerEncoder", lambda: lt.nn.TransformerEncoder(enc_layer, 2), [Q(), None, None],
     {"src": Q()}, one_arg=Q())
case("EmbeddingBag", lambda: lt.nn.EmbeddingBag(10, 4), [i64([1, 3, 0, 2]), i32([0], [1])],
     {"input": i64([1, 3, 0, 2]), "offsets": i32([0], [1])}, one_arg=i64([1, 3, 0, 2]))
case("LSTMCell", lambda: lt.nn.LSTMCell(4, 4), [f32(V, [2, 4]), f32(V, [2, 4]), f32(V, [2, 4])],
     {"input": f32(V, [2, 4]), "hx": f32(V, [2, 4]), "cx": f32(V, [2, 4])}, one_arg=f32(V, [2, 4]))
case("RNNCell", lambda: lt.nn.RNNCell(4, 4), [f32(V, [2, 4]), f32(V, [2, 4])],
     {"input": f32(V, [2, 4]), "hx": f32(V, [2, 4])}, one_arg=f32(V, [2, 4]))
case("GRUCell", lambda: lt.nn.GRUCell(4, 4), [f32(V, [2, 4]), f32(V, [2, 4])],
     {"input": f32(V, [2, 4]), "hx": f32(V, [2, 4])}, one_arg=f32(V, [2, 4]))
S324 = lambda: f32(V, [3, 2, 4])
H = lambda: f32(V, [1, 2, 4])
rnn = lt.nn.RNN(4, 4)
check(same(rnn(S324(), H()), rnn.forward_full(S324(), H())), "RNN 2-arg call==forward_full")
check(same(rnn(input=S324(), hx=H()), rnn.forward_full(S324(), H())), "RNN kwargs call==forward_full")
check(same(rnn(S324())[0], rnn.forward(S324())), "RNN 1-arg call==forward")
lstm = lt.nn.LSTM(4, 4)
lc, lf = lstm(S324(), H(), H()), lstm.forward_full(S324(), H(), H())
check(same(lc[0], lf[0]) and same(lc[1][0], lf[1]) and same(lc[1][1], lf[2]), "LSTM 3-arg call==forward_full")
check(same(lstm(S324())[0], lstm.forward(S324())), "LSTM 1-arg call==forward")
gru = lt.nn.GRU(4, 4)
check(same(gru(S324(), H()), gru.forward_full(S324(), H())), "GRU 2-arg call==forward_full")
check(same(gru(input=S324(), hx=H()), gru.forward_full(S324(), H())), "GRU kwargs call==forward_full")
check(same(gru(S324())[0], gru.forward(S324())), "GRU 1-arg call==forward")

check(throws(TypeError, lambda: lt.nn.MSELoss()(A22())), "MSELoss 1-arg call raises TypeError (torch-aligned)")
check(throws(TypeError, lambda: lt.nn.CrossEntropyLoss()(f32(V, [2, 4]))), "CrossEntropyLoss 1-arg call raises TypeError")
lin = lt.nn.Linear(4, 4)
X24 = f32(V, [2, 4])
check(same(lin(X24), lin.forward(X24)), "Linear 1-arg call==forward (regression)")
seq = lt.nn.Sequential([lt.nn.Linear(4, 4), lt.nn.ReLU()])
check(same(seq(X24), seq.forward(X24)), "Sequential 1-arg call==forward (regression)")

print("D22-TESTS: %d checks, %d failures" % (checks, fails))
sys.exit(1 if fails else 0)
