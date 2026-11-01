import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import litetorch as lt
import torch
import torch.nn as tnn

cpu = lt.Device("cpu")
VALS = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
TOL = 1e-4
GELU_TOL = 5e-4
fails = []

def lt_t(vals, shape):
    n = 1
    for s in shape:
        n *= s
    vals = (list(vals) * ((n // len(vals)) + 1))[:n]
    return lt.Tensor.from_vector([float(v) for v in vals], shape, cpu)

def th_t(vals, shape):
    n = 1
    for s in shape:
        n *= s
    vals = (list(vals) * ((n // len(vals)) + 1))[:n]
    return torch.tensor(vals, dtype=torch.float32).reshape(shape)

def lt_vec(t):
    return torch.tensor(t.to_vector(), dtype=torch.float32)

def md(a, b):
    return (a - b).abs().max().item()

def copy_params(lt_params, th_params):
    with lt.no_grad():
        for lp, tp in zip(lt_params, th_params):
            tp.data = lt_vec(lp).reshape(tuple(lp.shape))

def build_pair(activation, batch_first, lt_kwargs=None):
    kw = dict(lt_kwargs or {})
    ld = lt.nn.TransformerDecoderLayer(8, 2, 2048, activation, batch_first, **kw)
    td = tnn.TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=2048,
                                     dropout=0.0, batch_first=batch_first,
                                     activation=activation)
    tps = [td.self_attn.in_proj_weight, td.self_attn.in_proj_bias,
           td.self_attn.out_proj.weight, td.self_attn.out_proj.bias,
           td.multihead_attn.in_proj_weight, td.multihead_attn.in_proj_bias,
           td.multihead_attn.out_proj.weight, td.multihead_attn.out_proj.bias,
           td.linear1.weight, td.linear1.bias, td.linear2.weight, td.linear2.bias,
           td.norm1.weight, td.norm1.bias, td.norm2.weight, td.norm2.bias,
           td.norm3.weight, td.norm3.bias]
    copy_params(list(ld.parameters()), tps)
    ld.eval()
    td.eval()
    return ld, td

def check(name, d, tol):
    ok = d < tol
    print(f"{'PASS' if ok else 'FAIL'} {name}: maxdiff={d:.3e} (tol {tol:.0e})", flush=True)
    if not ok:
        fails.append(name)

tgt = lt_t(VALS, [2, 2, 8])
mem = lt_t(VALS, [2, 2, 8])
ttgt = th_t(VALS, [2, 2, 8])
tmem = th_t(VALS, [2, 2, 8])

ld, td = build_pair("relu", False)
check("A docs two-arg defaults relu/bf=False [2,2,8]",
      md(lt_vec(ld.forward(tgt, mem)), td(ttgt, tmem).detach().reshape(-1)), TOL)

torch.manual_seed(0)
rtgt = torch.randn(3, 2, 8)
rmem = torch.randn(3, 2, 8)
lt_rtgt = lt.Tensor.from_vector(rtgt.reshape(-1).tolist(), [3, 2, 8], cpu)
lt_rmem = lt.Tensor.from_vector(rmem.reshape(-1).tolist(), [3, 2, 8], cpu)

ld, td = build_pair("relu", False)
check("B nonsym [3,2,8] relu/bf=False",
      md(lt_vec(ld.forward(lt_rtgt, lt_rmem)), td(rtgt, rmem).detach().reshape(-1)), TOL)

ld, td = build_pair("gelu", False)
check("C nonsym [3,2,8] gelu/bf=False (tanh-approx residual)",
      md(lt_vec(ld.forward(lt_rtgt, lt_rmem)), td(rtgt, rmem).detach().reshape(-1)), GELU_TOL)

ld, td = build_pair("gelu", True)
check("D nonsym [3,2,8] gelu/bf=True (tanh-approx residual)",
      md(lt_vec(ld.forward(lt_rtgt, lt_rmem)), td(rtgt, rmem).detach().reshape(-1)), GELU_TOL)

ld, td = build_pair("relu", True)
check("E nonsym [3,2,8] relu/bf=True",
      md(lt_vec(ld.forward(lt_rtgt, lt_rmem)), td(rtgt, rmem).detach().reshape(-1)), TOL)

ld, td = build_pair("relu", False)
x1 = td.norm1(rtgt + td.self_attn(rtgt, rtgt, rtgt)[0].detach())
nomem_th = td.norm3(x1 + td.linear2(td.activation(td.linear1(x1))))
check("F one-arg vs manual torch no-memory decoder relu/bf=False",
      md(lt_vec(ld(lt_rtgt)), nomem_th.reshape(-1)), TOL)

try:
    lt.nn.TransformerDecoderLayer(8, 2, 2048, "silu", False)
    print("FAIL G invalid activation accepted", flush=True)
    fails.append("G invalid activation")
except RuntimeError as e:
    ok = "activation must be relu or gelu" in str(e)
    print(f"{'PASS' if ok else 'FAIL'} G invalid activation raises: {e}", flush=True)
    if not ok:
        fails.append("G invalid activation")

ld_kw = lt.nn.TransformerDecoderLayer(8, 2, dim_feedforward=2048, activation="gelu", batch_first=True)
ld_def = lt.nn.TransformerDecoderLayer(8, 2)
ld_pos = lt.nn.TransformerDecoderLayer(8, 2, 32, "gelu", True)
print("PASS H kwargs + defaults + positional construction", flush=True)

le = lt.nn.TransformerEncoderLayer(8, 2, 2048, 0.0, "relu", False)
te = tnn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=2048, dropout=0.0,
                                 activation="relu", batch_first=False)
tps = [te.self_attn.in_proj_weight, te.self_attn.in_proj_bias,
       te.self_attn.out_proj.weight, te.self_attn.out_proj.bias,
       te.linear1.weight, te.linear1.bias, te.linear2.weight, te.linear2.bias,
       te.norm1.weight, te.norm1.bias, te.norm2.weight, te.norm2.bias]
copy_params(list(le.parameters()), tps)
le.eval()
te.eval()
check("I encoder regression relu/bf=False [3,2,8]",
      md(lt_vec(le.forward(lt_rtgt)), te(rtgt).detach().reshape(-1)), TOL)

print(f"RESULT: {'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}", flush=True)
sys.exit(0 if not fails else 1)
