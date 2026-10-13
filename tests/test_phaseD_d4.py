import os
import shutil
import tempfile
import litetorch as lt
from litetorch import Ops

def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond

def make_model():
    m = lt.nn.Sequential()
    m.add_module("l1", lt.nn.Linear(4, 8, True))
    m.add_module("l2", lt.nn.Linear(8, 2, True))
    return m

def param_snapshot(model):
    return {k: v.clone() for k, v in model.state_dict()}

def maxdiff(a, b):
    return Ops.abs(a - b).max().item()

def test_save_load_roundtrip(tmp):
    ok = True
    cfg = lt.TrainerConfig()
    cfg.ckpt_dir = os.path.join(tmp, "ckpts")
    cfg.async_save = False
    model = make_model()
    opt = lt.optim.AdamW(model.parameters(), lr=0.01)
    tr = lt.Trainer(model, opt, cfg)
    ref = param_snapshot(model)
    tr.save_checkpoint(10)
    path = os.path.join(cfg.ckpt_dir, "step-10.safetensors")
    ok &= check("save: file written", os.path.exists(path))
    for k, v in model.state_dict():
        lt.nn.init.fill_(v, 0.0)
    step = tr.load_latest()
    ok &= check("load: returns step", step == 10, f"step={step}")
    for k, v in model.state_dict():
        ok &= check(f"load: {k} restored", maxdiff(v, ref[k]) == 0.0)
    return ok

def test_load_empty(tmp):
    ok = True
    cfg = lt.TrainerConfig()
    cfg.ckpt_dir = os.path.join(tmp, "empty")
    cfg.async_save = False
    model = make_model()
    opt = lt.optim.AdamW(model.parameters(), lr=0.01)
    tr = lt.Trainer(model, opt, cfg)
    ok &= check("load empty: returns -1", tr.load_latest() == -1)
    ok &= check("latest_step empty: -1", tr.latest_step() == -1)
    return ok

def test_garbage_collect(tmp):
    ok = True
    cfg = lt.TrainerConfig()
    cfg.ckpt_dir = os.path.join(tmp, "gc")
    cfg.async_save = False
    cfg.max_checkpoints = 3
    model = make_model()
    opt = lt.optim.AdamW(model.parameters(), lr=0.01)
    tr = lt.Trainer(model, opt, cfg)
    for s in [1, 2, 3, 4, 5]:
        tr.save_checkpoint(s)
    tr.wait_pending_saves()
    files = sorted(f for f in os.listdir(cfg.ckpt_dir) if f.startswith("step-"))
    ok &= check("gc: keeps 3", len(files) == 3, f"files={files}")
    ok &= check("gc: keeps newest", files == ["step-3.safetensors", "step-4.safetensors", "step-5.safetensors"], f"files={files}")
    ok &= check("latest_step: 5", tr.latest_step() == 5)
    return ok

def test_train_step(tmp):
    ok = True
    cfg = lt.TrainerConfig()
    cfg.ckpt_dir = os.path.join(tmp, "train")
    cfg.async_save = False
    model = make_model()
    opt = lt.optim.AdamW(model.parameters(), lr=0.1)
    tr = lt.Trainer(model, opt, cfg)
    x = lt.randn([16, 4])
    before = param_snapshot(model)
    for _ in range(5):
        opt.zero_grad()
        out = model(x)
        loss = Ops.sum(out * out)
        loss.backward()
        tr.train_step()
    changed = any(maxdiff(v, before[k]) > 0 for k, v in model.state_dict())
    ok &= check("train_step: params updated", changed)
    tr.save_checkpoint(1)
    ok &= check("train: checkpoint saved", tr.latest_step() == 1)
    return ok

def test_async_save(tmp):
    ok = True
    cfg = lt.TrainerConfig()
    cfg.ckpt_dir = os.path.join(tmp, "async")
    cfg.async_save = True
    model = make_model()
    opt = lt.optim.AdamW(model.parameters(), lr=0.01)
    tr = lt.Trainer(model, opt, cfg)
    ref = param_snapshot(model)
    tr.save_checkpoint(7)
    tr.wait_pending_saves()
    path = os.path.join(cfg.ckpt_dir, "step-7.safetensors")
    ok &= check("async: file written", os.path.exists(path))
    loaded = lt.load_safetensors(path)
    ok &= check("async: keys match", set(loaded.keys()) == set(ref.keys()))
    return ok

def test_rank_worldsize(tmp):
    ok = True
    cfg = lt.TrainerConfig()
    cfg.ckpt_dir = os.path.join(tmp, "rw")
    model = make_model()
    opt = lt.optim.AdamW(model.parameters(), lr=0.01)
    tr = lt.Trainer(model, opt, cfg)
    ok &= check("single: not distributed", not tr.is_distributed())
    ok &= check("single: rank 0", tr.rank() == 0)
    ok &= check("single: world_size 1", tr.world_size() == 1)
    return ok

def main():
    tmp = tempfile.mkdtemp(prefix="lt_d4_")
    ok = True
    try:
        ok &= test_save_load_roundtrip(tmp)
        ok &= test_load_empty(tmp)
        ok &= test_garbage_collect(tmp)
        ok &= test_train_step(tmp)
        ok &= test_async_save(tmp)
        ok &= test_rank_worldsize(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
