import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import litetorch as lt

CPU = lt.Device("cpu")
PASS = 0
FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")


def tvec(t):
    return t.to(CPU).to_vector()


def make_ds(n=20, d=4):
    xs = lt.Tensor.from_vector([float(i * d + j) for i in range(n) for j in range(d)], [n, d], CPU)
    ys = lt.Tensor.from_vector([float(i % 2) for i in range(n)], [n], CPU)
    return lt.data.TensorDataset(xs, ys)


def batch_rows(bx):
    v = tvec(bx)
    d = list(bx.shape)[1]
    return [tuple(v[i * d:(i + 1) * d]) for i in range(list(bx.shape)[0])]


def all_rows(dl):
    rows = []
    for bx, by in dl:
        rows.extend(batch_rows(bx))
    return rows


# ---------------- AMP autocast ----------------
check("autocast_disabled_by_default", not lt.amp.is_autocast_enabled())

with lt.amp.autocast("cpu"):
    check("autocast_cpu_enabled", lt.amp.is_autocast_enabled())
    check("autocast_cpu_default_bf16", lt.amp.get_autocast_dtype() == lt.DataType.BF16)
check("autocast_exit_restores", not lt.amp.is_autocast_enabled())

with lt.amp.autocast("cpu", dtype=lt.DataType.FP16):
    check("autocast_fp16", lt.amp.get_autocast_dtype() == lt.DataType.FP16)
check("autocast_exit_restores_2", not lt.amp.is_autocast_enabled())

with lt.amp.autocast("cpu", dtype="float16"):
    check("autocast_str_dtype", lt.amp.get_autocast_dtype() == lt.DataType.FP16)

with lt.amp.autocast("cpu", dtype=lt.DataType.FP16):
    with lt.amp.autocast("cpu", dtype=lt.DataType.BF16):
        check("autocast_nested_inner", lt.amp.get_autocast_dtype() == lt.DataType.BF16)
    check("autocast_nested_restore", lt.amp.get_autocast_dtype() == lt.DataType.FP16)
check("autocast_nested_outer_exit", not lt.amp.is_autocast_enabled())

with lt.amp.autocast("cpu", enabled=False):
    check("autocast_disabled_flag", not lt.amp.is_autocast_enabled())

try:
    with lt.amp.autocast("tpu_device"):
        pass
    check("autocast_bad_device_raises", False, "no exception")
except Exception:
    check("autocast_bad_device_raises", True)

if not lt.is_gpu_available():
    try:
        with lt.amp.autocast("cuda"):
            pass
        check("autocast_cuda_fail_loud", False, "no exception on CPU-only build")
    except RuntimeError:
        check("autocast_cuda_fail_loud", True)
    except Exception as e:
        check("autocast_cuda_fail_loud", False, f"wrong exc: {type(e)}")

# ---------------- grad mode context managers ----------------
check("grad_enabled_default", lt.is_grad_enabled())
with lt.no_grad():
    check("no_grad_disables", not lt.is_grad_enabled())
    with lt.enable_grad():
        check("enable_grad_inside_no_grad", lt.is_grad_enabled())
    check("enable_grad_exit_restores", not lt.is_grad_enabled())
check("no_grad_exit_restores", lt.is_grad_enabled())

with lt.no_grad():
    with lt.enable_grad():
        pass
    check("enable_grad_nested_restore", not lt.is_grad_enabled())

with lt.inference_mode():
    check("inference_mode_no_grad", not lt.is_grad_enabled())
    check("inference_mode_flag", lt.is_inference_mode())
check("inference_mode_exit_grad", lt.is_grad_enabled())
check("inference_mode_exit_flag", not lt.is_inference_mode())

with lt.inference_mode():
    with lt.inference_mode():
        check("inference_mode_nested", lt.is_inference_mode() and not lt.is_grad_enabled())
    check("inference_mode_nested_restore", lt.is_inference_mode())
check("inference_mode_outer_exit", not lt.is_inference_mode() and lt.is_grad_enabled())

# ---------------- Dataset / Subset / random_split ----------------
ds = make_ds()
check("dataset_size", ds.size() == 20 and len(ds) == 20)
x0, y0 = ds.get(0)
check("dataset_get_shapes", list(x0.shape) == [4] and list(y0.shape) == [1], f"{x0.shape} {y0.shape}")
g3 = ds[3]
check("dataset_getitem", isinstance(g3, tuple) and len(g3) == 2)

sub = lt.data.Subset(ds, [0, 5, 10])
check("subset_size", sub.size() == 3 and len(sub) == 3)
sx, _ = sub.get(1)
check("subset_get_maps", tvec(sx) == tvec(ds.get(5)[0]))

try:
    lt.data.Subset(ds, [0, 999])
    check("subset_oob_raises", False, "no exception")
except Exception:
    check("subset_oob_raises", True)

splits = lt.data.random_split(ds, [12, 8], seed=42)
check("random_split_sizes", splits[0].size() == 12 and splits[1].size() == 8)
seen = set()
for s in splits:
    for i in range(s.size()):
        seen.add(tuple(tvec(s.get(i)[0])))
check("random_split_disjoint_cover", len(seen) == 20, f"got {len(seen)}")

try:
    lt.data.random_split(ds, [10, 5])
    check("random_split_bad_sum_raises", False, "no exception")
except Exception:
    check("random_split_bad_sum_raises", True)

# ---------------- Samplers ----------------
seq = lt.data.SequentialSampler(10)
check("seq_sampler", seq.indices() == list(range(10)) and len(seq) == 10)

rs = lt.data.RandomSampler(10, seed=123)
idx = rs.indices()
check("rand_sampler_perm", sorted(idx) == list(range(10)) and len(rs) == 10)
rs2 = lt.data.RandomSampler(10, seed=123)
check("rand_sampler_deterministic", rs2.indices() == idx)

rsr = lt.data.RandomSampler(10, replacement=True, num_samples=25, seed=7)
idxr = rsr.indices()
check("rand_sampler_replacement", len(idxr) == 25 and all(0 <= i < 10 for i in idxr) and len(rsr) == 25)

try:
    lt.data.RandomSampler(10, num_samples=11)
    check("rand_sampler_oob_raises", False, "no exception")
except Exception:
    check("rand_sampler_oob_raises", True)

bs = lt.data.BatchSampler(lt.data.SequentialSampler(10), 4, drop_last=False)
batches = bs.batches()
check("batch_sampler", batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]] and len(bs) == 3, str(batches))
bs2 = lt.data.BatchSampler(lt.data.SequentialSampler(10), 4, drop_last=True)
check("batch_sampler_drop_last", bs2.batches() == [[0, 1, 2, 3], [4, 5, 6, 7]] and len(bs2) == 2)

# ---------------- DataLoader ----------------
dl = lt.data.DataLoader(ds, 5, shuffle=False)
check("dataloader_len", len(dl) == 4 and dl.num_batches() == 4)
rows = all_rows(dl)
expected = [tuple(float(i * 4 + j) for j in range(4)) for i in range(20)]
check("dataloader_ordered", rows == expected, f"first rows {rows[:2]}")

count = sum(1 for _ in dl)
check("dataloader_iter_twice", count == 4)

it = iter(dl)
next(it)
check("dataloader_iter_resets", sum(1 for _ in dl) == 4)

dl_s = lt.data.DataLoader(ds, 5, shuffle=True)
srows = all_rows(dl_s)
check("dataloader_shuffle_covers", sorted(srows) == sorted(expected))

ds22 = make_ds(22)
dl_d = lt.data.DataLoader(ds22, 5, shuffle=False, drop_last=True)
check("dataloader_drop_last", len(dl_d) == 4 and sum(1 for _ in dl_d) == 4)
dl_nd = lt.data.DataLoader(ds22, 5, shuffle=False, drop_last=False)
check("dataloader_no_drop_last", len(dl_nd) == 5 and sum(1 for _ in dl_nd) == 5)

dl_w = lt.data.DataLoader(ds, 5, shuffle=False, num_workers=2)
wrows = all_rows(dl_w)
check("dataloader_num_workers", wrows == expected, "worker results differ")

dl_w2 = lt.data.DataLoader(ds, 6, shuffle=True, num_workers=3)
srows2 = sorted(all_rows(dl_w2))
exp22 = sorted([tuple(float(i * 4 + j) for j in range(4)) for i in range(20)])
check("dataloader_workers_shuffle", srows2 == exp22)

calls = []


def my_collate(batch):
    calls.append(len(batch))
    xs = lt.Ops.cat([lt.Ops.unsqueeze(x, 0) for x, y in batch], 0)
    ys = lt.Ops.cat([lt.Ops.unsqueeze(y, 0) for x, y in batch], 0)
    return xs, ys


dl_c = lt.data.DataLoader(ds, 5, shuffle=False, collate_fn=my_collate)
crows = all_rows(dl_c)
check("dataloader_collate_fn", crows == expected and calls == [5, 5, 5, 5], str(calls))


class PyDS(lt.data.Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def size(self):
        return self.n

    def get(self, i):
        x = lt.Tensor.from_vector([float(i), float(i) + 0.5], [2], CPU)
        y = lt.Tensor.from_vector([float(i % 3)], [1], CPU)
        return x, y


pyds = PyDS(10)
dl_p = lt.data.DataLoader(pyds, 4, shuffle=False)
prows = []
for bx, by in dl_p:
    v = tvec(bx)
    prows.extend([(v[i * 2], v[i * 2 + 1]) for i in range(4 if len(prows) < 8 else 2)])
check("dataloader_python_dataset", prows == [(float(i), float(i) + 0.5) for i in range(10)], str(prows[:3]))

dl_pw = lt.data.DataLoader(pyds, 4, shuffle=False, num_workers=2)
vals = []
for bx, by in lt.data.DataLoader(pyds, 4, shuffle=False, num_workers=2):
    v = tvec(bx)
    vals.extend([v[i * 2] for i in range(list(bx.shape)[0])])
check("dataloader_python_ds_workers", sorted(vals) == [float(i) for i in range(10)], str(sorted(vals)))

dl_samp = lt.data.DataLoader(ds, 5, shuffle=False, sampler=lt.data.SequentialSampler(20))
check("dataloader_sampler", all_rows(dl_samp) == expected)

dl_bs = lt.data.DataLoader(ds, 5, shuffle=False,
                           sampler=lt.data.BatchSampler(lt.data.SequentialSampler(20), 5))
check("dataloader_batch_sampler", all_rows(dl_bs) == expected)

try:
    lt.data.DataLoader(ds, 5, shuffle=True, sampler=lt.data.SequentialSampler(20))
    check("dataloader_sampler_shuffle_conflict", False, "no exception")
except Exception:
    check("dataloader_sampler_shuffle_conflict", True)

try:
    lt.data.DataLoader(ds, 0)
    check("dataloader_zero_batch_raises", False, "no exception")
except Exception:
    check("dataloader_zero_batch_raises", True)

# save/load state roundtrip
dl_a = lt.data.DataLoader(ds, 5, shuffle=False)
first = dl_a.next()
path = "/tmp/dl_state_test.bin"
dl_a.save_state(path)
dl_b = lt.data.DataLoader(ds, 5, shuffle=False)
dl_b.load_state(path)
rest_a = []
while True:
    b = dl_a.next()
    if b is None:
        break
    rest_a.append(batch_rows(b[0]))
rest_b = []
while True:
    b = dl_b.next()
    if b is None:
        break
    rest_b.append(batch_rows(b[0]))
check("dataloader_save_load_state", rest_a == rest_b and len(rest_a) == 3, f"{len(rest_a)} vs {len(rest_b)}")

# legacy next() API still works (test_core.py compat)
dl_leg = lt.data.DataLoader(ds, 5, True, CPU)
nb = 0
while True:
    b = dl_leg.next()
    if b is None:
        break
    bx, by = b
    nb += 1
    if list(bx.shape) != [5, 4] or list(by.shape) != [5]:
        check("dataloader_legacy_shapes", False, f"{bx.shape} {by.shape}")
        break
else:
    pass
check("dataloader_legacy_count", nb == 4)
check("dataloader_legacy_shapes", True)

print(f"\nPASS {PASS} FAIL {FAIL}")
for f in FAILURES:
    print("FAIL:", f)
sys.exit(1 if FAIL else 0)
