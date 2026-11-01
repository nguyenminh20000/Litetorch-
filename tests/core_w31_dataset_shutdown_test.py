import os
import subprocess
import sys

import litetorch as lt

CPU = lt.Device(lt.DeviceType.CPU, 0)

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print("PASS: %s" % name)
    else:
        print("FAIL: %s %s" % (name, detail))
        fails += 1


def tvec(t):
    return t.to_vector()


class TorchStyleDataset(lt.data.Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = lt.Tensor.from_vector([float(idx), float(idx) + 0.5], [2], CPU, False)
        y = lt.Tensor.from_vector([float(idx % 3)], [1], CPU, False)
        return x, y


class LegacyDataset(lt.data.Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def size(self):
        return self.n

    def get(self, idx):
        x = lt.Tensor.from_vector([float(idx), float(idx) + 0.5], [2], CPU, False)
        y = lt.Tensor.from_vector([float(idx % 3)], [1], CPU, False)
        return x, y


class BothDataset(lt.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 7

    def __getitem__(self, idx):
        return (
            lt.Tensor.from_vector([float(idx), float(idx) + 0.5], [2], CPU, False),
            lt.Tensor.from_vector([float(idx % 3)], [1], CPU, False),
        )

    def size(self):
        return 3

    def get(self, idx):
        return (
            lt.Tensor.from_vector([float(idx) + 100.0, float(idx) + 100.5], [2], CPU, False),
            lt.Tensor.from_vector([float(idx % 3)], [1], CPU, False),
        )


class RaisingDataset(lt.data.Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def size(self):
        return self.n

    def get(self, idx):
        raise ValueError("boom %d" % idx)


def all_rows(loader):
    rows = []
    for bx, by in loader:
        v = tvec(bx)
        rows.extend(v[i * 2] for i in range(list(bx.shape)[0]))
    return rows


ts = TorchStyleDataset(10)
check("torchstyle_len", len(ts) == 10, "got %s" % len(ts))
check("torchstyle_size_dispatch", ts.size() == 10, "got %s" % ts.size())
g3 = ts[3]
check("torchstyle_getitem", tvec(g3[0]) == [3.0, 3.5] and tvec(g3[1]) == [0.0], str(tvec(g3[0])))
check("torchstyle_loader_rows",
      all_rows(lt.data.DataLoader(ts, 4, shuffle=False)) == [float(i) for i in range(10)])

lg = LegacyDataset(10)
check("legacy_size", lg.size() == 10 and len(lg) == 10)
check("legacy_loader_rows",
      all_rows(lt.data.DataLoader(lg, 4, shuffle=False)) == [float(i) for i in range(10)])

bd = BothDataset()
check("both_len_wins",
      len(bd) == 7 and bd.size() == 3 and lt.data.Dataset.size(bd) == 7,
      "len=%s size=%s cxx_size=%s" % (len(bd), bd.size(), lt.data.Dataset.size(bd)))
check("both_getitem_wins",
      all_rows(lt.data.DataLoader(bd, 3, shuffle=False)) == [float(i) for i in range(7)])

x = lt.Tensor.from_vector([float(i) for i in range(40)], [10, 4], CPU, False)
y = lt.Tensor.from_vector([float(i) for i in range(10)], [10, 1], CPU, False)
tds = lt.data.TensorDataset(x, y)
tds_x = []
tds_y = []
for bx, by in lt.data.DataLoader(tds, 4, shuffle=False):
    tds_x.extend(tvec(bx))
    tds_y.extend(tvec(by))
check("tensor_dataset_loader",
      tds_x == [float(i) for i in range(40)] and tds_y == [float(i) for i in range(10)],
      "x=%s y=%s" % (tds_x, tds_y))

rd = RaisingDataset(10)
try:
    lt.data.DataLoader(rd, 4, shuffle=False).next()
    check("raising_dataset_surfaces_error", False, "no exception")
except RuntimeError as e:
    check("raising_dataset_surfaces_error", "Python exception raised in DataLoader worker" in str(e), str(e)[:100])

LT_DIR = os.path.dirname(os.path.abspath(lt.__file__))
ENV = dict(os.environ)
ENV["PYTHONPATH"] = LT_DIR + os.pathsep + ENV.get("PYTHONPATH", "")

SHUTDOWN_SCRIPT = (
    "import sys; sys.path.insert(0, {lt_dir!r}); import litetorch as lt; "
    "CPU = lt.Device(lt.DeviceType.CPU, 0); "
    "exec(open({cls_file!r}).read()); "
    "ds = CLS(10); loader = lt.data.DataLoader(ds, 2, shuffle=False, device=CPU); "
    "b = loader.next(); print(list(b[0].shape), b[1].to_vector())"
)

CLS_TORCHSTYLE = (
    "class CLS(lt.data.Dataset):\n"
    "    def __init__(self, n):\n"
    "        super().__init__()\n"
    "        self.n = n\n"
    "    def __len__(self):\n"
    "        return self.n\n"
    "    def __getitem__(self, idx):\n"
    "        return (lt.Tensor.from_vector([float(idx)] * 4, [4], CPU, False), "
    "                lt.Tensor.from_vector([float(idx * idx)], [1], CPU, False))\n"
)

CLS_LEGACY = (
    "class CLS(lt.data.Dataset):\n"
    "    def __init__(self, n):\n"
    "        super().__init__()\n"
    "        self.n = n\n"
    "    def size(self):\n"
    "        return self.n\n"
    "    def get(self, idx):\n"
    "        return (lt.Tensor.from_vector([float(idx)] * 4, [4], CPU, False), "
    "                lt.Tensor.from_vector([float(idx * idx)], [1], CPU, False))\n"
)


def run_shutdown(name, cls_src, reps=5):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(cls_src)
        cls_file = f.name
    script = SHUTDOWN_SCRIPT.format(lt_dir=LT_DIR, cls_file=cls_file)
    codes = []
    for _ in range(reps):
        r = subprocess.run([sys.executable, "-c", script], env=ENV,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        codes.append(r.returncode)
    os.unlink(cls_file)
    check(name, all(c == 0 for c in codes), "exit codes %s" % codes)


run_shutdown("shutdown_no_del_legacy", CLS_LEGACY)
run_shutdown("shutdown_no_del_torchstyle", CLS_TORCHSTYLE)

print("FAILURES: %d" % fails)
sys.exit(1 if fails else 0)
