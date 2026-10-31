import sys

import litetorch as lt
import torch

CPU = lt.Device("cpu")


def vec(t):
    return t.to(CPU).to_vector()


def values_match(a, b):
    av, bv = vec(a), vec(b)
    if len(av) != len(bv):
        return False
    return all(abs(x - y) < 1e-6 for x, y in zip(av, bv))


class MLP(lt.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = lt.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


class ParamMod(lt.nn.Module):
    def __init__(self):
        super().__init__()
        with lt.no_grad():
            self.p = lt.nn.Parameter(lt.Tensor.randn([2], CPU, False))

    def forward(self, x):
        return x


class BufMod(lt.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = lt.nn.Buffer(lt.Tensor.randn([2], CPU))

    def forward(self, x):
        return x


class PlainMod(lt.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = lt.Tensor.randn([2], CPU)
        self.n = 5
        self.s = "x"

    def forward(self, x):
        return x


class Outer(lt.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = MLP()

    def forward(self, x):
        return self.inner(x)


def test_attr_assign_registers_linear():
    m = MLP()
    if len(m.parameters()) != 2:
        return False, "parameters() == %d, expected 2" % len(m.parameters())
    keys = sorted(m.state_dict().keys())
    if keys != ["linear.bias", "linear.weight"]:
        return False, "state_dict keys %s" % keys
    x = lt.Tensor.randn([2, 4], CPU)
    if not values_match(m(x), m.linear(x)):
        return False, "forward does not use registered linear"
    return True, ""


def test_attr_assign_registers_parameter():
    m = ParamMod()
    if len(m.parameters()) != 1:
        return False, "parameters() == %d, expected 1" % len(m.parameters())
    if sorted(m.state_dict().keys()) != ["p"]:
        return False, "state_dict keys %s" % sorted(m.state_dict().keys())
    if m.p is not m.parameters()[0]:
        return False, "attribute read does not return registered parameter"
    return True, ""


def test_attr_assign_registers_buffer():
    m = BufMod()
    if len(m.parameters()) != 0:
        return False, "parameters() == %d, expected 0" % len(m.parameters())
    if len(m.buffers()) != 1:
        return False, "buffers() == %d, expected 1" % len(m.buffers())
    if sorted(m.state_dict().keys()) != ["b"]:
        return False, "state_dict keys %s" % sorted(m.state_dict().keys())
    return True, ""


def test_plain_tensor_not_registered():
    m = PlainMod()
    if len(m.parameters()) != 0 or len(m.buffers()) != 0:
        return False, "plain tensor registered as param/buffer"
    if len(m.state_dict()) != 0:
        return False, "state_dict not empty: %s" % sorted(m.state_dict().keys())
    if m.t is None or m.n != 5 or m.s != "x":
        return False, "plain attributes not readable"
    return True, ""


def test_reassign_replaces():
    m = MLP()
    old = m.linear
    m.linear = lt.nn.Linear(4, 3)
    if len(m.parameters()) != 2:
        return False, "parameters() == %d after reassign" % len(m.parameters())
    if m.linear is old:
        return False, "reassign did not replace the module"
    if sorted(m.state_dict().keys()) != ["linear.bias", "linear.weight"]:
        return False, "state_dict keys changed after reassign"
    return True, ""


def test_assign_none_keeps_readable():
    m = MLP()
    m.linear = None
    if m.linear is not None:
        return False, "m.linear is not None after None assign"
    if not hasattr(m, "linear"):
        return False, "hasattr False after None assign"
    if len(m.parameters()) != 0:
        return False, "parameters() == %d after None assign" % len(m.parameters())
    if len(m.state_dict()) != 0:
        return False, "state_dict not empty after None assign"
    m.linear = lt.nn.Linear(4, 3)
    if len(m.parameters()) != 2:
        return False, "parameters() == %d after reassign from None" % len(m.parameters())
    p = ParamMod()
    p.p = None
    if p.p is not None or len(p.parameters()) != 0 or len(p.state_dict()) != 0:
        return False, "parameter None assign not readable-as-None with empty registries"
    b = BufMod()
    b.b = None
    if b.b is not None or len(b.buffers()) != 0 or len(b.state_dict()) != 0:
        return False, "buffer None assign not readable-as-None with empty registries"
    m.linear = None
    del m.linear
    if hasattr(m, "linear"):
        return False, "hasattr True after del of None-assigned name"
    try:
        m.linear
    except AttributeError:
        return True, ""
    return False, "reading deleted attribute did not raise AttributeError"


def test_delattr_removes():
    m = MLP()
    del m.linear
    if len(m.parameters()) != 0 or len(m.state_dict()) != 0:
        return False, "del did not remove the module"
    try:
        m.linear
    except AttributeError:
        return True, ""
    return False, "reading deleted attribute did not raise AttributeError"


class Empty(lt.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test_add_module_still_works():
    m = Empty()
    m.add_module("linear", lt.nn.Linear(4, 3))
    if len(m.parameters()) != 2:
        return False, "add_module parameters() == %d" % len(m.parameters())
    if m.linear is None:
        return False, "add_module child not readable as attribute"
    old = m.linear
    m.linear = lt.nn.Linear(4, 3)
    if len(m.parameters()) != 2 or m.linear is old:
        return False, "attribute assign did not replace add_module child"
    return True, ""


def test_nested_submodule():
    m = Outer()
    if len(m.parameters()) != 2:
        return False, "parameters() == %d, expected 2" % len(m.parameters())
    keys = sorted(m.state_dict().keys())
    if keys != ["inner.linear.bias", "inner.linear.weight"]:
        return False, "state_dict keys %s" % keys
    return True, ""


def test_load_state_dict_roundtrip():
    m1, m2 = MLP(), MLP()
    m2.load_state_dict(m1.state_dict())
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if not values_match(p1, p2):
            return False, "load_state_dict values do not match source"
    return True, ""


def test_to_keeps_registered():
    m = MLP()
    m.to(CPU)
    if len(m.parameters()) != 2:
        return False, "parameters() == %d after to()" % len(m.parameters())
    if sorted(m.state_dict().keys()) != ["linear.bias", "linear.weight"]:
        return False, "state_dict keys changed after to()"
    return True, ""


def test_weight_assign_preserved():
    lin = lt.nn.Linear(4, 3)
    w = lt.Tensor.from_vector([1.0] * 12, [3, 4], CPU)
    b = lt.Tensor.from_vector([0.5] * 3, [3], CPU)
    lin.weight = w
    lin.bias = b
    x = lt.Tensor.from_vector([1.0, 0.0, 0.0, 0.0], [1, 4], CPU)
    if not values_match(lin(x), lt.Tensor.from_vector([1.5, 1.5, 1.5], [1, 3], CPU)):
        return False, "plain-tensor weight/bias assign no longer drives forward"
    return True, ""


def test_wrong_type_raises():
    m = MLP()
    try:
        m.linear = 5
    except TypeError:
        return True, ""
    except Exception as e:
        return False, "wrong exception: %s: %s" % (type(e).__name__, e)
    return False, "assigning int to registered module name did not raise TypeError"


def test_torch_parity_keys():
    class TMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.linear(x)

    tm = TMLP()
    if sorted(tm.state_dict().keys()) != sorted(MLP().state_dict().keys()):
        return False, "state_dict keys differ from torch"
    return True, ""


TESTS = [
    ("attr_assign_registers_linear", test_attr_assign_registers_linear),
    ("attr_assign_registers_parameter", test_attr_assign_registers_parameter),
    ("attr_assign_registers_buffer", test_attr_assign_registers_buffer),
    ("plain_tensor_not_registered", test_plain_tensor_not_registered),
    ("reassign_replaces", test_reassign_replaces),
    ("assign_none_keeps_readable", test_assign_none_keeps_readable),
    ("delattr_removes", test_delattr_removes),
    ("add_module_still_works", test_add_module_still_works),
    ("nested_submodule", test_nested_submodule),
    ("load_state_dict_roundtrip", test_load_state_dict_roundtrip),
    ("to_keeps_registered", test_to_keeps_registered),
    ("weight_assign_preserved", test_weight_assign_preserved),
    ("wrong_type_raises", test_wrong_type_raises),
    ("torch_parity_keys", test_torch_parity_keys),
]


def main():
    failures = 0
    for name, fn in TESTS:
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, "unexpected exception: %s: %s" % (type(e).__name__, e)
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else " :: " + msg))
        if not ok:
            failures += 1
    print("%d/%d passed" % (len(TESTS) - failures, len(TESTS)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
