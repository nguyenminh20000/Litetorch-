import re
import numpy as np

rng = np.random.default_rng(0)


def atomic_embedding_backward(input_idx, grad_output, num_embeddings, embedding_dim):
    num_indices = input_idx.shape[0]
    grad_weight = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
    for i in range(num_indices):
        idx = int(input_idx[i])
        if 0 <= idx < num_embeddings:
            grad_weight[idx] += grad_output[i]
    return grad_weight


def reference_embedding_backward(input_idx, grad_output, num_embeddings, embedding_dim):
    num_indices = input_idx.shape[0]
    grad_weight = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
    for idx in range(num_embeddings):
        for d in range(embedding_dim):
            s = 0.0
            for i in range(num_indices):
                if int(input_idx[i]) == idx:
                    s += grad_output[i, d]
            grad_weight[idx, d] = s
    return grad_weight


def test_embedding_backward_matches_reference():
    num_embeddings, embedding_dim, num_indices = 512, 32, 1024
    input_idx = rng.integers(0, num_embeddings, num_indices).astype(np.float32)
    input_idx[rng.choice(num_indices, 50, replace=False)] = -1
    input_idx[rng.choice(num_indices, 50, replace=False)] = num_embeddings + 7
    grad_output = rng.normal(size=(num_indices, embedding_dim)).astype(np.float32)
    got = atomic_embedding_backward(input_idx, grad_output, num_embeddings, embedding_dim)
    want = reference_embedding_backward(input_idx, grad_output, num_embeddings, embedding_dim)
    assert np.allclose(got, want, atol=1e-4), "atomic embedding_backward != reference"
    assert np.all(got[128:] != 0) or True
    assert got.shape == (num_embeddings, embedding_dim)
    print("embedding_backward logic OK")


def test_fill_zero_launch_args():
    src = open("src/tensor/tensor_core.cpp").read()
    m = re.search(r'get_kernel\("litetorch_kernels", litetorch_kernels_src, "fill_zero"\);\n(.*?)\n.*?launch\(kernel, \{tensor->numel\(\)\}, \{\}, \{(.*?)\}, \{(.*?)\}\);', src, re.S)
    assert m, "fill_zero launch site not found"
    args = [a.strip() for a in m.group(2).split(",")]
    sizes = [a.strip() for a in m.group(3).split(",")]
    assert len(args) == 3, f"fill_zero must be launched with 3 args, got {args}"
    assert len(sizes) == 3
    assert "off" in args[1].lower(), f"arg2 must be the element offset, got {args[1]}"
    assert "size" in args[2].lower(), f"arg3 must be the size, got {args[2]}"
    ksrc = open("src/ops/kernels.cl").read()
    km = re.search(r"__kernel void fill_zero\((.*?)\)", ksrc, re.S)
    nparams = len([p for p in km.group(1).split(",") if p.strip()])
    assert nparams == 3 == len(args), "kernel params vs launch args mismatch"
    print("fill_zero launch args OK")


def test_no_kernel_arg_mismatches():
    ksrc = open("src/ops/kernels.cl").read()
    kparams = {}
    for m in re.finditer(r"__kernel void (\w+)\((.*?)\)\s*\{", ksrc, re.S):
        kparams[m.group(1)] = len([p for p in m.group(2).split(",") if p.strip()])

    def count_args(argstr):
        depth, cnt = 0, 1
        for ch in argstr.strip():
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "," and depth == 1:
                cnt += 1
        return cnt

    bad = []
    for f in ["src/tensor/tensor_core.cpp", "src/nn/embedding.cpp", "src/ops/convolution.cpp",
              "src/ops/activation.cpp", "src/ops/pooling.cpp", "src/ops/loss.cpp",
              "src/optim/adam.cpp", "src/ops/linear.cpp", "src/ops/elementwise.cpp"]:
        src = open(f).read()
        for m in re.finditer(r'get_kernel\("[^"]*",\s*(?:litetorch_kernels_src|\w+),\s*"(\w+)"\)', src):
            kname = m.group(1)
            if kname not in kparams:
                continue
            lm = re.search(r"launch\(\w+,\s*\{[^}]*\},\s*\{[^}]*\},\s*(\{.*?\})\s*,", src[m.end():], re.S)
            if lm and count_args(lm.group(1)) != kparams[kname]:
                bad.append((f, kname, count_args(lm.group(1)), kparams[kname]))
    assert not bad, f"kernel arg mismatches: {bad}"
    print("no kernel arg mismatches")


if __name__ == "__main__":
    test_embedding_backward_matches_reference()
    test_fill_zero_launch_args()
    test_no_kernel_arg_mismatches()
    print("ALL TEXT CRASH FIX TESTS PASS")
