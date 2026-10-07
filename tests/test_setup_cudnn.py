import ast
import glob
import os
import sys
import tempfile

SETUP_PY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py")

_WANTED = {
    "_cudnn_link_files",
    "_cudnn_major",
    "_cudnn_layout",
    "_cudnn_site_roots",
    "_CUDNN_SITE_PATTERNS",
    "find_cudnn",
}


def load_cudnn_helpers():
    with open(SETUP_PY, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.Assign))
        and getattr(n, "name", None) in _WANTED
    ]
    ns = {"os": os, "glob": glob}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), SETUP_PY, "exec"), ns)
    return ns


def make_fake_cudnn(root, major="9"):
    inc = os.path.join(root, "include")
    lib = os.path.join(root, "lib")
    os.makedirs(inc, exist_ok=True)
    os.makedirs(lib, exist_ok=True)
    with open(os.path.join(inc, "cudnn.h"), "w") as f:
        f.write("#define CUDNN_MAJOR %s\n" % major)
    open(os.path.join(lib, "libcudnn.so.%s" % major), "w").close()
    open(os.path.join(lib, "libcudnn.so"), "w").close()
    return root


def with_env(**kwargs):
    saved = {}
    for key, value in kwargs.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return saved


def restore_env(saved):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_env_root_detected():
    ns = load_cudnn_helpers()
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_cudnn(tmp)
        saved = with_env(LITETORCH_CUDNN_ROOT=tmp, LITETORCH_USE_CUDNN=None)
        try:
            info = ns["find_cudnn"]()
        finally:
            restore_env(saved)
        assert info is not None, "find_cudnn() returned None with valid LITETORCH_CUDNN_ROOT"
        assert info["include_dir"] == os.path.join(tmp, "include"), info
        assert info["lib_dir"] == os.path.join(tmp, "lib"), info
        assert info["major"] == "9", info
        assert info["link_files"], info
        assert any("libcudnn" in p for p in info["link_files"]), info
    print("PASS test_env_root_detected")


def test_env_root_invalid_falls_through():
    ns = load_cudnn_helpers()
    ns["_CUDNN_SITE_PATTERNS"] = ()
    saved = with_env(
        LITETORCH_CUDNN_ROOT="/nonexistent/cudnn",
        LITETORCH_USE_CUDNN=None,
        CUDA_HOME=None,
        CUDA_PATH=None,
    )
    try:
        info = ns["find_cudnn"]()
    finally:
        restore_env(saved)
    assert info is None, "expected None when LITETORCH_CUDNN_ROOT is invalid and no fallback, got %r" % (info,)
    print("PASS test_env_root_invalid_falls_through")


def test_site_glob_fallback_without_import():
    ns = load_cudnn_helpers()
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "fakepy", "python3.13", "dist-packages", "nvidia", "cudnn")
        make_fake_cudnn(fake, major="8")
        ns["_CUDNN_SITE_PATTERNS"] = (
            os.path.join(tmp, "fakepy", "python*", "dist-packages", "nvidia", "cudnn"),
        )
        saved = with_env(LITETORCH_CUDNN_ROOT=None, LITETORCH_USE_CUDNN=None)
        try:
            info = ns["find_cudnn"]()
        finally:
            restore_env(saved)
        assert info is not None, "site-glob fallback did not find the fake cuDNN layout"
        assert info["include_dir"] == os.path.join(fake, "include"), info
        assert info["lib_dir"] == os.path.join(fake, "lib"), info
        assert info["major"] == "8", info
    print("PASS test_site_glob_fallback_without_import")


def test_opt_out_returns_none():
    ns = load_cudnn_helpers()
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_cudnn(tmp)
        saved = with_env(LITETORCH_USE_CUDNN="0", LITETORCH_CUDNN_ROOT=tmp)
        try:
            info = ns["find_cudnn"]()
        finally:
            restore_env(saved)
        assert info is None, "LITETORCH_USE_CUDNN=0 should disable detection, got %r" % (info,)
    print("PASS test_opt_out_returns_none")


if __name__ == "__main__":
    test_env_root_detected()
    test_env_root_invalid_falls_through()
    test_site_glob_fallback_without_import()
    test_opt_out_returns_none()
    print("ALL TESTS PASSED")
