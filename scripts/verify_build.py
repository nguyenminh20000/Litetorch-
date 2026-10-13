#!/usr/bin/env python3
import os
import subprocess
import sys


def fail(msg):
    sys.stderr.write("[verify_build] FAIL: %s\n" % msg)
    sys.stderr.flush()
    sys.exit(1)


def git_head():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15, cwd=repo,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return None


def gpu_lib_candidates(mod_dir):
    lib = "liblitetorch_gpu.dll" if sys.platform.startswith("win") else "liblitetorch_gpu.so"
    paths = [
        os.path.join(mod_dir, lib),
        os.path.join(os.getcwd(), "build", lib),
        os.path.join(os.getcwd(), lib),
        os.path.join("/tmp", lib),
        os.path.join("/usr/local/lib", lib),
        os.path.join("/usr/lib", lib),
        os.path.join("/opt/rocm/lib", lib),
        os.path.join("/opt/rocm/lib64", lib),
    ]
    temp_env = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp_env:
        paths.append(os.path.join(temp_env, lib))
    return paths


def main():
    head = git_head()
    if not head:
        fail("could not determine git HEAD; run from the repo checkout")
    try:
        import litetorch as lt
    except Exception as e:
        fail("could not import litetorch: %s" % e)
    get_commit = getattr(lt, "build_commit", None)
    if get_commit is None:
        fail("litetorch.build_commit() is missing; the module was built without the build-commit guard, rebuild it")
    built = get_commit()
    if built != head:
        fail("stale build: module was built from commit %s but repo HEAD is %s; rm -rf build/ and rebuild" % (built, head))
    sys.stdout.write("[verify_build] build commit OK: %s\n" % built)
    if not os.path.exists(lt.__file__) or os.path.getsize(lt.__file__) == 0:
        fail("litetorch module file missing or empty: %s" % lt.__file__)
    sys.stdout.write("[verify_build] module file OK: %s\n" % lt.__file__)
    if os.environ.get("LITETORCH_CUDA") == "1" or os.environ.get("LITETORCH_ROCM") == "1":
        mod_dir = os.path.dirname(os.path.abspath(lt.__file__))
        found = None
        for p in gpu_lib_candidates(mod_dir):
            if os.path.exists(p) and os.path.getsize(p) > 0:
                found = p
                break
        if not found:
            fail("GPU build requested but liblitetorch_gpu was not produced; the nvcc/hipcc build failed or was skipped silently")
        sys.stdout.write("[verify_build] native GPU lib OK: %s\n" % found)
    sys.stdout.write("[verify_build] ALL OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
