import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

NAN_RE = re.compile(r"\b(nan|inf|-inf|\+inf)\b", re.IGNORECASE)
TRAIN_RE = re.compile(r"for\s+epoch|epochs\s*=|num_epochs|mp\.get_context|p\.join\(", re.IGNORECASE)
NON_CPU_DEVICES = {"gpu", "tpu", "arm"}


def run_cmd(cmd, cwd, env, timeout):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.perf_counter() - t0, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return None, out, err, time.perf_counter() - t0, True


def build_tree(tree):
    env = dict(os.environ)
    env["LITETORCH_NO_NATIVE_GPU"] = "1"
    env["LT_BUILD_JOBS"] = "1"
    t0 = time.perf_counter()
    rc, out, err, wall, _ = run_cmd(
        [sys.executable, "setup.py", "build_ext", "--inplace"], tree, env, timeout=7200)
    wall = time.perf_counter() - t0
    return rc, out, err, wall


def assert_extension(tree, ext_dir=None):
    ext_dir = ext_dir or os.path.join(tree, "python")
    so_files = glob.glob(os.path.join(ext_dir, "litetorch*.so"))
    if not so_files:
        return False, "no litetorch*.so under %s" % ext_dir
    code = (
        "import sys; sys.path.insert(0, %r); import litetorch; "
        "print(getattr(litetorch, 'build_commit', lambda: 'n/a')())" % ext_dir
    )
    rc, out, err, _, _ = run_cmd([sys.executable, "-c", code], tree, dict(os.environ), timeout=120)
    if rc != 0:
        return False, "import litetorch failed: %s" % (err.strip().splitlines() or [""])[-1]
    return True, "import ok (%s), commit %s" % (so_files[0], out.strip())


def check_output(code, stdout, stderr):
    failures = []
    if "Traceback" in stderr:
        failures.append("traceback in stderr")
    if "Error" in stderr:
        failures.append("Error in stderr")
    for line in stderr.splitlines():
        if "WARNING" in line and "fail" in line.lower():
            failures.append("failing WARNING in stderr")
            break
    if "nan" not in code.lower() and NAN_RE.search(stdout):
        failures.append("NaN/Inf in stdout")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.path.expanduser("~/workspace/framework/integration"))
    ap.add_argument("--manifest", default=os.path.expanduser("~/workspace/framework/docs/examples/manifest.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--train-timeout", type=int, default=600)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--ext-dir", default=None)
    args = ap.parse_args()
    docs_root = os.path.expanduser("~/workspace/framework/docs")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out or os.path.join(docs_root, "results", "cpu-" + stamp)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.tree, capture_output=True, text=True).stdout.strip()
    torch_ver = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True).stdout.strip()
    with open(os.path.join(docs_root, "tools", "requirements-check.txt"), "w", encoding="utf-8") as f:
        f.write("torch==%s\n" % torch_ver)
    build_rc, build_wall, build_note = None, None, "skipped (--no-build)"
    if not args.no_build:
        rc, bout, berr, wall = build_tree(args.tree)
        build_rc, build_wall = rc, wall
        with open(os.path.join(out_dir, "build.log"), "w", encoding="utf-8") as f:
            f.write(bout)
            f.write(berr)
        if rc != 0:
            build_note = "BUILD FAILED rc=%d wall=%.1fs" % (rc, wall)
        else:
            build_note = "build rc=0 wall=%.1fs (BUILD OK line ignored as gate)" % wall
    ok_ext, ext_note = assert_extension(args.tree, args.ext_dir)
    example_env = dict(os.environ)
    if args.ext_dir:
        example_env["PYTHONPATH"] = args.ext_dir + os.pathsep + example_env.get("PYTHONPATH", "")
    results = []
    for ex in manifest["examples"]:
        ex_id = "%s.%s" % (ex["id"], ex["lang"])
        log_path = os.path.join(out_dir, ex_id + ".log")
        if ex["fragment"]:
            results.append((ex, "SKIPPED (fragment)", 0.0, "skip-run fence"))
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("SKIPPED (fragment)\n")
            continue
        header = ex.get("header") or {}
        device = (header.get("device") or "unknown").lower()
        if device in NON_CPU_DEVICES:
            results.append((ex, "SKIPPED (requires-%s)" % device, 0.0, "device=%s, CPU harness does not run" % device))
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("SKIPPED (requires-%s)\n" % device)
            continue
        with open(ex["path"], encoding="utf-8") as f:
            code = f.read()
        timeout = args.train_timeout if TRAIN_RE.search(code) else args.timeout
        rc, out, err, wall, timed_out = run_cmd(
            [sys.executable, ex["path"]], args.tree, example_env, timeout=timeout)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("exit=%s wall=%.1fs timeout=%s\n" % (rc, wall, timed_out))
            f.write("--- stdout ---\n" + out)
            f.write("--- stderr ---\n" + err)
        if timed_out:
            results.append((ex, "FAIL", wall, "timeout after %ds" % timeout))
            continue
        failures = check_output(code, out, err)
        if rc != 0:
            failures.insert(0, "exit code %s" % rc)
        if failures:
            results.append((ex, "FAIL", wall, "; ".join(failures)))
        else:
            notes = []
            if ex.get("dependent"):
                notes.append("DEPENDENT on %s" % ",".join(ex["dependent"]))
            if not ex.get("header"):
                notes.append("no-header")
            results.append((ex, "PASS", wall, "; ".join(notes) or "ok"))
    n_pass = sum(1 for _, s, _, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _, _ in results if s.startswith("SKIPPED"))
    lines = []
    lines.append("# CPU verification run %s" % stamp)
    lines.append("")
    lines.append("commit: %s" % commit)
    lines.append("torch: %s" % torch_ver)
    lines.append("build: %s" % build_note)
    if args.ext_dir:
        lines.append("extension dir (override): %s" % args.ext_dir)
    lines.append("extension gate: %s" % ("PASS " + ext_note if ok_ext else "FAIL " + ext_note))
    lines.append("totals: PASS %d / FAIL %d / SKIPPED %d / total %d" % (n_pass, n_fail, n_skip, len(results)))
    lines.append("")
    lines.append("| example | lang | status | wall (s) | notes |")
    lines.append("|---|---|---|---|---|")
    for ex, status, wall, notes in results:
        lines.append("| %s | %s | %s | %.1f | %s |" % (ex["id"], ex["lang"], status, wall, notes))
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("run %s: PASS %d FAIL %d SKIPPED %d total %d" % (stamp, n_pass, n_fail, n_skip, len(results)))
    print("build: %s" % build_note)
    print("extension gate: %s" % ("PASS " + ext_note if ok_ext else "FAIL " + ext_note))
    print("summary: %s" % os.path.join(out_dir, "summary.md"))
    if not ok_ext or (build_rc is not None and build_rc != 0) or n_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
