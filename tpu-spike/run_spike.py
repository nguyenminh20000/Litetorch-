import argparse
import json
import os
import subprocess
import sys

import numpy as np

CONFIGS = [
    ("f32_512_timing", ["512", "512", "512", "f32", "11", "3", "0"]),
    ("f32_4096", ["4096", "4096", "4096", "f32", "7", "2", "0"]),
    ("bf16_4096", ["4096", "4096", "4096", "bf16", "7", "2", "0"]),
    ("bf16_rect_1024x4096x4096", ["1024", "4096", "4096", "bf16", "7", "2", "0"]),
    ("bf16_sanity_256", ["256", "256", "256", "bf16", "1", "0", "1"]),
]


def run_one(binary, args, extra, workdir):
    cmd = [binary] + extra + args
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=workdir)
    entry = {"cmd": cmd, "returncode": proc.returncode}
    result = None
    device_lines = []
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            result = json.loads(line[len("RESULT_JSON "):])
        if line.startswith(("device[", "platform_", "num_addressable", "pjrt_api_version", "libtpu_path", "plugin_attribute", "compile_ms")):
            device_lines.append(line)
    entry["result"] = result
    entry["info_lines"] = device_lines
    entry["stdout_tail"] = proc.stdout.splitlines()[-15:]
    entry["stderr_tail"] = proc.stderr.splitlines()[-15:]
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--workdir", required=True)
    ns = ap.parse_args()
    workdir = ns.workdir
    os.makedirs(workdir, exist_ok=True)
    out = {"role": "spike_under_test", "runs": {}}
    rng = np.random.default_rng(20261119)
    a = (rng.standard_normal((512, 512)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((512, 512)) * 0.5).astype(np.float32)
    a_path = os.path.join(workdir, "gate_a_512.bin")
    b_path = os.path.join(workdir, "gate_b_512.bin")
    a.tofile(a_path)
    b.tofile(b_path)
    out["gate_inputs"] = {"a": a_path, "b": b_path, "seed": 20261119, "scale": 0.5}
    gate_extra = ["--load-a=" + a_path, "--load-b=" + b_path,
                  "--save-out=" + os.path.join(workdir, "out_spike_512.bin")]
    out["runs"]["f32_512_gate"] = run_one(ns.bin, ["512", "512", "512", "f32", "3", "1", "1"], gate_extra, workdir)
    for name, args in CONFIGS:
        out["runs"][name] = run_one(ns.bin, args, [], workdir)
    with open(os.path.join(workdir, "partial_spike.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: (v.get("result") if isinstance(v, dict) else v) for k, v in out["runs"].items()}, indent=2))
    print("partial_spike.json written")


if __name__ == "__main__":
    sys.exit(main())
