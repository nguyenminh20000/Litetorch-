import argparse
import json
import os
import subprocess
import sys

import numpy as np

os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_reserve_recv_ready_sync_flags=true"

RUNS = [
    ("allreduce_gate_1024", ["allreduce", "1024", "f32", "3", "1", "1"], "ar_gate"),
    ("allreduce_bw_1mb", ["allreduce", "512", "f32", "7", "2", "0"], None),
    ("allreduce_bw_16mb", ["allreduce", "2048", "f32", "7", "2", "0"], None),
    ("allreduce_bw_64mb", ["allreduce", "4096", "f32", "7", "2", "0"], None),
    ("matmul_rowshard_fp32_512_gate", ["matmul-rowshard", "--partitions=8", "512", "512", "512", "f32", "3", "1", "1"], "mm_gate"),
    ("matmul_rowshard_fp32_8192", ["matmul-rowshard", "--partitions=8", "8192", "8192", "8192", "f32", "7", "2", "0"], None),
    ("matmul_rowshard_fp32_8192_1dev", ["matmul-rowshard", "--partitions=1", "8192", "8192", "8192", "f32", "7", "2", "0"], None),
    ("matmul_rowshard_bf16_8192", ["matmul-rowshard", "--partitions=8", "8192", "8192", "8192", "bf16", "7", "2", "0"], None),
    ("matmul_rowshard_bf16_8192_1dev", ["matmul-rowshard", "--partitions=1", "8192", "8192", "8192", "bf16", "7", "2", "0"], None),
    ("matmul_rowshard_bf16_rect_1024x4096x4096", ["matmul-rowshard", "--partitions=8", "1024", "4096", "4096", "bf16", "7", "2", "0"], None),
    ("matmul_kshard_fp32_512_gate", ["matmul-kshard", "--partitions=8", "512", "512", "512", "f32", "3", "1", "1"], "mm_gate"),
    ("matmul_kshard_fp32_4096", ["matmul-kshard", "--partitions=8", "4096", "4096", "4096", "f32", "7", "2", "0"], None),
    ("matmul_kshard_bf16_4096", ["matmul-kshard", "--partitions=8", "4096", "4096", "4096", "bf16", "7", "2", "0"], "mm_bf16_4096"),
    ("matmul_kshard_bf16_rect_1024x4096x4096", ["matmul-kshard", "--partitions=8", "1024", "4096", "4096", "bf16", "7", "2", "0"], None),
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
        if line.startswith(("device[", "platform_", "num_addressable", "pjrt_api_version", "libtpu_path", "plugin_attribute", "compile_ms", "compile_options_hex", "on_device_bytes")):
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
    out = {"role": "multichip_under_test", "runs": {}}
    rng = np.random.default_rng(20261119)
    a = (rng.standard_normal((512, 512)) * 0.5).astype(np.float32)
    b = (rng.standard_normal((512, 512)) * 0.5).astype(np.float32)
    a_path = os.path.join(workdir, "gate_a_512.bin")
    b_path = os.path.join(workdir, "gate_b_512.bin")
    a.tofile(a_path)
    b.tofile(b_path)
    out["gate_inputs"] = {"a": a_path, "b": b_path, "seed": 20261119, "scale": 0.5}
    rng_ar = np.random.default_rng(20261122)
    ar_inputs = (rng_ar.standard_normal((8, 1024, 1024)) * 0.5).astype(np.float32)
    ar_path = os.path.join(workdir, "allreduce_inputs_1024.bin")
    ar_inputs.tofile(ar_path)
    out["allreduce_gate_inputs"] = {"file": ar_path, "shape_per_device": [1024, 1024], "seed": 20261122, "scale": 0.5}
    rng_bf16 = np.random.default_rng(20261123)
    ba = (rng_bf16.standard_normal((4096, 4096)) * 0.5).astype(np.float32)
    bb = (rng_bf16.standard_normal((4096, 4096)) * 0.5).astype(np.float32)
    ba_path = os.path.join(workdir, "gate_bf16_4096_a.bin")
    bb_path = os.path.join(workdir, "gate_bf16_4096_b.bin")
    ba.tofile(ba_path)
    bb.tofile(bb_path)
    out["bf16_4096_inputs"] = {"a": ba_path, "b": bb_path, "seed": 20261123, "scale": 0.5,
                               "note": "FP32 files; both sides round to BF16 with round-to-nearest-even before compute"}
    gate_extra = ["--load-a=" + a_path, "--load-b=" + b_path,
                  "--save-out=" + os.path.join(workdir, "out_rowshard_512.bin")]
    kshard_gate_extra = ["--load-a=" + a_path, "--load-b=" + b_path,
                         "--save-out=" + os.path.join(workdir, "out_kshard_512.bin")]
    ar_gate_extra = ["--load-inputs=" + ar_path,
                     "--save-out=" + os.path.join(workdir, "out_allreduce_1024.bin")]
    bf16_4096_extra = ["--load-a=" + ba_path, "--load-b=" + bb_path,
                       "--save-out=" + os.path.join(workdir, "out_kshard_bf16_4096.bin")]
    extras = {"ar_gate": ar_gate_extra, "mm_gate": gate_extra, "mm_bf16_4096": bf16_4096_extra}
    out["libtpu_init_args"] = os.environ.get("LIBTPU_INIT_ARGS", "")
    print("LIBTPU_INIT_ARGS=" + out["libtpu_init_args"])
    for name, args, kind in RUNS:
        if kind == "mm_gate" and name.startswith("matmul_kshard"):
            extra = kshard_gate_extra
        else:
            extra = extras.get(kind, [])
        out["runs"][name] = run_one(ns.bin, args, extra, workdir)
    with open(os.path.join(workdir, "partial_multichip.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: (v.get("result") if isinstance(v, dict) else v) for k, v in out["runs"].items()}, indent=2))
    print("partial_multichip.json written")
    hexes = sorted({line.split("=", 1)[1] for e in out["runs"].values()
                    for line in e.get("info_lines", []) if line.startswith("compile_options_hex=")})
    print("compile_options_hex values seen: " + ",".join(hexes))
    print("LIBTPU_INIT_ARGS=" + out["libtpu_init_args"])


if __name__ == "__main__":
    sys.exit(main())
