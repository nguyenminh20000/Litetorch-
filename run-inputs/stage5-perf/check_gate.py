import argparse
import json
import sys

BLOCK_THRESHOLD_PCT = 10.0


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--threshold", type=float, default=BLOCK_THRESHOLD_PCT)
    args = ap.parse_args()

    base = {b["name"]: b for b in load(args.baseline)["benchmarks"]}
    cand = {b["name"]: b for b in load(args.candidate)["benchmarks"]}

    blocked = []
    warned = []
    for name, c in cand.items():
        b = base.get(name)
        if b is None:
            warned.append((name, "no baseline entry"))
            continue
        for side in ("torch", "litetorch"):
            cb, bb = c.get(side, {}), b.get(side, {})
            if cb.get("skipped") or bb.get("skipped"):
                continue
            if bb["median_ms"] <= 0:
                continue
            pct = (cb["median_ms"] - bb["median_ms"]) / bb["median_ms"] * 100.0
            if pct > args.threshold:
                blocked.append((name, side, pct, bb["median_ms"], cb["median_ms"]))

    for name, side, pct, bmed, cmed in blocked:
        sys.stdout.write("BLOCK %s [%s]: +%.1f%% (baseline %.3fms -> candidate %.3fms)\n" % (name, side, pct, bmed, cmed))
    for name, reason in warned:
        sys.stdout.write("WARN %s: %s\n" % (name, reason))
    if not blocked and not warned:
        sys.stdout.write("GATE PASS: no BLOCKING regression over %.1f%%\n" % args.threshold)
    sys.stdout.write("verdict: " + ("BLOCK" if blocked else "PASS") + "\n")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
