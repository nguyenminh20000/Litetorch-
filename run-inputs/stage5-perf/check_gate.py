import argparse
import json
import sys

BLOCK_THRESHOLD_PCT = 10.0
BLOCKING_DEVICES = ("cuda", "tpu")


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--threshold", type=float, default=BLOCK_THRESHOLD_PCT)
    args = ap.parse_args()

    base_payload = load(args.baseline)
    cand_payload = load(args.candidate)
    base = {b["name"]: b for b in base_payload["benchmarks"]}
    cand = {b["name"]: b for b in cand_payload["benchmarks"]}

    device = cand_payload.get("device", base_payload.get("device", "unknown"))
    blocking = device in BLOCKING_DEVICES or device == "unknown"

    blocked = []
    blocked_skip = []
    skipped = []
    warned = []
    for name, c in cand.items():
        b = base.get(name)
        if b is None:
            warned.append((name, "no baseline entry"))
            continue
        for side in ("torch", "litetorch"):
            cb, bb = c.get(side, {}), b.get(side, {})
            cb_skip = cb.get("skipped")
            bb_skip = bb.get("skipped")
            if not cb_skip and not bb_skip:
                if bb["median_ms"] <= 0:
                    continue
                pct = (cb["median_ms"] - bb["median_ms"]) / bb["median_ms"] * 100.0
                if pct > args.threshold:
                    blocked.append((name, side, pct, bb["median_ms"], cb["median_ms"]))
                continue
            reason = (cb.get("reason") or bb.get("reason") or "skipped")
            skipped.append((name, side, reason))
            if side != "litetorch":
                continue
            if blocking:
                if cb_skip and not bb_skip:
                    blocked_skip.append((name, "candidate litetorch measurement missing"))
                elif bb_skip and not cb_skip:
                    blocked_skip.append((name, "baseline litetorch measurement missing"))
                else:
                    blocked_skip.append((name, "litetorch measurement missing on both sides"))
            else:
                warned.append((name, "litetorch skipped: " + reason))

    if blocking:
        measured_lt = 0
        for c in cand.values():
            if not c.get("litetorch", {}).get("skipped"):
                measured_lt += 1
        if measured_lt == 0 and not blocked_skip:
            blocked_skip.append(("(all)", "no litetorch measurements in candidate on BLOCKING device " + device))

    for name, side, pct, bmed, cmed in blocked:
        sys.stdout.write("BLOCK %s [%s]: +%.1f%% (baseline %.3fms -> candidate %.3fms)\n" % (name, side, pct, bmed, cmed))
    for name, reason in blocked_skip:
        sys.stdout.write("BLOCK %s [litetorch]: skipped (%s) - no measurement, cannot pass BLOCKING gate on device %s\n" % (name, reason, device))
    for name, side, reason in skipped:
        sys.stdout.write("SKIP %s [%s]: %s\n" % (name, side, reason))
    for name, reason in warned:
        sys.stdout.write("WARN %s: %s\n" % (name, reason))
    if blocked or blocked_skip:
        sys.stdout.write("verdict: BLOCK\n")
        return 1
    if warned or skipped:
        sys.stdout.write("GATE PASS WITH WARNINGS: %d warned, %d skipped\n" % (len(warned), len(skipped)))
        sys.stdout.write("verdict: PASS\n")
        return 0
    sys.stdout.write("GATE PASS: no BLOCKING regression over %.1f%%\n" % args.threshold)
    sys.stdout.write("verdict: PASS\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
