import argparse
import ast
import json
import os
import re
import subprocess
import sys

FENCE_RE = re.compile(r"^```(\S*)(.*)$")
HEADING_RE = re.compile(r"^#{1,6}\s+(\d+(?:\.\d+)*)\b")
HEADER_RE = re.compile(r"<!--\s*ex:\s*(.*?)\s*-->")
KV_RE = re.compile(r"([a-z-]+)=([^\s]+)")
PRINT_STR_RE = re.compile(r"""print\(([^)]*)\)""")

BUILTINS = set(dir(__builtins__)) | {"print", "range", "len", "list", "dict", "set", "tuple"}


def parse_header(text):
    m = HEADER_RE.search(text)
    if not m:
        return None
    fields = dict(KV_RE.findall(m.group(1)))
    required = {"id", "device", "backend", "runtime", "torch-parity", "status"}
    if not required.issubset(fields):
        fields["_malformed"] = sorted(required - set(fields))
    return fields


def strip_print_strings(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return re.sub(r"print\([^)]*\)", "print()", code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        sub.value = ""
    return ast.dump(tree, include_attributes=False)


def block_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    assigned = set()
    used = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
        elif isinstance(node, ast.arg):
            assigned.add(node.arg)
    return assigned, (used - imported - BUILTINS)


def scan_file(path, section, lang, tree_root):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    blocks = []
    subsection = section
    counts = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        hm = HEADING_RE.match(line)
        if hm:
            subsection = hm.group(1)
            i += 1
            continue
        fm = FENCE_RE.match(line)
        if fm and fm.group(1) == "python":
            info = (fm.group(2) or "").strip()
            fragment = info == "skip-run"
            header = None
            for back in range(1, 4):
                if i - back >= 0 and HEADER_RE.search(lines[i - back]):
                    header = parse_header(lines[i - back])
                    break
            j = i + 1
            body = []
            while j < len(lines) and not lines[j].startswith("```"):
                body.append(lines[j])
                j += 1
            code = "\n".join(body)
            n = counts.get(subsection, 0) + 1
            counts[subsection] = n
            ex_id = "%s-%d" % (subsection, n)
            blocks.append({
                "id": ex_id,
                "lang": lang,
                "file": os.path.basename(path),
                "line": i + 1,
                "header": header,
                "fragment": fragment,
                "code": code,
                "section": section,
            })
            i = j + 1
            continue
        i += 1
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.path.expanduser("~/workspace/framework/integration"))
    ap.add_argument("--docs", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/workspace/framework/docs/examples"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    docs_root = args.docs or os.path.join(args.tree, "docs-en")
    docs_root_vi = os.path.join(os.path.dirname(docs_root.rstrip("/")), "docs-vie") if args.docs is None else None
    if args.docs is not None:
        docs_root_vi = os.path.join(os.path.dirname(args.docs.rstrip("/")), "docs-vie")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.tree, capture_output=True, text=True).stdout.strip()
    all_blocks = []
    errors = []
    for lang, root in (("en", docs_root), ("vi", docs_root_vi)):
        for name in sorted(os.listdir(root)):
            if not name.endswith(".md"):
                continue
            section = name.split(".")[0]
            all_blocks.extend(scan_file(os.path.join(root, name), section, lang, args.tree))
    by_id = {}
    for b in all_blocks:
        by_id.setdefault(b["id"], {})[b["lang"]] = b
    for ex_id in sorted(by_id):
        pair = by_id[ex_id]
        if set(pair) != {"en", "vi"}:
            errors.append("%s: missing language side %s" % (ex_id, sorted(set(["en", "vi"]) - set(pair))))
            continue
        en_norm = strip_print_strings(pair["en"]["code"])
        vi_norm = strip_print_strings(pair["vi"]["code"])
        if en_norm != vi_norm:
            errors.append("%s: EN/VI code bodies diverge after print-string stripping" % ex_id)
    for lang in ("en", "vi"):
        seen = set()
        cumulative = set()
        for b in [x for x in all_blocks if x["lang"] == lang]:
            assigned, used = block_names(b["code"])
            deps = sorted((used - assigned) & cumulative)
            b["dependent"] = deps
            b["code_norm"] = strip_print_strings(b["code"])
            cumulative |= assigned
            seen.add(b["id"])
    manifest = {"tree": args.tree, "commit": commit, "examples": []}
    for b in all_blocks:
        out_dir = os.path.join(args.out, b["lang"], b["file"].replace(".md", ""))
        os.makedirs(out_dir, exist_ok=True)
        fname = "ex-%s.py" % b["id"].replace(".", "-")
        path = os.path.join(out_dir, fname)
        preamble = "import sys\nsys.path.insert(0, %r)\n" % os.path.join(args.tree, "python")
        with open(path, "w", encoding="utf-8") as f:
            f.write(preamble + b["code"] + "\n")
        entry = {
            "id": b["id"],
            "lang": b["lang"],
            "file": b["file"],
            "line": b["line"],
            "header": b["header"],
            "fragment": b["fragment"],
            "dependent": b.get("dependent", []),
            "path": path,
        }
        manifest["examples"].append(entry)
    manifest_path = args.manifest or os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    n_frag = sum(1 for b in all_blocks if b["fragment"])
    n_dep = sum(1 for b in all_blocks if b["dependent"])
    n_nohdr = sum(1 for b in all_blocks if b["header"] is None)
    n_bad = sum(1 for b in all_blocks if b["header"] and "_malformed" in b["header"])
    print("extracted %d blocks (%d en, %d vi) commit %s" % (
        len(all_blocks),
        sum(1 for b in all_blocks if b["lang"] == "en"),
        sum(1 for b in all_blocks if b["lang"] == "vi"),
        commit))
    print("fragments(skip-run): %d  dependent: %d  no-header: %d  malformed-header: %d" % (n_frag, n_dep, n_nohdr, n_bad))
    for e in errors:
        print("PARITY-FAIL " + e)
    if args.strict and (errors or n_nohdr or n_bad):
        print("strict mode: failing on missing/malformed headers or parity divergence")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
