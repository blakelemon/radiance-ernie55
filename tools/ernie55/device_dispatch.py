#!/usr/bin/env python3
"""ERNIE device dispatch inventory (story 4-1).

Reads the REAL declared graph (`radiance --debug-graph` output: every operation the ernie55 plugin declares, with the host kernel the registry chose) and the REAL
libr4d row table (libr4d/r4d_rows.cpp, parsed as text: libr4d is a gfx1201 library and cannot be loaded without HIP), and writes a JSON report with, per distinct
operation: target, ABI, shape/parameters, dtype, cache layout, host kernel, device result (kernel rows, or the missing kernel named with a follow-up story) and the fallback.

  device_dispatch.py --graph graph.txt [--plugin ernie55.so] --out report.json [--repo DIR]
  device_dispatch.py --rad model.rad --radiance BIN --home DIR [--plugin ernie55.so] --out report.json

Nothing here runs a device kernel or reads GPU sysfs: gfx1201 compile-only commands are listed, never executed. A graph with no declared operations is an error.
Row matching is by operation, dtype word (the cache segment for kv_dtype) and the shape clauses that name a declared graph parameter (`match`: op+dtype+shape, evaluated lexically: multiple-of, <=, >=, a..b, exactly); every candidate row keeps its own constraint text and result. A shape every row rejects is `missing` with the host fallback.
"""
import argparse, ctypes, json, os, re, subprocess, sys, tempfile
from pathlib import Path

ABI_RE = re.compile(r"#define\s+RAD_ABI_VERSION\s+(\d+)u")
HDR = re.compile(r"^([a-z][a-z0-9_]*)(?:\s+(.*))?$")
BAND = re.compile(r"^\s+M\s*<=\s*(\d+)\s*->\s*(\S+?)\s*\((\w+),\s*prio\s*(-?\d+)\)")
LAYOUT_KEYS = ("kv_dtype", "block_size", "n_head_kv", "cmp_layout", "rope_table", "layout", "ratio", "window", "fold", "a_order", "page", "num_pages")


class Fail(Exception):
    pass


def strings(text, i):
    """One C string-literal group (adjacent literals concatenated) starting at or after i; -> (value, end)."""
    n = len(text)
    while i < n and text[i] in " \t\r\n":
        i += 1
    if i >= n or text[i] != '"':
        raise Fail("string expected")
    out = []
    while i < n and text[i] == '"':
        j = i + 1
        while text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        out.append(text[i + 1:j])
        i = j + 1
        while i < n and text[i] in " \t\r\n":
            i += 1
    return "".join(out), i


def r4d_rows(path):
    text = Path(path).read_text()
    m = re.search(r"static const RadKernelInfo kKernels\[\]\s*=\s*\{", text)
    if not m:
        raise Fail(f"{path}: no kKernels table")
    rows, i = [], m.end()
    end = text.find("\n};", i)
    while True:
        k = text.find("{ \"", i, end)
        if k < 0:
            break
        try:
            f, p = [], k + 1
            for _ in range(6):
                s, p = strings(text, p)
                f.append(s)
                while text[p] in " \t\r\n,":
                    p += 1
            rows.append(dict(zip(("name", "op", "family", "computes", "shape", "dtypes"), f)))
            i = p
        except (Fail, IndexError):
            i = k + 3
    if not rows:
        raise Fail("no libr4d rows parsed")
    parsed = {r["name"] for r in rows}
    unparsed = []   # rows the text parser cannot read (macro-built names such as attn_*_h#HD, or unusual literals): never silently absent
    for mm in re.finditer(r'^\{ "([^"]*)"[^\n]*?(?:,\s*"([A-Za-z0-9_]+)",)?', text[m.end():end + 3], re.M):
        if mm.group(1) not in parsed or "#" in mm.group(0):
            head = text[m.end() + mm.start():m.end() + mm.start() + 400]
            ops = re.findall(r'"([a-z][a-z0-9_]*)"\s*,', head)
            macro = "#" in head.split("\n")[0]
            i_op = 0 if macro else 1   # a macro-built name ("attn_window_h" #HD ...) has no comma after the literal, so the op is the first comma-terminated literal
            unparsed.append({"opener": mm.group(1), "op_guess": ops[i_op] if len(ops) > i_op else None})
    unparsed = [u for u in unparsed if u["opener"] not in parsed or "#" in u["opener"] or u["op_guess"] is not None and u["opener"] not in parsed]
    return rows, unparsed


def parse_graph(text):
    ops, cur, banner, inside = [], None, None, False
    for line in text.splitlines():
        if line.startswith("==== declared graph"):
            banner, inside = line, True
            continue
        if line.startswith("----"):
            inside = False
            continue
        if not inside or not line.strip():
            if not line.strip():
                cur = None
            continue
        m = HDR.match(line)
        if m and not line.startswith(" "):
            params = {}
            for tok in (m.group(2) or "").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    params[k] = v
            cur = {"op": m.group(1), "params": params, "bands": []}
            ops.append(cur)
            continue
        b = BAND.match(line)
        if b and cur is not None:
            cur["bands"].append({"max_rows": int(b.group(1)), "kernel": b.group(2), "library": b.group(3), "prio": int(b.group(4))})
    if banner is None or not ops:
        raise Fail("no declared graph in the input (a `radiance --debug-graph` dump is expected)")
    mm = re.search(r"rank (\d+)/(\d+)\s+max_tok (\d+)\s+max_seqs (\d+)", banner)
    return ({"banner": banner, "rank": int(mm.group(1)) if mm else None, "world": int(mm.group(2)) if mm else None,
             "max_tok": int(mm.group(3)) if mm else None, "max_seqs": int(mm.group(4)) if mm else None}, ops)


def host_source(repo, op):
    hits = []
    for p in sorted((Path(repo) / "libref").glob("*.cpp")):
        if p.name == "ref_registry.cpp":
            continue
        if re.search(rf"\bint\s+ref_{re.escape(op)}\s*\(", p.read_text(errors="replace")):
            hits.append(f"libref/{p.name}")
    return hits or ["libref/ref_registry.cpp"]


_NUM = r"(\d+)"
_SUB = r"(?<![*/+\-\w])"   # a name that is an operand of an expression (M*top_k) is not a parameter constraint
CLAUSES = [  # (regex over a row's shape text, check(value, n) -> bool, description)
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s+a\s+multiple\s+of\s+{_NUM}\b"), lambda v, n: v % n == 0, "a multiple of"),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s*%\s*{_NUM}\b(?!\s*==)"), lambda v, n: v % n == 0, "% (a multiple of)"),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|at most|up to)\s*{_NUM}\b"), lambda v, n: v <= n, "<="),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*{_NUM}\b"), lambda v, n: v >= n, ">="),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s+{_NUM}\s+exactly\b"), lambda v, n: v == n, "exactly"),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s*(?<![<>!=])=(?!=)\s*{_NUM}\b"), lambda v, n: v == n, "="),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s+divisible\s+by\s+{_NUM}\b"), lambda v, n: v % n == 0, "divisible by"),
    # K divisible by SK*16 and K/SK by 256: SK is a runtime split count, so K must be a multiple of the numeric factor (necessary condition)
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s+divisible\s+by\s+[A-Za-z_][A-Za-z0-9_]*\s*\*\s*{_NUM}\b"), lambda v, n: v % n == 0, "divisible by <split>*N (K a multiple of N)"),
    (re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s*/\s*[A-Za-z_][A-Za-z0-9_]*\s+by\s+{_NUM}\b"), lambda v, n: v % n == 0, "per-split divisible by (K a multiple of N)"),
]
UNRES = re.compile(r"multiple of|divisible|%|<=|>=|\bat most\b|\bup to\b|\bexactly\b|\d+\.\.\d+|\b[A-Za-z_][A-Za-z0-9_]*\s*(?<![<>!=])=(?!=)\s*\d|\bbounded\b|\bpitch|aligned")
# storage-quantised markers: a row mentioning one of these is not a plain bf16/f16/f32 row
QMARK = re.compile(r"(?<![a-z0-9])(iqr|int8|i8|fp8|e4m3|i4|w4a8|trellis|four-bit|4-bit)(?![a-z0-9])", re.I)
RANGE = re.compile(rf"{_SUB}\b([A-Za-z_][A-Za-z0-9_]*)\s+{_NUM}\.\.{_NUM}\b")


def eval_shape(shape, params):
    """Lexical evaluation of the shape clauses of a libr4d row that name a parameter of this graph entry (case-sensitive: N and n differ). A name that is
    an operand of an expression (M*top_k <= 1024) is not a parameter constraint. Returns (checks, spans) where spans are the text ranges the evaluated clauses cover."""
    checks, spans = [], []
    for rx, fn, label in CLAUSES:
        for m in rx.finditer(shape):
            name, n = m.group(1), int(m.group(2))
            if name in params and re.fullmatch(r"-?\d+", params[name]):
                v = int(params[name])
                checks.append({"param": name, "rule": f"{label} {n}", "value": v, "ok": bool(fn(v, n))})
                spans.append(m.span())
    for m in RANGE.finditer(shape):
        name, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        if name in params and re.fullmatch(r"-?\d+", params[name]):
            v = int(params[name])
            checks.append({"param": name, "rule": f"{lo}..{hi}", "value": v, "ok": lo <= v <= hi})
            spans.append(m.span())
    return checks, spans


STORAGE_ALIAS = {"e4m3": "fp8", "int8": "i8"}


ROLE_AFTER = re.compile(r"\s*(weights?|arithmetic|accumulat\w*|outputs?|out|scales?|indices|index|partials|masks?|descales?)\b")
ACT_AFTER = re.compile(r"\s*(rows?|table|activations?|input|inputs|query|queries|a|x|k|v|q)\b")


def activation_dtype_ok(dtype, dtypes):
    """The dtype parameter names the activation/table type (a whitelist, not a blacklist). An occurrence counts when it leads the row text (the first operand
    named) or is followed by an activation-role word (table, activation, input, ...), and is not followed by another role (weights, arithmetic, output, ...);
    "i32 slot mapping", "f32 arithmetic", "f32 weights and output" never match."""
    for m in re.finditer(rf"(?<![a-z0-9]){re.escape(dtype)}(?![a-z0-9])", dtypes):
        rest = dtypes[m.end():]
        if ROLE_AFTER.match(rest):
            continue
        if not dtypes[:m.start()].strip(" ,") or ACT_AFTER.match(rest):
            return True
    return False


def storage_ok(want, dtypes, strict=False):
    """Storage-role match. Quantised markers in a row's text name its storage (IQR, int8, fp8, i4, ...). A plain want (bf16/f16/f32) accepts only rows naming no
    quantised storage; with strict (the cache role) it must also be the row's own plain dtype word, and an unknown want matches nothing."""
    marks = {STORAGE_ALIAS.get(m.lower(), m.lower()) for m in QMARK.findall(dtypes)}
    w = STORAGE_ALIAS.get(want, want)
    if marks:
        return w in marks
    if w in ("bf16", "f16", "f32"):
        return (not strict) or bool(re.search(rf"(?<![a-z0-9]){w}(?![a-z0-9])", dtypes))
    return not strict


def device_result(op, dtype, rows, params=None, kv_dtype=None, unparsed_ops=()):
    params = params or {}
    cands = [r for r in rows if r["op"] == op]
    if not cands and op in unparsed_ops:
        return {"status": "missing", "kernels": [], "match": "op+dtype+shape", "missing_reason": f"libr4d has rows for {op} the text parser cannot read (macro-built names): not inventoried, see targets[gfx1201].unparsed_rows"}
    if not cands:
        return {"status": "missing", "kernels": [], "match": "op+dtype+shape", "missing_reason": f"no libr4d row for operation {op}"}
    if dtype:
        ok = [r for r in cands if activation_dtype_ok(dtype, r["dtypes"])]
        if not ok:
            return {"status": "missing", "kernels": [], "match": "op+dtype+shape", "missing_reason": f"libr4d has {len(cands)} row(s) for {op} but none lists dtype {dtype}"}
        cands = [r for r in ok if storage_ok(dtype, r["dtypes"])]
        if not cands:
            return {"status": "missing", "kernels": [], "match": "op+dtype+shape", "missing_reason": f"libr4d has {len(ok)} row(s) for {op} listing {dtype} but every one stores a quantised operand (IQR/int8/fp8/i4), not a plain {dtype} one"}
    if kv_dtype:
        ok = [r for r in cands if storage_ok(kv_dtype, r["dtypes"], strict=True)]
        if not ok:
            return {"status": "missing", "kernels": [], "match": "op+dtype+shape", "missing_reason": f"libr4d has {len(cands)} row(s) for {op} but none stores a {kv_dtype} cache"}
        cands = ok
    detail = []
    for r in cands:
        checks, spans = eval_shape(r["shape"], params)
        unres = [m.group(0) for m in UNRES.finditer(r["shape"]) if not any(a <= m.start() < b for a, b in spans)]
        detail.append({"kernel": r["name"], "shape_constraint": r["shape"], "checks": checks, "eligible": all(c["ok"] for c in checks), "dtypes": r["dtypes"],
                       "verified": not unres, "unresolved_clauses": unres})
    elig = [d for d in detail if d["eligible"]]
    if not elig:
        bad = next(c for d in detail for c in d["checks"] if not c["ok"])
        return {"status": "missing", "kernels": [], "match": "op+dtype+shape", "candidates": detail,
                "missing_reason": f"every libr4d row for {op} rejects this shape: {bad['param']}={bad['value']} violates '{bad['rule']}' ({len(detail)} candidate row(s))"}
    return {"status": "kernel", "kernels": sorted(d["kernel"] for d in elig), "match": "op+dtype+shape", "constraints_checked": "lexical",
            "verified": any(d["verified"] for d in elig),
            "constraints_note": "clauses that name a declared graph parameter are evaluated; other clauses (M, tiles, pitches) are carried as text per candidate",
            "candidates": detail}


def registry_schema(repo, op):
    """Operand names and roles plus the registry's own description of an operation (libref/ref_registry.cpp), the host source of operand types and shapes."""
    try:
        text = (Path(repo) / "libref/ref_registry.cpp").read_text(errors="replace")
    except OSError:
        return None
    m = re.search(rf'\{{\s*"{re.escape(op)}",\s*ARR\((\w+)\),\s*ARR\((\w+)\),', text)
    if not m:
        return None
    try:
        doc, _ = strings(text, m.end())
    except (Fail, IndexError):
        doc = None
    arr = re.search(rf"static const RadOperandSpec {m.group(2)}\[\]\s*=\s*\{{(.*?)\}};", text, re.S)
    roles = {"OPD": "in", "OPD_O": "in (optional)", "OUT": "out", "OUT_O": "out (optional)", "INOUT": "inout", "INOUT_O": "inout (optional)", "WGT": "weight", "WGT_O": "weight (optional)"}
    ops = [{"name": n, "role": roles.get(k, k)} for k, n in re.findall(r'\b(OPD_O|OPD|OUT_O|OUT|INOUT_O|INOUT|WGT_O|WGT)\("([^"]+)"\)', arr.group(1))] if arr else []
    words = sorted({w.lower() for w in re.findall(r"(?<![A-Za-z0-9])(bf16|f16|f32|i32|u8|e4m3|fp8|i8|int8|u32|i64)(?![A-Za-z0-9])", doc or "", re.I)})
    return {"operands": ops, "doc": doc, "dtype_words_in_doc": words, "source": "libref/ref_registry.cpp"}


def plugin_info(path):
    if not path:
        return None
    lib = ctypes.CDLL(str(path), mode=os.RTLD_LAZY)
    lib.rad_plugin_abi_version.restype = ctypes.c_uint32
    lib.rad_arch_id.restype = ctypes.c_char_p
    return {"id": lib.rad_arch_id().decode(), "abi": int(lib.rad_plugin_abi_version()), "path": Path(path).name}


def graph_from_radiance(a):
    with tempfile.TemporaryDirectory(prefix="dd-") as d:
        y = Path(d) / "g.yaml"
        y.write_text("\n".join(["server:", "  host: 127.0.0.1", "  port: 0", "  log: info", f"  radiance-home: {a.home}", "  kernels: libref", "  debug-accept-reference-kernels: true",
                                "cards: all", "models:", "  ern:", f"    model: {a.rad}", "    max-num-seqs: 4", "    max-num-batched-tokens: 64", "    max-model-len: 64",
                                "    debug-graph: true", "    kv-cache-dtype: bf16", ""]))
        env = dict(os.environ, ROCR_VISIBLE_DEVICES="99", RADIANCE_HOST_VRAM_MIB="512", OMP_NUM_THREADS="2")
        r = subprocess.run([a.radiance, "--config", str(y)], capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            raise Fail("radiance --debug-graph failed: " + r.stderr[-400:])
        return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--graph")
    ap.add_argument("--rad")
    ap.add_argument("--radiance")
    ap.add_argument("--home")
    ap.add_argument("--plugin")
    ap.add_argument("--allow-no-plugin", action="store_true", help="synthetic graphs only: no plugin ABI cross-check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo)
    try:
        header_abi = int(ABI_RE.search((repo / "abi/rad_abi.h").read_text()).group(1))
        if a.graph:
            gtext = Path(a.graph).read_text()
        elif a.rad and a.radiance and a.home:
            gtext = graph_from_radiance(a)
        else:
            raise Fail("give --graph FILE or --rad/--radiance/--home")
        graph, ops = parse_graph(gtext)
        rows, unparsed = r4d_rows(repo / "libr4d/r4d_rows.cpp")
        plugin = None
        if a.plugin and not a.allow_no_plugin:
            plugin = plugin_info(a.plugin)
            if plugin["abi"] != header_abi:
                raise Fail(f"plugin ABI {plugin['abi']} is not the header's RAD_ABI_VERSION {header_abi}")
        elif not a.allow_no_plugin:
            raise Fail("--plugin is required (or --allow-no-plugin for a synthetic graph)")
        decisions = json.loads((repo / "docs/ernie55/device-dispatch-decisions.json").read_text())
    except (Fail, OSError, ValueError, AttributeError) as e:
        print(f"device_dispatch: {e}", file=sys.stderr)
        return 2
    groups = {}
    for o in ops:
        key = (o["op"], json.dumps(o["params"], sort_keys=True), o["bands"][0]["kernel"] if o["bands"] else None)
        groups.setdefault(key, []).append(o)
    out_ops = []
    for (op, _, hk), items in sorted(groups.items()):
        p = items[0]["params"]
        dtype = p.get("dtype") or p.get("kv_dtype")
        dev = device_result(op, p.get("dtype"), rows, p, p.get("kv_dtype"), {u["op_guess"] for u in unparsed if u["op_guess"]})   # kv_dtype is the cache role, dtype the activation/weight role
        band = items[0]["bands"][0] if items[0]["bands"] else {}
        entry = {"op": op, "count": len(items), "target": "gfx1201", "abi": header_abi, "dtype": dtype,
                 "dtype_source": "graph parameter" if dtype else "unresolved in the graph dump: see operand_schema (the registry's operand list and description, with the dtype words it names)",
                 "params": p, "cache_layout": {k: p[k] for k in LAYOUT_KEYS if k in p},
                 "host": {"kernel": band.get("kernel") or hk, "library": band.get("library", "libref"), "max_rows": band.get("max_rows")},
                 "device": dev, "fallback": {"kind": "host", "kernel": band.get("kernel") or hk, "library": band.get("library", "libref")}}
        entry["operand_schema"] = registry_schema(repo, op) or {"unresolved": "no registry entry found for this operation"}
        if dev["status"] == "missing":
            has_rows = bool(dev.get("candidates")) or "rejects this shape" in dev.get("missing_reason", "")
            entry["follow_up"] = {"kind": "rows exist, shape rejected: re-run at production geometry before writing a kernel" if has_rows else "no device row: a new kernel", "story": f"dev-{op.replace('_', '-')}", "host_parity": f"device {op} against the libref reference {band.get('kernel') or hk} within the recorded parity band",
                                  "code_map": host_source(repo, op) + [f"libr4d/r4d_rows.cpp (new row for {op})" if not has_rows else f"libr4d/r4d_rows.cpp (existing {op} rows reject this shape: see device.candidates)"]}
        out_ops.append(entry)
    names = sorted({o["op"] for o in out_ops})
    missing = sorted({o["op"] for o in out_ops if o["device"]["status"] == "missing"})
    report = {
        "schema": 1, "story": "4-1-device-dispatch-inventory", "abi": header_abi, "plugin": plugin,
        "graph": {"banner": graph["banner"], "single_rank": graph["world"] == 1, "world_size": graph["world"], "max_tokens": graph["max_tok"], "max_seqs": graph["max_seqs"],
                  "operations_declared": len(ops), "distinct_entries": len(out_ops)},
        "baseline": {"dtype": "bf16", "kv_cache_dtype": "bf16", "note": "BF16 model and cache; the fp8 expert/attention checkpoints are a separate inventory"},
        "targets": [
            {"name": "host", "library": "libref", "role": "reference and fallback"},
            {"name": "gfx1201", "library": "libr4d", "rows": len(rows), "unparsed_rows": unparsed, "compile_only": {
                "executed": False, "reason": "no hipcc on the CPU host; the commands are listed for the GPU work items",
                "commands": ["cmake -S . -B build-gfx1201 -G Ninja -DRAD_WITH_HIP=ON -DRAD_GPU_TARGETS=gfx1201",
                             "nice ninja -C build-gfx1201 -j 8 libr4d   # compile only: gfx1201 code objects, nothing is launched"]}}],
        "operations": out_ops,
        "summary": {"device_ready": [n for n in names if n not in missing], "device_missing": missing, "ops_distinct": len(names),
                    "device_ready_verified": [n for n in names if n not in missing and all(o["device"].get("verified") for o in out_ops if o["op"] == n)],
                    "device_ready_unverified": [n for n in names if n not in missing and not all(o["device"].get("verified") for o in out_ops if o["op"] == n)]},
        "decisions": decisions,
    }
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
