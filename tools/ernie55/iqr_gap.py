#!/usr/bin/env python3
"""ERNIE 5.5 Flash IQR-V4 / IQ8R / BF16 kernel gap list (story 5-0-iqr-v4-target, CPU only).

For every tensor class of the real checkpoint (classes come from the safetensors HEADERS only: no tensor bytes are read) and every operation of the 4-1 device
dispatch inventory, say which libr4d row serves it on gfx1201 in each serving format, at ERNIE's real shapes, and which is missing.

  iqr_gap.py classes --checkpoint DIR --out classes.json          # headers only
  iqr_gap.py gap [--classes C] [--rows libr4d/r4d_rows.cpp] [--dispatch D] --out-json J [--out-md M]
  iqr_gap.py probe --rows R --row-name NAME KEY=VAL ...           # evaluate ONE row's constraint array at one shape

Verdicts per (class, format):
  exists          every M band (1, 16, 64, 2048) has a libr4d row whose constraint array accepts the shape
  partial         some M bands have an accepting row, others do not (the bands are listed)
  shape_rejected  rows for the operation and format exist and every one rejects the shape (the failing clause is named)
  missing         no libr4d row exists for the operation in that format (custom ernie_* operations, or a format with no kernel)
  not_applicable  a vector or scalar that is never a GEMM operand (norm gain, bias, mHC scalar): stored plain
The constraint arrays are parsed from libr4d/r4d_rows.cpp itself (C_EQ/C_LE/C_GE/C_DIV/C_IN), so the verdict follows the table.
"""
import argparse, collections, json, re, struct, sys
from pathlib import Path

GROUP_MACROS = {"R4D_FP8_BLOCK": 128, "R4D_I8_GROUP_WIDE": 256, "R4D_W4_GROUP": 128}
M_BANDS = (1, 16, 64, 2048)
FORMATS = ("iqr_v4", "iq8r", "bf16")
SIZES = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4, "F8_E8M0": 1}


class Fail(Exception):
    pass


# ---------------------------------------------------------------- the checkpoint (headers only)
def norm_name(k):
    n = re.sub(r"(?<=layers\.)\d+", "L", k)
    n = re.sub(r"(?<=experts\.)\d+", "E", n)
    return n


def checkpoint_classes(ckpt):
    """Group every tensor of the checkpoint by (layer-normalised name, shape, dtype), reading only each shard's JSON header."""
    ckpt = Path(ckpt)
    groups = collections.OrderedDict()
    n_tensors = 0
    shards = sorted(ckpt.glob("*.safetensors"))
    if not shards:
        raise Fail(f"{ckpt}: no safetensors shards")
    for f in shards:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > 64 << 20:
                raise Fail(f"{f}: header {n} bytes")
            hdr = json.loads(fh.read(n))
        hdr.pop("__metadata__", None)
        for k, v in hdr.items():
            key = (norm_name(k), tuple(v["shape"]), v["dtype"])
            g = groups.setdefault(key, {"count": 0, "bytes": 0, "example": k, "layers": set(), "experts": set()})
            g["count"] += 1
            m_l = re.search(r"layers\.(\d+)\.", k)
            if m_l:
                g["layers"].add(int(m_l.group(1)))
            m_e = re.search(r"experts\.(\d+)\.", k)
            if m_e:
                g["experts"].add(int(m_e.group(1)))
            g["bytes"] += v["data_offsets"][1] - v["data_offsets"][0]
            n_tensors += 1
    out = []
    for k, v in groups.items():
        v = dict(v)
        v["layers"] = sorted(v["layers"])
        v["experts"] = len(v["experts"])
        out.append({"name": k[0], "shape": list(k[1]), "dtype": k[2], **v})
    return {"schema": "ernie55-checkpoint-classes/1", "tensors": n_tensors, "shards": len(shards), "classes": out}


# ---------------------------------------------------------------- libr4d constraint arrays
def row_blocks(text):
    """name -> text of that row's entry (from its opening line to the next row's)."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'^\{ "([A-Za-z0-9_]+)",', text, re.M)]
    blocks = {}
    for i, (s, name) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        blocks[name] = text[s:e]
    return blocks


def constraint_arrays(text):
    arrs = {}
    for m in re.finditer(r"static const RadConstraint (\w+)\[\]\s*=\s*\{(.*?)\};", text, re.S):
        cons = []
        bad = False
        body = re.sub(r"/\*.*?\*/", "", m.group(2), flags=re.S)
        body = re.sub(r"//[^\n]*", "", body)
        if re.search(r"\bC_(?!EQ\b|LE\b|GE\b|DIV\b|IN\b|P2P\b)\w+\(", body):
            bad = True       # a constraint macro this evaluator does not know: the array is unparsed, never silently more permissive
        for c in re.finditer(r'C_(EQ|LE|GE|DIV|IN)\(\s*"(\w+)"\s*,\s*([^)]*?)\s*\)', body):
            op, key, val = c.group(1), c.group(2), c.group(3).strip()
            if op == "IN":
                cons.append((op, key, val.strip('"').split()))
            else:
                v = GROUP_MACROS.get(val)
                if v is None:
                    try:
                        v = int(val)
                    except ValueError:
                        bad = True   # a macro-built array (attention head dims): not used by any row this tool evaluates
                        continue
                cons.append((op, key, v))
        arrs[m.group(1)] = None if bad else cons
    return arrs


def parse_rows(rows_path):
    text = Path(rows_path).read_text()
    arrs = constraint_arrays(text)
    blocks = row_blocks(text)
    rows = {}
    for name, b in blocks.items():
        m = re.match(r'\{ "[A-Za-z0-9_]+",\s*"([a-z0-9_]+)"', b)
        if not m:
            continue
        a = re.search(r"ROW\((c\w+)\)", b)
        rows[name] = {"op": m.group(1), "array": a.group(1) if a else None,
                      "constraints": arrs.get(a.group(1)) if a else None}
    if not rows:
        raise Fail(f"{rows_path}: no libr4d rows parsed")
    return rows


def evaluate(cons, params):
    """-> (ok, failed clause text or None, [keys the row constrains that the caller gave no value for])."""
    unev = []
    for op, key, v in cons:
        if key not in params:
            unev.append(key)
            continue
        x = params[key]
        if op == "IN":
            ok = str(x) in v
            txt = f'{key} in "{" ".join(v)}"'
        else:
            ok = {"EQ": x == v, "LE": x <= v, "GE": x >= v, "DIV": v != 0 and x % v == 0}[op]
            txt = f"{key} {({'EQ': '==', 'LE': '<=', 'GE': '>=', 'DIV': 'multiple of'})[op]} {v}"
        if not ok:
            return False, f"{txt} (given {x})", unev
    return True, None, unev


# ---------------------------------------------------------------- the model: tensor classes -> operation, shape, per-format rows
# kind: gemm (gemm_nt family) | gemm_f32 | moe | logits | embed | custom (an ernie_* op reads the weight) | plain (never a GEMM operand)
# rows[fmt]: libr4d row names that serve that format for the kind; param dtype/group lists are tried in turn.
ROWSETS = {
    "gemm": {
        "bf16": ["gemm_bf16_nt_m16", "gemm_bf16_nt_m64", "gemm_bf16_nt_tiled"],
        "iq8r": ["gemm_iq8r_nt_m16", "gemm_iq8r_tiled"],
        "iqr_v4": ["gemm_iqr_nt_m64", "gemm_iqr_tiled"],
    },
    "gemm_f32": {"bf16": ["gemm_nt_f32_bf16"], "iq8r": [], "iqr_v4": []},
    "moe": {
        "bf16": ["moe_gemm_bf16"],
        "iq8r": ["moe_gemm_i8a8"],
        "iqr_v4": ["moe_gemm_iqr"],
    },
    "logits": {
        "bf16": ["logits_gemm_bf16", "logits_gemm_bf16_cols"],
        "iq8r": ["logits_gemm_i8"],
        "iqr_v4": ["logits_gemm_iqr", "logits_gemm_vocab_iqr"],
    },
    "embed": {
        "bf16": ["embed_lookup_bf16"],
        "iq8r": ["embed_lookup_q_bf16"],
        "iqr_v4": ["embed_lookup_iqr"],
    },
}
# the operation each row set must serve: a row is credited only when the parsed row table says the row is for that operation
ROW_OPS = {("gemm", "bf16"): {"gemm_nt"}, ("gemm", "iq8r"): {"gemm_nt_q"}, ("gemm", "iqr_v4"): {"gemm_nt_q"}, ("gemm_f32", "bf16"): {"gemm_nt_f32"},
           ("moe", "bf16"): {"moe_gemm"}, ("moe", "iq8r"): {"moe_gemm_q"}, ("moe", "iqr_v4"): {"moe_gemm_q"},
           ("logits", "bf16"): {"logits_gemm"}, ("logits", "iq8r"): {"logits_gemm"}, ("logits", "iqr_v4"): {"logits_gemm", "logits_gemm_vocab"},
           ("embed", "bf16"): {"embed_lookup"}, ("embed", "iq8r"): {"embed_lookup_q"}, ("embed", "iqr_v4"): {"embed_lookup"}}
DTYPES = {"bf16": ["bf16"], "iq8r": ["i8a8h", "i8a8"], "iqr_v4": ["i8a8h"]}
GROUPS = {"bf16": [0], "iq8r": [256, 128], "iqr_v4": [128]}

# Closest rows Radiance already has for the custom ERNIE operations (other models, other geometry): reported as neighbours, never counted as exists.
NEIGHBOURS = {
    "ernie_mla_project": ["mla_project_bf16 (head_dim 128 or 256, latent 512; ERNIE k_dim 192 / v_dim 256, 64 heads)"],
    "ernie_mla_sparse_attn": ["mla_attn_paged_bf16 (latent 512, rope 64, heads<=64, block<=256; dense page table, not the DSA gather)", "mla_attn_bf16 / mla_attn_rows_bf16 (latent 512, heads<=64, block_size 4)"],
    "ernie_idx_query": ["dsa_query_fp8 (n=128, flattened heads)"],
    "ernie_idx_key": ["dsa_pool_fp8 (n=128, ratio 4)"],
    "ernie_idx_logits": ["dsa_score_fp8 (n=128, heads=32; ERNIE has 64 indexer heads)"],
    "ernie_mhc_pre_rms": ["mhc_pre_bf16 (n up to 16384, hc 2..8, f32 coefficients)"],
    "ernie_mhc_head": ["mhc_contract_bf16"],
    "ernie_router_topk": ["router_sigmoid_topk_f32 (single group, experts<=512, top_k<=32, f32 bias)", "router_gemm_topk_scatter_bf16"],
    "ernie_hca_wo_a": [],
    "ernie_vha_mix": [],
    "ernie_hca_compress": [],
    "ernie_hca_union_attn": ["attn_paged / attn_paged_sink (rows with macro-built names, unparsed by 4-1)"],
    "situ_mul": ["silu_mul_bf16 / swiglu_step_bf16 / swiglu_clamped_bf16 (SiLU family; SiTU has no row)"],
}

# (regex on the layer-normalised checkpoint name, attributes). First match wins; the shape of the checkpoint class gives (N, K) for gemms.
CLASS_RULES = [
    (r"\.weight_scale_inv$|_scale_inv$|linear_o_group_proj_scale_inv$", dict(role="fp8 block-scale plane of its weight (a source-only plane; the converter folds it)", kind="plain", op=None)),
    (r"experts\.E\.w[123]\.weight$", dict(role="routed expert w1/w3 (gate/up) and w2 (down), latent width 2048", kind="moe", op="moe_gemm", logical="blk.L.ffn_{gate,up,down}_exps.E.weight")),
    (r"fc[12]_latent_proj\.weight$", dict(role="latent down/up projection around the routed block", kind="gemm", op="gemm_nt", logical="blk.L.ffn_latent_{in,out}.weight")),
    (r"shared_experts\.w[123]\.weight$", dict(role="shared expert gate/up/down (4096 <-> 2048)", kind="gemm", op="gemm_nt", logical="blk.L.ffn_{gate,up,down}_shexp.weight")),
    (r"block_sparse_moe\.gate\.weight(_1)?$", dict(role="the two f32 routers (bf16 stored, f32 compute)", kind="gemm_f32", op="gemm_nt_f32", logical="blk.L.ffn_gate_inp{,_1}.weight")),
    (r"mlp\.(gate|up|down)_proj\.weight$", dict(role="layer-0 dense feed-forward", kind="gemm", op="gemm_nt", logical="blk.0.ffn_{gate,up,down}.weight")),
    (r"self_attn\.linear_q_down_proj\.weight$", dict(role="HCA q down", kind="gemm", op="gemm_nt", logical="blk.L.attn_q_a.weight")),
    (r"self_attn\.linear_q_up_proj\.weight$", dict(role="HCA q up (64 heads x 512)", kind="gemm", op="gemm_nt", logical="blk.L.attn_q_b.weight")),
    (r"self_attn\.linear_kv_proj\.weight$", dict(role="HCA kv (one 512-wide head)", kind="gemm", op="gemm_nt", logical="blk.L.attn_kv_a_mqa.weight")),
    (r"self_attn\.linear_o_group_proj$", dict(role="HCA grouped output projection wo_a (read inside ernie_hca_wo_a)", kind="custom", op="ernie_hca_wo_a", logical="blk.L.attn_o_group.weight")),
    (r"self_attn\.q_a_proj\.weight$", dict(role="MLA-DSA q down", kind="gemm", op="gemm_nt", logical="blk.L.attn_q_a.weight")),
    (r"self_attn\.q_b_proj\.weight$", dict(role="MLA-DSA q up (64 heads x (192+64))", kind="gemm", op="gemm_nt", logical="blk.L.attn_q_b.weight")),
    (r"self_attn\.kv_a_proj_with_mqa\.weight$", dict(role="MLA-DSA kv_a (latent 512 + rope 64)", kind="gemm", op="gemm_nt", logical="blk.L.attn_kv_a_mqa.weight")),
    (r"self_attn\.kv_b_proj\.weight$", dict(role="MLA-DSA kv_b (read inside ernie_mla_project, absorb and expand)", kind="custom", op="ernie_mla_project", logical="blk.L.attn_kv_b.weight")),
    (r"self_attn\.gate_proj\.weight$", dict(role="attention output gate (sigmoid)", kind="gemm", op="gemm_nt", logical="blk.L.attn_gate.weight")),
    (r"self_attn\.o_proj\.weight$", dict(role="attention output projection wo_b", kind="gemm", op="gemm_nt", logical="blk.L.attn_output.weight")),
    (r"indexer\.wq_b\.weight$", dict(role="DSA indexer query up (64 heads x 128)", kind="gemm", op="gemm_nt", logical="blk.L.attn_idx_q_b.weight")),
    (r"indexer\.wk\.weight$", dict(role="DSA indexer key", kind="gemm", op="gemm_nt", logical="blk.L.attn_idx_k.weight")),
    (r"indexer\.weights_proj\.weight$", dict(role="DSA indexer head weights (f32 compute)", kind="gemm_f32", op="gemm_nt_f32", logical="blk.L.attn_idx_w.weight")),
    (r"compressor\.linear_w(kv|gate)\.weight$", dict(role="HCA compressor kv/score projections (bf16 stored, f32 compute)", kind="gemm_f32", op="gemm_nt_f32", logical="blk.L.attn_comp_{kv,gate}.weight")),
    (r"layers\.L\.(e|h)_proj\.weight$", dict(role="MTP e/h projections", kind="gemm", op="gemm_nt", logical="blk.43.mtp_{e,h}_proj.weight")),
    (r"embed_tokens\.weight$", dict(role="token embedding", kind="embed", op="embed_lookup", logical="token_embd.weight")),
    (r"^lm_head\.weight$", dict(role="output head (MultiMax transform follows in f32)", kind="logits", op="logits_gemm", logical="output.weight")),
    (r"mapping_proj\.weight$", dict(role="mHC mixing matrix [hc*hidden, 24] f32 (read inside ernie_mhc_pre_rms)", kind="custom", op="ernie_mhc_pre_rms", logical="blk.L.{attn,ffn}_hc.fn")),
    (r"hc_head_fn$", dict(role="mHC head mixing matrix f32 (read inside ernie_mhc_head)", kind="custom", op="ernie_mhc_head", logical="head_hc.fn")),
    (r"vha_postmix_[UV]$", dict(role="VHA post-mix low-rank factors (read inside ernie_vha_mix)", kind="custom", op="ernie_vha_mix", logical="blk.L.attn_vha_{u,v}.weight")),
    (r"core_attention\.compressor\.ape$", dict(role="HCA compressor position bias (read inside ernie_hca_compress)", kind="custom", op="ernie_hca_compress", logical="blk.L.attn_comp_ape.weight")),
]
PLAIN_ROLE = "vector or scalar (norm gain, bias, sink, mHC alpha/base/scale, MultiMax table): stored plain, never a GEMM operand"


def classify(name, shape):
    for rx, attrs in CLASS_RULES:
        if re.search(rx, name):
            return dict(attrs)
    if len(shape) <= 1 or (len(shape) == 2 and min(shape) <= 4 and "mapping" not in name):
        return dict(role=PLAIN_ROLE, kind="plain", op=None)
    raise Fail(f"unclassified checkpoint class {name} {shape}: add it to CLASS_RULES")


def nk(attrs, name, shape):
    """The GEMM (N, K) of a class: the weight is [N, K]; the moe/gemm shapes are as stored."""
    if attrs["kind"] in ("gemm", "gemm_f32", "moe", "custom", "logits", "embed") and len(shape) == 2:
        return shape[0], shape[1]
    return None, None


def eval_format(kind, fmt, N, K, rows, n_expert=512, top_k=10):
    names = ROWSETS.get(kind, {}).get(fmt, [])
    if not names:
        return dict(verdict="missing", rows=[], reason=f"no libr4d {kind} row for format {fmt}", bands={})
    present = [n for n in names if n in rows and rows[n]["op"] in ROW_OPS[(kind, fmt)]]
    if not present:
        return dict(verdict="missing", rows=[], reason=f"rows {names} are not in the table", bands={})
    bands, rej, accepted, unevaluated = {}, {}, set(), set()
    for M in M_BANDS:
        ok_rows = []
        for rn in present:
            r = rows[rn]
            if r["constraints"] is None:
                continue
            fails = []
            ok_any = False
            for dt in DTYPES[fmt] if kind in ("gemm", "moe") else ["bf16"]:
                for g in GROUPS[fmt] if kind in ("gemm", "moe") else [0]:
                    p = dict(M=M, N=N, K=K, dtype=dt, group=g, n_expert=n_expert, top_k=top_k, a_group=128, n_embd=K, n=K, n_vocab=N)
                    ok, why, unev = evaluate(r["constraints"], p)
                    if ok:
                        ok_any = True
                        accepted.add(rn)
                        unevaluated.update(f"{rn}:{k}" for k in unev)
                        break
                    fails.append(why)
                if ok_any:
                    break
            if ok_any:
                ok_rows.append(rn)
            else:
                rej.setdefault(rn, fails[0] if fails else "rejected")
        bands[str(M)] = ok_rows
    full = [b for b, rs in bands.items() if rs]
    if len(full) == len(M_BANDS):
        verdict = "exists"
    elif full:
        verdict = "partial"
    else:
        verdict = "shape_rejected"
    reason = None
    if verdict != "exists":
        reason = "; ".join(f"{rn}: {w}" for rn, w in rej.items())
    return dict(verdict=verdict, rows=sorted(accepted), reason=reason, bands=bands, unevaluated=sorted(unevaluated))


def trellis_run(N):
    """Largest power of two <= 1024 dividing N: the longest tcq_rows an IQR trellis over N rows may use (the rows say 'N a multiple of a trellis's run')."""
    r = 1
    while r < 1024 and N % (r * 2) == 0:
        r *= 2
    return r


def build_gap(classes, rows, dispatch, production=None):
    rows_by_op0 = {}
    for rn, r in rows.items():
        rows_by_op0.setdefault(r["op"], []).append(rn)
    entries = []
    for c in classes["classes"]:
        attrs = classify(c["name"], c["shape"])
        N, K = nk(attrs, c["name"], c["shape"])
        e = dict(class_name=c["name"], shape=c["shape"], dtype=c["dtype"], count=c["count"], bytes=c["bytes"], example=c["example"], layers=c.get("layers", []), experts=c.get("experts", 0),
                 role=attrs["role"], kind=attrs["kind"], op=attrs["op"], logical=attrs.get("logical"), N=N, K=K, formats={})
        if c["name"].endswith("weight_scale_inv") or attrs["kind"] == "plain" and "scale_inv" in c["name"]:
            e["kind"] = "plain"
        for fmt in FORMATS:
            k = e["kind"]
            if k == "plain":
                e["formats"][fmt] = dict(verdict="not_applicable", rows=[], reason=attrs["role"], bands={})
            elif k == "custom":
                orows = sorted(rows_by_op0.get(attrs["op"], []))
                if orows:
                    e["formats"][fmt] = dict(verdict="partial", rows=orows, bands={},
                                             reason=f"the row table now lists {orows} for operation {attrs['op']}, which reads this weight itself: whether they accept its encoding and geometry is not established (unverified)",
                                             neighbours=NEIGHBOURS.get(attrs["op"], []))
                else:
                    e["formats"][fmt] = dict(verdict="missing", rows=[], bands={},
                                             reason=f"operation {attrs['op']} has no row in the libr4d table, so no weight format can be read by it (the weight is read inside the operation)",
                                             neighbours=NEIGHBOURS.get(attrs["op"], []))
            else:
                e["formats"][fmt] = eval_format(k, fmt, N, K, rows)
        if e["kind"] in ("gemm", "moe") and N:
            e["trellis_run_max"] = trellis_run(N)
        entries.append(e)
    # operations of the 4-1 inventory
    summ = dispatch["summary"]
    ready = set(summ["device_ready"])
    prod_ready = set((production or {}).get("summary", {}).get("device_ready", []))
    rows_by_op = {}
    for rn, r in rows.items():
        rows_by_op.setdefault(r["op"], []).append(rn)
    cls_by_op = {}
    for e in entries:
        if e["op"]:
            cls_by_op.setdefault(e["op"], []).append(e)
    ops_out = []
    for op in sorted(set(summ["device_ready"]) | set(summ["device_missing"])):
        in_table = sorted(rows_by_op.get(op, []))
        s41 = "exists" if op in ready else "missing"
        prod = "exists" if op in prod_ready else ("missing" if production and op in set(production["summary"].get("device_missing", [])) else "not probed")
        shape_ok = True
        if op in cls_by_op:     # a weight-reading operation: the rows must still accept the REAL shapes (bf16 is the 4-1 baseline), not just have been accepted by an old probe
            shape_ok = all(c["formats"]["bf16"]["verdict"] in ("exists", "partial") for c in cls_by_op[op] if c["kind"] != "custom")
            if not any(c["kind"] != "custom" for c in cls_by_op[op]):
                shape_ok = True
        if in_table and (s41 == "exists" or prod == "exists") and shape_ok:
            status = "exists"
        elif in_table:
            status = "rows_present_unverified"      # the table names the operation but neither the 4-1 fixture nor the production probe shows a row accepting it
        else:
            status = "missing"
        o = dict(op=op, status=status, status_4_1=s41, status_at_production_geometry=prod, rows=in_table, neighbours=NEIGHBOURS.get(op, []))
        if op in cls_by_op:
            o["formats"] = {f: sorted({c["formats"][f]["verdict"] for c in cls_by_op[op]}) for f in FORMATS}
            o["weight_classes"] = len(cls_by_op[op])
        ops_out.append(o)
    per = {f: collections.Counter() for f in FORMATS}
    for e in entries:
        if e["kind"] == "plain":
            continue
        for f in FORMATS:
            per[f][e["formats"][f]["verdict"]] += 1
    summary = dict(
        classes=len(entries), tensors=sum(e["count"] for e in entries),
        per_format={f: {v: per[f].get(v, 0) for v in ("exists", "partial", "shape_rejected", "missing")} for f in FORMATS},
        plain_classes=sum(1 for e in entries if e["kind"] == "plain"),
        ops=dict(exists=sum(1 for o in ops_out if o["status"] == "exists"), missing=sum(1 for o in ops_out if o["status"] == "missing"), rows_present_unverified=sum(1 for o in ops_out if o["status"] == "rows_present_unverified")),
    )
    # the distinct missing kernels the serving path needs beyond the 4-1 ernie_* set
    extra = [
        dict(kernel="situ_had_quant (SiTU gate then rotated int8 quantise)", why="gemm_iqr_gated and gated_had_quant_i8g fold SiLU only (act silu); SiTU has no row, so the routed w2 and shared/dense down inputs need situ_mul plus had_quant_act_i8g as two launches, or a new fused row", classes=["routed w2", "shared w2", "dense down"]),
        dict(kernel="quantised-activation producers after the ERNIE fused ops", why="ernie_mhc_pre_rms and ernie_vha_mix write bf16; IQ8R/IQR GEMMs read rotated int8 (i8a8h). had_quant_act_i8g / rmsnorm_had_quant rows exist (n multiple of 128), the plugin graph must add them", classes=["every IQR/IQ8R gemm and moe_gemm input"]),
        dict(kernel="IQR trellis run vs N", why="IQR rows need N a multiple of the trellis run (tcq_rows); kv_a N=576 allows 64, q_b/kv/gate widths allow 256-1024, the 128-wide indexer key allows 128", classes=["attn kv_a", "indexer wk"]),
        dict(kernel="PLUGIN: arch/ernie55 declares BF16 gemm_nt / moe_gemm only and refuses quantised weights", why="rad-convert --plan-only with the IQR-V4 expert rule stops at declare: 'checkpoint tensor model.layers.1.block_sparse_moe.experts.0.w1.weight ... is u4 [2048, 2048], but this model needs bf16 [2048, 2048] (or block-fp8 of that shape)' (arch/ernie55/ernie55.cpp:293, :645). The graph builds gemm_nt dtype bf16 (:1083) and moe_gemm at bf16 (:1414); IQ8R/IQR need gemm_nt_q / moe_gemm_q at i8a8h, a had_quant_act_i8g producer after each norm and the f32 hc.fn cast rules", classes=["every IQR-V4 and IQ8R class"]),
        dict(kernel="HOST: no libref or libavx kernel reads an IQR (trellis) weight", why="grep of libref/ and libavx/ finds no trellis/tcq reader; libref's gemm_nt_q reads rotated int8 only. The core's decoder (enc_decode_rows) reads every form, so a CPU parity path is a load-time decode to int8 or a new libref reader: until one exists CPU-only stories cannot run the IQR-V4 forward pass, only compare decoded weights (this story's measurement)", classes=["routed experts"]),
        dict(kernel="moe_gemm_iqr at n_expert 512 is the row's ceiling", why="C_LE n_expert 512: ERNIE has exactly 512 experts per layer; the MTP layer too", classes=["routed experts"]),
    ]
    return dict(schema="ernie55-iqr-v4-gap/1", target="gfx1201", abi=43, formats=list(FORMATS), m_bands=list(M_BANDS), classes=entries, operations=ops_out, summary=summary, extra_gaps=extra)


def render_md(g):
    L = ["# ERNIE 5.5 Flash: IQR-V4 / IQ8R / BF16 kernel gap list (story 5-0-iqr-v4-target)", "",
         "Generated by `tools/ernie55/iqr_gap.py` from the safetensors headers of `~/models/ernie55-flash-fp8` and `libr4d/r4d_rows.cpp` (the constraint arrays are parsed, not retyped). gfx1201, ABI 43. exists = a row accepts the shape in every M band (1, 16, 64, 2048).", ""]
    s = g["summary"]
    L += ["## Counts (GEMM-operand classes, excluding the plain vectors)", "", "| format | exists | partial | shape_rejected | missing |", "|---|---:|---:|---:|---:|"]
    for f in g["formats"]:
        p = s["per_format"][f]
        L.append(f"| {f} | {p['exists']} | {p['partial']} | {p['shape_rejected']} | {p['missing']} |")
    miss_e = sum(1 for o in g["operations"] if o["status"] == "missing" and o["op"].startswith("ernie_"))
    miss_o = ", ".join(o["op"] for o in g["operations"] if o["status"] == "missing" and not o["op"].startswith("ernie_")) or "no other operation"
    L += ["", f"{s['classes']} tensor classes ({s['tensors']} tensors, {s['plain_classes']} plain classes). Operations of the 4-1 inventory: {s['ops']['exists']} have a device row, {s['ops']['missing']} do not ({miss_e} `ernie_*` operations and {miss_o} have no row at any shape); {s['ops'].get('rows_present_unverified', 0)} are listed in the row table but verified by no probe. `gemm_nt` and `moe_gemm` read missing only at the 4-1 fixture's tiny shapes and are ready at production geometry (device-dispatch-production-probe.json).", "",
          "## Per tensor class", "", "A `missing` for IQR-V4 or IQ8R on an f32-arithmetic class (routers, indexer head weights, compressor projections) means no quantised row exists for it, which is why the precision map keeps them BF16; a `missing` on a custom-op class means the `ernie_*` operation that reads the weight has no row in ANY format. The map (iqr-v4-precision-map.md) chooses only precisions that have a row or names the blocker.", "", "| class | shape | count | op | IQR-V4 | IQ8R | BF16 |", "|---|---|---:|---|---|---|---|"]
    for e in g["classes"]:
        if e["kind"] == "plain" and e["count"] and "scale_inv" in e["class_name"]:
            continue
        v = e["formats"]
        L.append(f"| `{e['class_name']}` | {e['shape']} {e['dtype']} | {e['count']} | {e['op'] or '-'} | {v['iqr_v4']['verdict']} | {v['iq8r']['verdict']} | {v['bf16']['verdict']} |")
    L += ["", "## Rows behind each exists", ""]
    for e in g["classes"]:
        if e["kind"] in ("gemm", "moe", "logits", "embed", "gemm_f32"):
            L.append(f"- `{e['logical'] or e['class_name']}` [{e['N']}x{e['K']}]: " + "; ".join(f"{f}: {','.join(e['formats'][f]['rows']) or e['formats'][f]['verdict']}" for f in g["formats"]))
    L += ["", "## Operations (4-1 inventory, status derived from the libr4d row table)", "", "`status` looks the operation up in the row table (rows) and takes the 4-1 fixture and the production probe as the shape evidence; `status_4_1` is the fixture's own word, kept as provenance.", "", "| op | status | rows in the table | 4-1 fixture | production geometry | nearest existing row (another model) |", "|---|---|---|---|---|---|"]
    for o in g["operations"]:
        L.append(f"| {o['op']} | {o['status']} | {', '.join(o['rows']) or '-'} | {o['status_4_1']} | {o['status_at_production_geometry']} | {'; '.join(o['neighbours']) or '-'} |")
    L += ["", "## Gaps beyond the ernie_* set", ""]
    for x in g["extra_gaps"]:
        L.append(f"- **{x['kernel']}**: {x['why']} ({', '.join(x['classes'])})")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("classes"); a.add_argument("--checkpoint", required=True); a.add_argument("--out", required=True)
    b = sub.add_parser("gap")
    b.add_argument("--classes", default="docs/ernie55/checkpoint-classes.json")
    b.add_argument("--rows", default="libr4d/r4d_rows.cpp")
    b.add_argument("--dispatch", default="docs/ernie55/device-dispatch.json")
    b.add_argument("--production", default="docs/ernie55/device-dispatch-production-probe.json")
    b.add_argument("--out-json", required=True); b.add_argument("--out-md")
    c = sub.add_parser("probe")
    c.add_argument("--rows", default="libr4d/r4d_rows.cpp"); c.add_argument("--row-name", required=True); c.add_argument("kv", nargs="*")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "classes":
            Path(args.out).write_text(json.dumps(checkpoint_classes(args.checkpoint), indent=1) + "\n")
        elif args.cmd == "gap":
            classes = json.loads(Path(args.classes).read_text())
            rows = parse_rows(args.rows)
            dispatch = json.loads(Path(args.dispatch).read_text())
            prod = json.loads(Path(args.production).read_text()) if args.production and Path(args.production).exists() else None
            g = build_gap(classes, rows, dispatch, prod)
            Path(args.out_json).write_text(json.dumps(g, indent=1) + "\n")
            if args.out_md:
                Path(args.out_md).write_text(render_md(g))
        else:
            rows = parse_rows(args.rows)
            if args.row_name not in rows:
                raise Fail(f"no row {args.row_name}")
            p = {}
            for kv in args.kv:
                k, v = kv.split("=", 1)
                p[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
            r = rows[args.row_name]
            if r["constraints"] is None:
                raise Fail(f"row {args.row_name} has no constraint array")
            ok, why, unev = evaluate(r["constraints"], p)
            print(json.dumps(dict(row=args.row_name, op=r["op"], accepted=ok, failed=why, unevaluated=unev)))
            return 0 if ok else 1
    except (Fail, OSError, ValueError, KeyError) as e:
        print(f"iqr_gap: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
