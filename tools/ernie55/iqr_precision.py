#!/usr/bin/env python3
"""ERNIE 5.5 Flash per-tensor precision map and recipe checker (story 5-0-iqr-v4-target).

  iqr_precision.py build --gap docs/ernie55/iqr-v4-gap.json --recipe data/recipes/ernie55-iqr-v4.recipe --out-json MAP.json [--out-md MAP.md] [--repo .]
  iqr_precision.py check-recipe --recipe FILE

The map assigns every checkpoint class one serving precision (iqr_v4, iq8r, bf16, f32). Rules, in order of authority:
  1. Operator decision 2026-11-23: routed experts IQR-V4; accuracy-sensitive small tensors at higher precision; FP8 is never a serving format.
  2. A precision is only chosen where a libr4d row exists for it (the gap list) or the entry names its blocker and a fallback that does.
  3. How Radiance itself treats the same classes (cited by file and line, found by searching the file, so a citation cannot drift): the flagship MoE keeps the router, embedding and norms bf16, serves block
     linears and the shared expert as int8, the mixing matrices as e4m3 or bf16, the head as int8, and the routed experts in the four-bit forms; the Qwen3.8 IQR-V4 recipes put attention and dense
     projections at IQ8R next to a 4-bit trellis FFN.
"""
import argparse, json, re, sys
from pathlib import Path


class Fail(Exception):
    pass


ALLOWED_QUANT = ("gptq", "rtn", "cast")


def parse_recipe(path):
    rules = []
    for ln, line in enumerate(Path(path).read_text().split("\n"), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise Fail(f"{path}:{ln}: a rule is PATTERN QUANTISER [key=value ...]")
        opts = {}
        for o in parts[2:]:
            if "=" not in o:
                raise Fail(f"{path}:{ln}: option {o!r} is not key=value")
            k, v = o.split("=", 1)
            opts[k] = v
        rules.append(dict(pattern=parts[0], quantiser=parts[1], opts=opts, line=line, lineno=ln))
    return rules


def glob_rx(p):
    return re.compile("^" + re.escape(p).replace(r"\*", ".*").replace(r"\?", ".") + "$")


def _opts_match(o, want):
    return all(o.get(k) == v for k, v in want.items())


# The effective encodings libr4d reads (rows moe_gemm_iqr / gemm_iqr_*, gemm_iq8r_*, moe_gemm_i8a8, logits_gemm_i8). A rule is classified by what it WRITES, not by its quantiser name:
IQ8R_FORMS = (
    dict(codes="i8", group="256", scale="u8", scale2="bf16", transform="fwht128"),      # IQ8R trunk: u8 scale a (row, 256 of K) under a bf16 row scale, rotated (gemm_iq8r_*)
    dict(codes="i8", group="128", scale="bf16", transform="fwht128"),                   # IQ8R-V1 at group 128, rotated (gemm_i8a8_* and moe_gemm_i8a8)
    dict(codes="i8", group="128", scale="bf16"),                                        # the unrotated int8 head (logits_gemm_i8)
)
IQR_REQUIRED = dict(tcq="16", tcq_code="mad", transform="fwht128")                      # the computed 16-bit trellis on the rotated weight (moe_gemm_iqr / gemm_iqr_*)


def rule_form(r):
    """What a rule WRITES: iqr_v4 | iq8r_trunk (rotated: the activation side is rotated int8) | iq8r_head (unrotated: logits_gemm_i8) | cast:<dtype> | None (not an encoding libr4d reads)."""
    o = r["opts"]
    if r["quantiser"] == "gptq":
        return "iqr_v4" if _opts_match(o, IQR_REQUIRED) and o.get("codes", "")[:1] == "u" else None
    if r["quantiser"] == "rtn" and o.get("codes") == "i8":
        if any(_opts_match(o, f) for f in IQ8R_FORMS[:2]):
            return "iq8r_trunk"
        if _opts_match(o, IQ8R_FORMS[2]) and "transform" not in o:
            return "iq8r_head"
        return None       # an int8 rule that is none of the forms libr4d reads (no rotation, or another group/scale layout)
    if r["quantiser"] == "cast":
        return "cast:" + str(o.get("dtype"))
    return None


def rule_precision(r):
    f = rule_form(r)
    if f is None:
        return None
    return {"iqr_v4": "iqr_v4", "iq8r_trunk": "iq8r", "iq8r_head": "iq8r"}.get(f) or f.split(":", 1)[1]


def match(rules, logical):
    for r in rules:
        if glob_rx(r["pattern"]).match(logical):
            return r
    return None


def check_recipe(path):
    errs = []
    rules = parse_recipe(path)
    if not rules:
        errs.append("the recipe has no rules")
    for r in rules:
        if r["quantiser"] not in ALLOWED_QUANT:
            errs.append(f"{r['pattern']}: quantiser {r['quantiser']} is not one of {ALLOWED_QUANT}")
        if "fp8" in r["opts"].get("codes", "") or r["opts"].get("dtype", "").startswith("fp8"):
            errs.append(f"{r['pattern']}: fp8 {r['opts'].get('codes') or r['opts'].get('dtype')} is not a serving format in Radiance 2.x (the FP8 checkpoint is the conversion source only)")
        if r["quantiser"] in ("gptq", "rtn") and rule_precision(r) is None:
            errs.append(f"{r['pattern']}: not an encoding libr4d reads (IQ8R needs codes=i8 with the rotated u8/bf16 or bf16 scale layouts, the head's unrotated group-128 form; IQR-V4 needs tcq=16 tcq_code=mad transform=fwht128): {r['line'].strip()}")
        if re.search(r"_exps\.", r["pattern"]) and not (r["quantiser"] == "gptq" and r["opts"].get("codes") == "u4" and r["opts"].get("tcq") == "16" and r["opts"].get("tcq_code") == "mad"):
            errs.append(f"{r['pattern']}: routed experts must be the IQR-V4 computed trellis (gptq codes=u4 tcq=16 tcq_code=mad), got {r['line'].strip()}")
    return errs


# ------------------------------------------------------------------ citations (found by searching, never typed as numbers)
def cite(repo, rel, needle, nth=1):
    n = 0
    for i, line in enumerate((Path(repo) / rel).read_text().split("\n"), 1):
        if needle in line:
            n += 1
            if n == nth:
                return f"{rel}:{i}"
    raise Fail(f"citation needle not found in {rel}: {needle!r}")


def cites(repo):
    A = "docs/ARCHITECTURES.md"
    R = "data/recipes/qwen4exp-u4-w6nl-bf16ple.recipe"
    return {
        "arch_bf16_router_embed": cite(repo, A, "the router, `shared_expert_gate` and `embed_tokens`"),
        "arch_int8_block_linears": cite(repo, A, "block linears (`self_attn.{q,k,v,o}_proj`"),
        "arch_mix_e4m3": cite(repo, A, "each connection's two mixing matrices"),
        "arch_head": cite(repo, A, "the lm_head: whatever a `logits_gemm` kernel reads"),
        "arch_per_layer_experts": cite(repo, A, "**Under an int8 routed arm each layer and projection takes its own experts' encoding.**"),
        "arch_iqr_experts": cite(repo, A, "scale a row, two to six bits, `moe_gemm_iqr`"),
        "flash_u4_experts": cite(repo, R, "blk.*.ffn_gate_up_exps.*.weight"),
        "flash_i8_attn": cite(repo, R, "blk.*.attn_k.weight                  rtn codes=i8"),
        "flash_i8_shared": cite(repo, R, "blk.*.ffn_gate_up_shexp.weight       rtn codes=i8"),
        "flash_i8_head": cite(repo, R, "output.weight                        rtn codes=i8"),
        "flash_fp8_mix": cite(repo, R, "*_hc_down.weight                     rtn codes=fp8_e4m3"),
        "muse_v4_ffn": cite(repo, "data/recipes/museglimmer-30b-iq4r-v4.recipe", "blk.*.ffn_gate_up.weight  gptq codes=u4"),
        "muse_v4_attn_out": cite(repo, "data/recipes/museglimmer-30b-iq4r-v4.recipe", "blk.*.attn_output.weight  gptq codes=u5"),
        "iq8r_recipe_head": cite(repo, "data/recipes/q38-27b-iq8r.recipe", "output.weight                        gptq"),
        "iq8r_recipe_trunk": cite(repo, "data/recipes/q38-27b-iq8r.recipe", "blk.*.attn_output.weight"),
        "iqr_v4_section": cite(repo, "docs/IQR.md", "## IQR-V4: measured widths"),
        "iqr_head_wide": cite(repo, "docs/IQR.md", "and the head are cheap to keep wide"),
        "row_moe_iqr": cite(repo, "libr4d/r4d_rows.cpp", "static const RadConstraint cMoeGemmIqr[]"),
        "row_head_i8": cite(repo, "libr4d/r4d_rows.cpp", "static const RadConstraint cLogitsGemmFp8[]"),
        "row_iq8r_m16": cite(repo, "libr4d/r4d_rows.cpp", "static const RadConstraint cGemmNtIq8rM16[]"),
        "row_iqr_m64": cite(repo, "libr4d/r4d_rows.cpp", "static const RadConstraint cGemmNtIqrM64[]"),
        "plugin_refuses": cite(repo, "arch/ernie55/ernie55.cpp", "(or block-fp8 of that shape)"),
        "ernie_mhc_recipe": cite(repo, "data/recipes/ernie55-weights.recipe", "blk.*.attn_hc.fn      cast dtype=f32"),
        "moe_forward_f32": cite(repo, "docs/ernie55/moe-forward.md", "The routers are declared f32"),
        "device_dispatch_ops": cite(repo, "docs/ernie55/device-dispatch.md", "every `ernie_*` operation (20 distinct)"),
    }


# ------------------------------------------------------------------ the map
KEPT_TEMPLATES = (
    (r"experts\.E\.w1\.weight$", "blk.{L}.ffn_gate_exps.{E}.weight"), (r"experts\.E\.w3\.weight$", "blk.{L}.ffn_up_exps.{E}.weight"), (r"experts\.E\.w2\.weight$", "blk.{L}.ffn_down_exps.{E}.weight"),
    (r"block_sparse_moe\.gate\.weight$", "blk.{L}.ffn_gate_inp.weight"), (r"block_sparse_moe\.gate\.weight_1$", "blk.{L}.ffn_gate_inp_1.weight"),
    (r"indexer\.wk\.weight$", "blk.{L}.attn_idx_k.weight"), (r"indexer\.weights_proj\.weight$", "blk.{L}.attn_idx_w.weight"),
    (r"compressor\.linear_wkv\.weight$", "blk.{L}.attn_comp_kv.weight"), (r"compressor\.linear_wgate\.weight$", "blk.{L}.attn_comp_gate.weight"), (r"compressor\.ape$", "blk.{L}.attn_comp_ape.weight"),
    (r"vha_postmix_U$", "blk.{L}.attn_vha_u.weight"), (r"vha_postmix_V$", "blk.{L}.attn_vha_v.weight"),
    (r"embed_tokens\.weight$", "token_embd.weight"), (r"^lm_head\.weight$", "output.weight"),
    (r"^model\.layers\.L\.self_attention_hyper_connection\.mapping_proj\.weight$", "blk.{L}.attn_hc.fn"), (r"^model\.layers\.L\.mlp_hyper_connection\.mapping_proj\.weight$", "blk.{L}.ffn_hc.fn"),
    (r"^model\.mhc_contract\.hc_head_fn$", "head_hc.fn"),
)


def template_for(n):
    for rx, t in KEPT_TEMPLATES:
        if re.search(rx, n):
            return t
    return None


def entry_for(c, C):
    n, shape, kind = c["class_name"], c["shape"], c["kind"]
    g = c["formats"]
    d = dict(class_name=n, shape=shape, dtype=c["dtype"], count=c["count"], logical=c.get("logical"), blocked_by=None, fallback=None, recipe_rule_key=None, logical_sample=None, cites=[])

    def put(group, prec, why, cs, logical_sample=None, rule_key=None, blocked=None, fallback=None, template=None):
        d.update(group=group, precision=prec, justification=why, cites=[C[k] for k in cs], logical_sample=logical_sample, recipe_rule_key=rule_key, blocked_by=blocked, fallback=fallback, logical_template=template or (re.sub(r"^blk\.\d+\.", "blk.{L}.", logical_sample) if logical_sample else template_for(n)))
        return d

    if n.endswith("_scale_inv") or "scale_inv" in n:
        return put("source_scale_plane", "f32", "The fp8 block-scale plane belongs to the SOURCE format only: the converter reads it to dequantise the weight and the serving container carries its own scales (IQR-V4: one f32 a row; IQ8R: u8 per 256 of K under a bf16 row scale). Nothing of it is served.", [])
    if re.search(r"experts\.E\.w([123])\.weight$", n):
        w = re.search(r"\.w([123])\.weight$", n).group(1)
        nm = {"1": "gate", "3": "up", "2": "down"}[w]
        return put("routed_experts", "iqr_v4",
                   f"The routed {nm} experts are 86% of the checkpoint (258 of 268.6 GiB), so they take the narrowest grid that keeps the model: IQR-V4's computed 16-bit trellis at four bits under one f32 scale a row (moe_gemm_iqr reads exactly this at i8a8h: N%16, K%128, 512 experts). "
                   "Radiance's own four-bit candidate puts the trellis on its 512 routed experts and the Qwen3.8 V4 recipes on every FFN; IQR-V4 beats block FP8 at 55% of its size on the 27B. Expert GEMMs are the bandwidth term of decode, so the bytes are what pays.",
                   ["flash_u4_experts", "muse_v4_ffn", "arch_iqr_experts", "arch_per_layer_experts", "row_moe_iqr", "iqr_v4_section"], f"blk.1.ffn_{nm}_exps.0.weight", f"ffn_{nm}_exps", template=f"blk.{{L}}.ffn_{nm}_exps.{{E}}.weight")
    if re.search(r"block_sparse_moe\.gate\.weight(_1)?$", n):
        return put("router", "bf16",
                   "The two routers decide which 10 of 512 experts run and Baidu's gate computes its matmul and logits in f32 (moe-forward.md: a bf16 logit moves the output by more than 1e-3). Radiance keeps the router bf16 for the flagship MoE and for every MoE it serves. "
                   "There is no quantised f32-arithmetic router row (gemm_nt_f32 has a bf16-weight row only), so IQ8R would need a new kernel; the bytes at stake are 0.34 GiB. The measured IQ8R top-10 agreement is in iqr-v4-layers.json (block kind router) as the evidence for that later lever.",
                   ["arch_bf16_router_embed", "moe_forward_f32", "device_dispatch_ops"])
    if re.search(r"mapping_proj\.weight$|hc_head_fn$", n):
        return put("mhc_mixing", "f32",
                   "The mHC mixing matrices feed a 20-iteration Sinkhorn and the residual stream mixing of every layer; an error here is multiplied through all 43 layers. Radiance stores the flagship model's mixing matrices as e4m3 rows or bf16 under its own fused read; ERNIE's ernie_mhc_pre_rms reads them f32 and has no device row in any format, "
                   "and they are 0.26 GiB (0.2% of the model), so they stay f32 (rule: cast dtype=f32, the transpose the plugin already does). bf16 is the fallback once a kernel exists and the Sinkhorn sensitivity is measured.",
                   ["arch_mix_e4m3", "flash_fp8_mix", "ernie_mhc_recipe", "device_dispatch_ops"], None, None,
                   "ernie_mhc_pre_rms / ernie_mhc_head have no libr4d row in any weight format (4-1 inventory, story dev-*): the f32 matrix is read by the host libref op until a device row is written", "bf16")
    if re.search(r"indexer\.wq_b\.weight$", n):
        return put("indexer", "iq8r",
                   "The DSA indexer's query up-projection (64 heads x 128) is a plain GEMM and an int8 grid keeps its scores to ~0.3% relative error; the selection it feeds is a top-2048 of dot products, so only near-ties at the cut move. Same class as the attention projections: IQ8R.",
                   ["arch_int8_block_linears", "flash_i8_attn", "row_iq8r_m16"], "blk.7.attn_idx_q_b.weight", "attn_idx_q_b")
    if re.search(r"indexer\.(wk|weights_proj)\.weight$", n):
        return put("indexer", "bf16",
                   "The indexer key (128 x 4096, a bf16 gemm_nt in the plugin) and head-weight (64 x 4096, read at f32) projections are 10 MiB per MLA layer in total, stored bf16 in the release (the key is then quantised to FP8 by the 4-1 quantiser rule, a second rounding we do not stack a third on). Radiance has no quantised row for the f32-arithmetic indexer gate. Kept as shipped.",
                   ["arch_bf16_router_embed", "device_dispatch_ops"])
    if re.search(r"compressor\.linear_w(kv|gate)\.weight$", n):
        return put("hca_compressor", "bf16",
                   "The HCA compressor's kv and score projections (512 x 4096, bf16, read f32) build the compressed cache every 128 tokens; their error accumulates in the fp32 compressor state and is carried by every later token. 296 MiB in total: kept as shipped.",
                   ["arch_bf16_router_embed", "moe_forward_f32"])
    if re.search(r"compressor\.ape$", n):
        return put("hca_compressor", "f32", "The compressor's absolute position bias [128, 512] f32, 9 MiB for all layers, read by ernie_hca_compress. Kept f32.", ["device_dispatch_ops"], None, None,
                   "ernie_hca_compress has no libr4d row in any format (4-1)", "bf16")
    if re.search(r"vha_postmix_[UV]$", n):
        return put("vha", "bf16", "The VHA post-mix factors are [64, 16] bf16 (0.1 MiB each, all layers 0.2 MiB): nothing to gain, and ernie_vha_mix has no quantised reader. Kept bf16.", ["device_dispatch_ops"], None, None,
                   "ernie_vha_mix has no libr4d row in any format (4-1)", "f32")
    if re.search(r"linear_o_group_proj$", n):
        return put("attention", "bf16",
                   "wo_a (the grouped output projection, [8192, 4096] per HCA layer) is read INSIDE ernie_hca_wo_a together with the inverse rotary, and no row of any format exists for that operation. It is dequantised from FP8 to bf16 exactly (e4m3 x power of two is representable) and served bf16 (1.1 GiB over 37 layers). IQ8R is the intended precision when the operation grows an int8 reader.",
                   ["arch_int8_block_linears", "device_dispatch_ops"], "blk.1.attn_o_group.weight", "attn_o_group",
                   "ernie_hca_wo_a has no libr4d row in any weight format (4-1); the weight is read inside the operation", "iq8r")
    if re.search(r"kv_b_proj\.weight$", n):
        return put("attention", "bf16",
                   "kv_b ([28672, 512] per MLA-DSA layer) is read INSIDE ernie_mla_project (the absorb and expand modes); the nearest existing row, mla_project_bf16, is head_dim 128/256 only. Dequantised to bf16 exactly and served bf16 (0.1 GiB total). IQ8R follows the operation.",
                   ["arch_int8_block_linears", "device_dispatch_ops"], "blk.7.attn_kv_b.weight", "attn_kv_b",
                   "ernie_mla_project has no libr4d row in any weight format (4-1); mla_project_bf16 is the nearest (head_dim 128 or 256, bf16)", "iq8r")
    attn = {r"self_attn\.(linear_q_down_proj|q_a_proj)\.weight$": ("attn_q_a", "q down"), r"self_attn\.(linear_q_up_proj|q_b_proj)\.weight$": ("attn_q_b", "q up (64 heads)"),
            r"self_attn\.(linear_kv_proj|kv_a_proj_with_mqa)\.weight$": ("attn_kv_a_mqa", "kv projection"), r"self_attn\.gate_proj\.weight$": ("attn_gate", "attention output gate"), r"self_attn\.o_proj\.weight$": ("attn_output", "attention output projection")}
    for rx, (logical, what) in attn.items():
        if re.search(rx, n):
            extra = " N = 576 has no prefill row (tiled rows need N%128): the narrow rows serve M <= 64 and a 640-row pad or split serves the rest." if c["formats"]["iq8r"]["verdict"] == "partial" else ""
            return put("attention", "iq8r",
                       f"The {what} sits on every token's critical path and is small (the whole attention stack is about 5 GiB of the 268.6). Radiance serves attention projections as int8 W8A8 for the flagship MoE and as rotated int8 (or u5 trellis for attn_output) in the Qwen IQR-V4 recipes; "
                       f"a 4-bit grid here buys 2.5 GiB and costs error in every token. IQ8R (gemm_iq8r_*, K a multiple of 256) it is.{extra}",
                       ["arch_int8_block_linears", "flash_i8_attn", "iq8r_recipe_trunk", "row_iq8r_m16", "muse_v4_attn_out"], f"blk.1.{logical}.weight", logical)
    if re.search(r"fc[12]_latent_proj\.weight$", n):
        lg = "ffn_latent_in" if "fc1" in n else "ffn_latent_out"
        return put("latent_moe", "iq8r", "The latent down/up projections wrap every routed block and see all tokens (4096 <-> 2048, 43 layers, 0.7 GiB): an error here scales the whole routed output, so IQ8R, not the expert grid.",
                   ["arch_int8_block_linears", "flash_i8_shared", "row_iq8r_m16"], f"blk.1.{lg}.weight", lg)
    if re.search(r"shared_experts\.w([123])\.weight$", n):
        w = re.search(r"\.w([123])\.weight$", n).group(1)
        lg = {"1": "ffn_gate_shexp", "3": "ffn_up_shexp", "2": "ffn_down_shexp"}[w]
        return put("shared_expert", "iq8r", "The shared expert is active for every token (unlike a routed expert it is never averaged out) and is 1 GiB over 43 layers. Radiance serves the flagship model's shared expert as int8 and keeps it a grid wider than the routed experts.",
                   ["flash_i8_shared", "arch_int8_block_linears", "row_iq8r_m16"], f"blk.1.{lg}.weight", lg)
    if re.search(r"mlp\.(gate|up|down)_proj\.weight$", n):
        lg = "ffn_" + re.search(r"mlp\.(\w+)_proj", n).group(1)
        return put("dense_ffn", "iq8r", "Layer 0 is the one dense feed-forward (14336 wide, 168 MiB): every token runs it, so IQ8R.", ["arch_int8_block_linears", "iq8r_recipe_trunk", "row_iq8r_m16"], f"blk.0.{lg}.weight", lg)
    if re.search(r"layers\.L\.(e|h)_proj\.weight$", n):
        lg = "mtp_" + ("e" if ".e_proj" in n else "h") + "_proj"
        return put("mtp", "iq8r", "The MTP draft layer's e/h projections (4096 x 4096). The plugin does not declare the draft layer yet, so these have no logical name today (mtp_e_proj / mtp_h_proj are proposed). IQ8R like the trunk linears; the draft is verified by the target, so a draft error costs acceptance rate, not correctness.",
                   ["arch_int8_block_linears", "row_iq8r_m16"], f"blk.43.{lg}.weight", lg)
    if re.search(r"embed_tokens\.weight$", n):
        return put("embedding", "bf16", "The embedding gather feeds every layer and costs 1.5 GiB bf16 (a 4-bit embedding costs IQ8R +0.00058 KL on the 27B, 6-bit +0.00003; docs/IQR.md). Radiance keeps embed_tokens bf16 for the flagship MoE. embed_lookup_q / embed_lookup_iqr exist if the bytes are wanted later (n_embd 4096 is a multiple of 128).",
                   ["arch_bf16_router_embed", "iqr_head_wide"])
    if re.search(r"^lm_head\.weight$", n):
        return put("head", "iq8r", "The output head is 1.5 GiB bf16 and decode reads all of it every token; int8 rows with a bf16 scale a (row, 128) (logits_gemm_i8, n_embd % 128) is what every IQ8R recipe of this tree does, GPTQ'd against the head's input Hessian when one exists. The head's measured top-1 and KL against the bf16 head are in iqr-v4-layers.json. The vocabulary-weighted IQR head (logits_gemm_vocab_iqr) is the later size lever.",
                   ["arch_head", "flash_i8_head", "iq8r_recipe_head", "row_head_i8", "iqr_head_wide"], "output.weight", "output")
    if c["kind"] == "plain":
        return put("vector_or_scalar", "f32" if c["dtype"] == "F32" else "bf16", "A norm gain, bias, sink, mHC alpha/base/scale, correction bias, per-expert scale or MultiMax table: a vector or scalar that no GEMM reads, tiny, and the plugin reads at its stored width. Kept exactly as shipped.", ["arch_bf16_router_embed"])
    raise Fail(f"no precision rule for class {n} {shape}")


def verify_every_logical(e, c, rules, gap):
    """The first matching rule of EVERY logical tensor the class stands for (each layer it occurs in, each expert id) must give the class's precision: a layer-specific rule placed ahead of the
    wildcard cannot hide behind a one-sample check. A class kept as shipped must be matched by no rule, or by a cast to its own dtype; an FP8-stored class needs a rule."""
    t = e.get("logical_template")
    if not t:
        return
    layers = c.get("layers") or [None]
    experts = range(c.get("experts") or 1) if "{E}" in t else [None]
    n = 0
    for L in layers:
        for E in experts:
            name = t.replace("{L}", str(L) if L is not None else "").replace("{E}", str(E) if E is not None else "")
            r = match(rules, name)
            want = e["precision"]
            got = rule_precision(r) if r else None
            if want == "iqr_v4":
                want_form = "iqr_v4"
            elif want == "iq8r":
                want_form = "iq8r_head" if t == "output.weight" else "iq8r_trunk"
            else:
                want_form = None
            if want_form:
                ok = bool(r) and rule_form(r) == want_form       # the head is the one unrotated int8 form, every trunk projection the rotated one
                got = rule_form(r) if r else None
                want = want_form
            elif r is None:
                ok = {"BF16": "bf16", "F32": "f32", "F16": "f16"}.get(e["dtype"]) == want       # kept as shipped: the stored dtype IS the precision; any conversion (or FP8) needs a rule
            else:
                ok = got == want
            if not ok:
                raise Fail(f"{e['class_name']}: logical weight {name} is {('matched by rule ' + r['pattern'] + ' = ' + str(got)) if r else 'matched by no rule'}, the map says {want}")
            n += 1
    e["logical_tensors_checked"] = n


def host_kernel(dispatch, op):
    for o in dispatch.get("operations", []):
        if o["op"] == op and (o.get("host") or {}).get("kernel"):
            return o["host"]["kernel"]
    return None


def build(gap, recipe, repo, dispatch):
    rules = parse_recipe(recipe)
    C = cites(repo)
    entries = []
    for c in gap["classes"]:
        e = entry_for(c, C)
        if e["precision"] in ("iqr_v4", "iq8r"):
            r = match(rules, e["logical_sample"])
            if r is None:
                raise Fail(f"{e['class_name']}: logical weight {e['logical_sample']} is matched by no recipe rule")
            if rule_precision(r) != e["precision"]:
                raise Fail(f"{e['class_name']}: recipe rule {r['pattern']} is {rule_precision(r)}, the map says {e['precision']}")
            e["recipe_rule"] = r["line"]
            e["recipe_line"] = r["lineno"]
        elif e["precision"] == "bf16" and e["class_name"].endswith(("linear_o_group_proj", "kv_b_proj.weight")):
            r = match(rules, e["logical_sample"])
            if r is None or rule_precision(r) != "bf16":
                raise Fail(f"{e['class_name']}: no cast-to-bf16 rule")
            e["recipe_rule"] = r["line"]
        else:
            e["recipe_rule"] = None
        e.pop("recipe_rule_key", None)
        verify_every_logical(e, c, rules, gap)
        # the gap verdict of the chosen precision
        fmt = "bf16" if e["precision"] == "f32" else e["precision"]
        e["gap_verdict"] = c["formats"][fmt]["verdict"]
        v = e["gap_verdict"]
        if v in ("missing", "shape_rejected") and not e["blocked_by"]:
            raise Fail(f"{e['class_name']}: precision {e['precision']} has no kernel ({v}) and no blocker is named")
        if e["blocked_by"]:
            # the fallback must be executable: the op's HOST kernel from the 4-1 inventory (libref), reading the weight at the stored precision; the precision the map would move to
            # once the operation grows a reader is recorded separately and is NOT a fallback
            hk = host_kernel(dispatch, c["op"])
            if not hk:
                raise Fail(f"{e['class_name']}: no host kernel for {c['op']} in the 4-1 inventory, so the blocked entry has no executable fallback")
            e["intended_precision"] = e.get("fallback") if e.get("fallback") not in (None, e["precision"]) else None
            e["blocking_op"] = c["op"]
            e["fallback"] = "host_libref"
            e["fallback_kernel"] = hk
        if v == "partial" and c["kind"] != "custom":
            miss = [b for b, rs in c["formats"][fmt]["bands"].items() if not rs]
            ok_rows = sorted({r for rs in c["formats"][fmt]["bands"].values() for r in rs})
            e["band_gap"] = f"no accepting row for M in {miss}: {c['formats'][fmt]['reason']}"
            n_, layers = c["N"], len(c.get("layers") or [])
            pad = -(-n_ // 128) * 128
            e["band_fallback"] = (f"pad the weight's N to the next multiple of 128 at convert ({n_} -> {pad}: {pad - n_} zero rows, +{(pad - n_) / n_ * 100:.0f}% of this matrix, "
                                  f"+{(pad - n_) * c['K'] * layers / 2 ** 20:.1f} MiB over its {layers} layers at int8, not in the memory plan) so the tiled prefill rows apply, or add an N%64 tile to libr4d; the rows {', '.join(ok_rows)} already serve the other bands")
        elif v == "partial":
            e["band_gap"] = f"rows are listed for {c['op']} but none is verified against this weight: {c['formats'][fmt]['reason']}"
            e["band_fallback"] = "keep the host fallback until a probe or test verifies the listed rows at this weight's encoding and geometry"
        entries.append(e)
    by_prec = {}
    for e in entries:
        by_prec.setdefault(e["precision"], {"classes": 0, "tensors": 0})
        by_prec[e["precision"]]["classes"] += 1
        by_prec[e["precision"]]["tensors"] += e["count"]
    blockers = sorted({e["blocked_by"] for e in entries if e["blocked_by"]})
    return dict(schema="ernie55-iqr-v4-precision-map/1", target_format="iqr_v4",
                source_format="block fp8 e4m3, 128x128 ue8m0 scales (conversion source only; not a Radiance 2.x serving format)",
                decision="operator decision 2026-11-23", recipe="data/recipes/ernie55-iqr-v4.recipe",
                entries=entries, summary=dict(by_precision=by_prec, blockers=blockers))


def render_md(m):
    L = ["# ERNIE 5.5 Flash precision map for the IQR-V4 target (story 5-0-iqr-v4-target)", "",
         "Generated by `tools/ernie55/iqr_precision.py` from the gap list and `data/recipes/ernie55-iqr-v4.recipe`. FP8 is the conversion source only.", "",
         "| group | class | shape | count | precision | kernel (gap verdict) | blocker |", "|---|---|---|---:|---|---|---|"]
    for e in m["entries"]:
        if e["group"] == "source_scale_plane":
            continue
        L.append(f"| {e['group']} | `{e['class_name']}` | {e['shape']} | {e['count']} | **{e['precision']}** | {e['gap_verdict']} | {e['blocked_by'] or '-'} |")
    L += ["", "## Why", ""]
    seen = set()
    for e in m["entries"]:
        key = (e["group"], e["precision"], e["justification"])
        if key in seen or e["group"] == "source_scale_plane":
            continue
        seen.add(key)
        L.append(f"- **{e['group']}** ({e['precision']}): {e['justification']} Cites: {', '.join(e['cites']) or '-'}")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build")
    a.add_argument("--dispatch", default="docs/ernie55/device-dispatch.json")
    a.add_argument("--gap", default="docs/ernie55/iqr-v4-gap.json"); a.add_argument("--recipe", default="data/recipes/ernie55-iqr-v4.recipe")
    a.add_argument("--out-json", required=True); a.add_argument("--out-md"); a.add_argument("--repo", default=".")
    b = sub.add_parser("check-recipe")
    b.add_argument("--recipe", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "check-recipe":
            errs = check_recipe(args.recipe)
            for e in errs:
                print(e)
            return 1 if errs else 0
        errs = check_recipe(args.recipe)
        if errs:
            raise Fail("recipe refused: " + "; ".join(errs))
        gap = json.loads(Path(args.gap).read_text())
        dispatch = json.loads(Path(args.dispatch).read_text())
        m = build(gap, args.recipe, args.repo, dispatch)
        Path(args.out_json).write_text(json.dumps(m, indent=1) + "\n")
        if args.out_md:
            Path(args.out_md).write_text(render_md(m))
    except (Fail, OSError, ValueError, KeyError) as e:
        print(f"iqr_precision: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
