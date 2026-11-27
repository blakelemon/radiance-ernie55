#!/usr/bin/env python3
"""attention_reference.py --metadata-dir DIR -- the ERNIE 5.5 attention reference report, offline.

    python3 tools/ernie55/attention_reference.py --metadata-dir /models/ERNIE-5.5-Flash-Base

Prints one deterministic JSON report (sections mla, hca, reuse, fixtures, plus pins, sources, config,
schedule and unresolved) for the two attention families of the pinned model: MLA-DSA (compress_ratios
entry -2) and HCA (ratio 128). Everything that depends on the model is computed here from the pinned
`config.json` in DIR: the per-entry family schedule, the trailing-window bounds, the HCA group
boundaries, the chunk-boundary table, the production boundary arithmetic and the derived constants.
Everything that depends on the upstream sources (Baidu vLLM-ERNIE b8414da, vLLM v0.32.0) or on the
Radiance sources is carried as DATA in attention_reference_facts.json: every fact has provenance
entries {pin, file, line, line_end, quote} (or, for checkpoint tensor geometry, a safetensors header
entry {pin: hf_headers, file, tensor, dtype, shape}); a fact that the pinned sources do not establish is
recorded with status `unresolved` and a resolver. The discriminating fixtures are computed here by a
small pure-Python reference of each equation and its tempting wrong variants.

    --sample-lengths a,b,...   sequence lengths for the window tables (default: the production boundaries)
    --chunks s:q,s:q,...       (start, query length) chunks for the group-completion table
    --verify-sources DIR       re-read every cited line from DIR/<pin>/<file> (pins baidu, vllm), from
                               `git show <commit>:<file>` (ernie_repo, in --repo-root) and from DIR/hf_config
                               and print a receipt; exit 1 if any citation, hash or quote does not hold
    --verify-headers MODEL_DIR re-read the cited safetensors headers (header bytes only) and the
                               per-family uniformity of the attention tensors

Valid input exits 0 with the JSON on stdout. Invalid input exits 2 with a message naming the key on
stderr and nothing on stdout. See docs/ernie55/attention-reference.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import sys

CONFIG_NAME = "config.json"
FACTS_NAME = "attention_reference_facts.json"
MODEL_TYPE = "ernie5_5_moe"
HCA_MODE = 128
MLA_DSA_MODE = -2
VHA_RANK = 16
DSA_PAGE_TOKENS = 64
SWA_PAGE_TOKENS = 64
C128A_TOPK_ALIGNMENT = 128


class RefError(Exception):
    """Input that cannot be audited; the message names the offending key or option."""


# ----------------------------------------------------------------------------- validation
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _int(cfg, key, lo=1, hi=1 << 40):
    v = cfg.get(key)
    if not _is_int(v) or v < lo or v > hi:
        raise RefError(f"{key} must be an integer in [{lo}, {hi}], got {v!r}")
    return v


def _num(cfg, key):
    v = cfg.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise RefError(f"{key} must be a positive finite number, got {v!r}")
    return v


def _canon(v):
    """Integral floats become ints so the report does not depend on how the JSON spelled 160000."""
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return int(v)
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_canon(x) for x in v]
    return v


def load_config(directory: pathlib.Path) -> dict:
    path = directory / CONFIG_NAME
    if not path.is_file():
        raise RefError(f"{CONFIG_NAME} is missing: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RefError(f"{CONFIG_NAME} cannot be read as JSON: {path}: {exc}")
    if not isinstance(cfg, dict):
        raise RefError(f"{CONFIG_NAME} is not a JSON object: {path}")
    cfg = _canon(cfg)
    if cfg.get("model_type") != MODEL_TYPE:
        raise RefError(f"model_type must be {MODEL_TYPE!r}, got {cfg.get('model_type')!r}")
    n = _int(cfg, "num_hidden_layers")
    d = _int(cfg, "num_nextn_predict_layers", lo=0)
    ratios = cfg.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) != n + d or any(
            not _is_int(r) or r not in (HCA_MODE, MLA_DSA_MODE) for r in ratios):
        raise RefError(
            f"compress_ratios must hold num_hidden_layers + num_nextn_predict_layers = {n + d} entries, "
            f"each {HCA_MODE} (HCA) or {MLA_DSA_MODE} (MLA-DSA); got {ratios!r}"[:300])
    for k in ("sliding_window", "index_topk", "index_n_heads", "index_head_dim", "hybrid_mla_num_attention_heads",
              "hybrid_mla_q_lora_rank", "hybrid_mla_kv_lora_rank", "hybrid_mla_qk_nope_head_dim",
              "hybrid_mla_v_head_dim", "head_dim", "num_attention_heads", "o_groups", "o_lora_rank", "max_position_embeddings"):
        _int(cfg, k)
    for k in ("hybrid_mla_qk_rope_head_dim", "qk_rope_head_dim"):
        v = _int(cfg, k)
        if v % 2:
            raise RefError(f"{k} must be even, got {v}")
    if cfg["qk_rope_head_dim"] >= cfg["head_dim"]:
        raise RefError(f"qk_rope_head_dim ({cfg['qk_rope_head_dim']}) must be smaller than head_dim ({cfg['head_dim']})")
    if (cfg["head_dim"] - cfg["qk_rope_head_dim"]) % 64:
        raise RefError("head_dim - qk_rope_head_dim must be a multiple of 64 (the FP8 nope quantisation block)")
    if cfg["num_attention_heads"] % cfg["o_groups"]:
        raise RefError("o_groups must divide num_attention_heads")
    for k in ("compress_rope_theta", "rope_theta", "rms_norm_eps"):
        _num(cfg, k)
    rs = cfg.get("rope_scaling")
    if not isinstance(rs, dict):
        raise RefError("rope_scaling must be an object")
    for k in ("original_max_position_embeddings", "beta_fast", "beta_slow"):
        v = rs.get(k)
        if not _is_int(v) or v < 1:
            raise RefError(f"rope_scaling.{k} must be a positive integer, got {v!r}")
    f = rs.get("factor")
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or f < 1:
        raise RefError(f"rope_scaling.factor must be a number >= 1, got {f!r}")
    return cfg


def parse_lengths(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not re.fullmatch(r"[0-9]+", part) or int(part) < 1 or int(part) > 1 << 40:
            raise RefError(f"--sample-lengths entries must be positive integers, got {part!r}")
        out.append(int(part))
    if not out:
        raise RefError("--sample-lengths is empty")
    return out


def parse_chunks(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        m = re.fullmatch(r"\s*([0-9]+):([0-9]+)\s*", part)
        if not m or int(m.group(2)) < 1 or int(m.group(1)) > 1 << 40 or int(m.group(2)) > 1 << 40:
            raise RefError(f"--chunks entries must be start:qlen with qlen >= 1, got {part!r}")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


# ----------------------------------------------------------------------------- tables
def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def next_pow2(n: int) -> int:
    return 1 << (max(n, 1) - 1).bit_length()


def schedule(cfg: dict) -> dict:
    n = cfg["num_hidden_layers"]
    entries = []
    for i, r in enumerate(cfg["compress_ratios"]):
        entries.append({"entry": i, "kind": "main" if i < n else "draft", "ratio": r,
                        "family": "mla_dsa" if r == MLA_DSA_MODE else "hca"})
    return {
        "main_layers": n,
        "draft_entries": [e["entry"] for e in entries if e["kind"] == "draft"],
        "num_entries": len(entries),
        "entries": entries,
        "mla_dsa_main_layers": [e["entry"] for e in entries if e["kind"] == "main" and e["family"] == "mla_dsa"],
        "hca_main_layers": [e["entry"] for e in entries if e["kind"] == "main" and e["family"] == "hca"],
        "rule": "family = mla_dsa when compress_ratios[entry] <= 0 else hca (Ernie5_5Config.is_dsa_layer); "
                "entries at or beyond num_hidden_layers are the draft (MTP) entries",
    }


def mla_row(n: int, window: int, topk: int) -> dict:
    hi = max(n - window, 0)
    sel = min(hi, topk)
    wc = min(n, window)
    return {"seq_len": n, "position": n - 1, "window_start": n - wc, "window_end_inclusive": n - 1,
            "window_count": wc, "indexer_lo": 0, "indexer_hi_exclusive": hi, "indexer_candidates": hi,
            "selected_count": sel, "merged_count": sel + wc}


def hca_row(n: int, window: int, ratio: int) -> dict:
    wc = min(n, window)
    ws = n - wc
    c = n // ratio
    return {"seq_len": n, "position": n - 1, "window_start": ws, "window_end_inclusive": n - 1, "window_count": wc,
            "compressed_count": c, "last_complete_group_end": c * ratio - 1 if c > 0 else None,
            "incomplete_start": c * ratio, "incomplete_tokens": n - c * ratio, "candidates": c + wc,
            "overlap_tokens": max(0, c * ratio - ws), "group_completes_here": n % ratio == 0}


def group_rows(ratio: int, max_compressed: int) -> list[dict]:
    js = sorted({j for j in (0, 1, 2, 3, max_compressed - 2, max_compressed - 1) if 0 <= j < max_compressed})
    return [{"group": j, "first_token": j * ratio, "last_token": j * ratio + ratio - 1,
             "completes_at_seq_len": j * ratio + ratio, "completes_at_position": j * ratio + ratio - 1,
             "rope_position": j * ratio} for j in js]


def chunk_row(start: int, qlen: int, ratio: int) -> dict:
    end = start + qlen
    before, after = start // ratio, end // ratio
    return {"start": start, "qlen": qlen, "end": end, "groups_before": before, "groups_after": after,
            "groups_completed": after - before, "boundary_flag": (start % ratio) + qlen >= ratio,
            "tail_state_tokens_after": end % ratio}


def default_lengths(cfg: dict, ratio: int | None) -> list[int]:
    w, k, m = cfg["sliding_window"], cfg["index_topk"], cfg["max_position_embeddings"]
    cand = {1, 2, w - 1, w, w + 1, w + 2, 2 * w, 3 * w, w + k - 1, w + k, w + k + 1, m - 1, m}
    if ratio:
        cand |= {ratio - 1, ratio, ratio + 1, 2 * ratio - 1, 2 * ratio, 2 * ratio + 1, 3 * ratio}
    return sorted(x for x in cand if 1 <= x <= m)


def default_chunks(ratio: int | None) -> list[tuple[int, int]]:
    r = ratio or HCA_MODE
    return [(0, 1), (0, r - 1), (0, r), (r - 1, 1), (max(r - 28, 0), 60), (r, r), (2 * r - 1, 2), (2 * r, 8 * r)]


def yarn_dim(num_rot: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def yarn_range(cfg: dict) -> tuple[int, int]:
    dim, rs = cfg["qk_rope_head_dim"], cfg["rope_scaling"]
    base, orig = cfg["compress_rope_theta"], rs["original_max_position_embeddings"]
    low = max(math.floor(yarn_dim(rs["beta_fast"], dim, base, orig)), 0)
    high = min(math.ceil(yarn_dim(rs["beta_slow"], dim, base, orig)), dim - 1)
    return int(low), int(high)


def mla_derived(cfg: dict) -> dict:
    nope, rope = cfg["hybrid_mla_qk_nope_head_dim"], cfg["hybrid_mla_qk_rope_head_dim"]
    qk = nope + rope
    lora = cfg["hybrid_mla_kv_lora_rank"]
    ih = cfg["index_head_dim"]
    return {
        "window": cfg["sliding_window"], "index_topk": cfg["index_topk"],
        "merged_row_width": cfg["index_topk"] + cfg["sliding_window"],
        "num_heads": cfg["hybrid_mla_num_attention_heads"], "q_lora_rank": cfg["hybrid_mla_q_lora_rank"],
        "kv_lora_rank": lora, "qk_nope_head_dim": nope, "qk_rope_head_dim": rope, "qk_head_dim": qk,
        "v_head_dim": cfg["hybrid_mla_v_head_dim"], "scale": qk ** -0.5,
        "kv_row_elements": lora + rope, "kv_row_bytes_bf16": 2 * (lora + rope),
        "kv_row_bytes_fp8_ds_mla": lora + 4 * cdiv(lora, 128) + 2 * rope,
        "kv_b_proj_rows": cfg["hybrid_mla_num_attention_heads"] * (nope + cfg["hybrid_mla_v_head_dim"]),
        "page_tokens": DSA_PAGE_TOKENS,
        "o_gate_width": cfg["hybrid_mla_num_attention_heads"] * cfg["hybrid_mla_v_head_dim"],
        "vha_rank": VHA_RANK,
        "rope": {"type": "default", "theta": cfg["rope_theta"], "pairing": "gptj_interleaved", "slice": "trailing",
                 "yarn": False},
        "indexer": {"n_heads": cfg["index_n_heads"], "head_dim": ih, "rope_dim": rope,
                    "weights_scale": ih ** -0.5 * cfg["index_n_heads"] ** -0.5,
                    "split_order": "rope_first", "rope_pairing": "neox_split_half", "rope_slice": "leading",
                    "k_norm": "layernorm_eps_1e-6_with_bias", "q_quant": "fp8_e4m3_ue8m0_group128",
                    "k_row_bytes_fp8": ih + 4 * cdiv(ih, 128), "compress_ratio": 1},
        "indexer_weights_scale": ih ** -0.5 * cfg["index_n_heads"] ** -0.5,
    }


def hca_derived(cfg: dict, ratio: int | None) -> dict:
    hd, rope = cfg["head_dim"], cfg["qk_rope_head_dim"]
    rs = cfg["rope_scaling"]
    low, high = yarn_range(cfg)
    nope = hd - rope
    return {
        "ratio": ratio, "window": cfg["sliding_window"], "head_dim": hd, "nope_dim": nope, "rope_dim": rope,
        "num_heads": cfg["num_attention_heads"], "scale": hd ** -0.5, "q_lora_rank": cfg["q_lora_rank"] if "q_lora_rank" in cfg else None,
        "rope": {"type": "deepseek_yarn", "theta": cfg["compress_rope_theta"], "factor": rs["factor"],
                 "beta_fast": rs["beta_fast"], "beta_slow": rs["beta_slow"],
                 "original_max_position_embeddings": rs["original_max_position_embeddings"],
                 "yarn_low": low, "yarn_high": high, "magnitude": 1,
                 "cos_sin_cache_rows": int(rs["original_max_position_embeddings"] * rs["factor"]),
                 "pairing": "gptj_interleaved", "slice": "trailing", "inverse_on_output": True},
        "compressed_entry_bytes_fp8_ds_mla": nope + 2 * rope + nope // 64 + 1,
        "compressed_entry_bytes_bf16": 2 * hd,
        "state_row_floats": 2 * hd, "state_row_bytes": 8 * hd,
        "state_rows_per_request_layer": ratio, "ape_shape": [ratio, hd] if ratio else None,
        "o_groups": cfg["o_groups"], "heads_per_group": cfg["num_attention_heads"] // cfg["o_groups"],
        "o_lora_rank": cfg["o_lora_rank"], "o_gate_width": cfg["o_groups"] * cfg["o_lora_rank"],
        "window_page_tokens": SWA_PAGE_TOKENS, "vha_rank": VHA_RANK,
    }


def mla_production(cfg: dict) -> dict:
    w, k, m = cfg["sliding_window"], cfg["index_topk"], cfg["max_position_embeddings"]
    return {"max_position_embeddings": m, "first_indexer_seq_len": w + 1, "first_topk_saturation_seq_len": w + k + 1,
            "merged_row_width": k + w, "indexer_candidates_at_max": max(m - w, 0),
            "selected_at_max": min(max(m - w, 0), k),
            "arithmetic": "candidates = max(seq - window, 0); selected = min(candidates, index_topk); row = selected + min(seq, window)"}


def hca_production(cfg: dict, ratio: int | None) -> dict:
    m = cfg["max_position_embeddings"]
    if not ratio:
        return {"max_position_embeddings": m}
    maxc = cdiv(m, ratio)
    width = cdiv(maxc, C128A_TOPK_ALIGNMENT) * C128A_TOPK_ALIGNMENT
    return {"max_position_embeddings": m, "max_compressed_entries": maxc, "c128a_topk_width": width,
            "first_compressed_seq_len": ratio, "first_full_window_seq_len": cfg["sliding_window"],
            "active_topk_width_at_max": min(max(next_pow2(max(m // ratio, 1)), C128A_TOPK_ALIGNMENT), width),
            "max_overlap_tokens": min(cfg["sliding_window"], ratio),
            "arithmetic": "compressed = seq // ratio; candidates = compressed + min(seq, window); "
                          "max compressed = cdiv(max_position_embeddings, ratio)"}


# ----------------------------------------------------------------------------- fixtures
def _round(x):
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, float):
        return round(x, 12)
    return x


def _flat(x):
    if isinstance(x, (list, tuple)):
        return [v for e in x for v in _flat(e)]
    return [x]


def _maxdiff(a, b):
    fa, fb = _flat(a), _flat(b)
    if len(fa) != len(fb):
        return float("inf")
    return max((abs(p - q) for p, q in zip(fa, fb)), default=0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _select_top(scores, hi, k):
    ids = list(range(max(hi, 0)))
    ids.sort(key=lambda i: (-scores[i], i))
    return ids[:k]


def _win(n, w, shift=0):
    start, end = max(n - w - shift, 0), n - 1 - shift
    return list(range(start, end + 1))


def _softmax_pool(scores, vals):
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    z = sum(e)
    return [sum(e[i] / z * vals[i][c] for i in range(len(vals))) for c in range(len(vals[0]))]


def _group_pool(inp, mode="channel", ape=True):
    ratio, kv, score, apev = inp["ratio"], inp["kv"], inp["score"], inp["ape"]
    n = len(kv)
    groups = n // ratio
    out = []
    for j in range(groups):
        idx = list(range(j * ratio, min(j * ratio + ratio, n)))
        chans = len(kv[0])
        sc = [[score[t][c] + (apev[t % ratio][c] if ape else 0.0) for c in range(chans)] for t in idx]
        if mode == "mean":
            out.append([sum(kv[t][c] for t in idx) / len(idx) for c in range(chans)])
        elif mode == "row_scalar":
            rs = [sum(r) / len(r) for r in sc]
            out.append(_softmax_pool(rs, [kv[t] for t in idx]))
        else:
            out.append([_softmax_pool([sc[i][c] for i in range(len(idx))], [[kv[t][c]] for t in idx])[0] for c in range(chans)])
    return out


def _rot(vec, lo, rd, pos, theta, mode):
    out = list(vec)
    half = rd // 2
    for k in range(half):
        ang = pos * theta ** (-2.0 * k / rd)
        c, s = math.cos(ang), math.sin(ang)
        i0 = lo + 2 * k if mode == "gptj" else lo + k
        i1 = lo + 2 * k + 1 if mode == "gptj" else lo + k + half
        out[i0] = vec[i0] * c - vec[i1] * s
        out[i1] = vec[i0] * s + vec[i1] * c
    return out


def _attend(logits, vals, sink=None):
    m = max(list(logits) + ([sink] if sink is not None else []))
    e = [math.exp(x - m) for x in logits]
    z = sum(e) + (math.exp(sink - m) if sink is not None else 0.0)
    return [sum(e[i] * vals[i][c] for i in range(len(vals))) / z for c in range(len(vals[0]))]


def _contrast(name, description, result, correct):
    d = _maxdiff(result, correct)
    return {"name": name, "description": description, "result": _round(result),
            "max_abs_diff": _round(d) if math.isfinite(d) else None, "differs": bool(d > 1e-3)}


def build_fixtures(cfg: dict, ratio: int | None) -> list[dict]:
    fixtures = []

    # -- DSA window merge ---------------------------------------------------------
    W, K = 4, 3
    seq_lens = [3, 4, 5, 7, 8, 11]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.95, 0.5, 0.4, 0.99]
    exp, c1, c2, c3 = [], [], [], []
    for n in seq_lens:
        sel, win = _select_top(scores, n - W, K), _win(n, W)
        exp.append({"seq_len": n, "selected": sel, "window": win, "row": sel + win})
        c1.append({"seq_len": n, "window": _win(n, W, 1)})
        c2.append({"seq_len": n, "selected": _select_top(scores, n - W + 1, K)})
        c3.append({"seq_len": n, "window": _win(n, W + 1)})
    def contrast_ids(name, desc, res, key):
        at = [e["seq_len"] for e, r in zip(exp, res) if e[key] != r[key]]
        return {"name": name, "description": desc, "result": res, "differs_at": at, "differs": bool(at)}
    fixtures.append({
        "id": "dsa-window-merge", "family": "mla_dsa", "kind": "window-merge",
        "purpose": "merge_swa_topk row: indexer ids from [0, seq - window) by score, then the inclusive trailing window seq-window .. seq-1 (reduced window 4, top-k 3, same equations as production)",
        "inputs": {"window": W, "topk": K, "seq_lens": seq_lens, "scores": scores,
                   "tie_rule": "fixture only: descending score, ties to the lower id (the pinned kernel's tie order is unresolved, m06)"},
        "expected": exp,
        "contrasts": [
            contrast_ids("window_excludes_current", "window shifted one back: positions p-W .. p-1 (query token left out)", c1, "window"),
            contrast_ids("indexer_overlaps_window", "indexer range [0, seq - window + 1): one token too many, duplicating the first window token", c2, "selected"),
            contrast_ids("window_one_wider", "window of window + 1 tokens (an exclusive-endpoint slip)", c3, "window")],
        "facts": ["m03", "m04", "m05"]})

    # -- HCA group pool ----------------------------------------------------------
    gp = {"ratio": 4, "channels": 2,
          "kv": [[1.0, -0.5], [0.2, 0.7], [-0.3, 1.1], [0.9, 0.4], [0.5, -1.2], [1.4, 0.3], [-0.8, 0.6], [0.1, 0.9], [2.0, -2.0], [-1.0, 1.5]],
          "score": [[0.3, -0.2], [0.1, 0.6], [-0.4, 0.2], [0.5, -0.1], [0.2, 0.4], [-0.3, 0.8], [0.6, -0.5], [0.0, 0.3], [0.7, 0.1], [-0.2, 0.9]],
          "ape": [[0.1, -0.1], [0.0, 0.2], [-0.2, 0.1], [0.3, 0.0]],
          "reduction": "ratio 4 stands for the production ratio 128; seq_len 10 = two complete groups + an incomplete group of 2"}
    n = len(gp["kv"])
    entries = _group_pool(gp)
    base_expected = {"complete_groups": n // gp["ratio"], "visible_entries": n // gp["ratio"], "entries": _round(entries),
                     "incomplete_tokens": list(range((n // gp["ratio"]) * gp["ratio"], n))}
    ceil_count = cdiv(n, gp["ratio"])
    fixtures.append({
        "id": "hca-group-pool", "family": "hca", "kind": "group-pool",
        "purpose": "compressed entry of each COMPLETE group: per-channel softmax over (score + APE[position % ratio]) of the group's rows, softmax-weighted sum of kv; the incomplete trailing group yields no entry",
        "inputs": gp, "expected": base_expected,
        "contrasts": [
            {"name": "ceil_count_includes_partial", "description": "counts the incomplete group as an entry",
             "result": {"count": ceil_count}, "differs": ceil_count != base_expected["complete_groups"]},
            _contrast("mean_pool", "plain mean of the group's kv rows", _group_pool(gp, mode="mean"), entries),
            _contrast("row_scalar_softmax", "one scalar score per row (ref_seg_softmax_pool style) instead of one per channel", _group_pool(gp, mode="row_scalar"), entries),
            _contrast("no_ape", "score without the position-dependent APE", _group_pool(gp, ape=False), entries)],
        "facts": ["h05", "h06", "h07"]})

    # -- sink ---------------------------------------------------------------------
    s = {"logits": [1.0, 0.5, -0.3], "values": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "sink": 0.7}
    good = _attend(s["logits"], s["values"], s["sink"])
    m = max(s["logits"] + [s["sink"]])
    e = [math.exp(x - m) for x in s["logits"]]
    in_max_only = [sum(e[i] * s["values"][i][c] for i in range(3)) / sum(e) for c in range(2)]
    fixtures.append({
        "id": "sink-denominator", "family": "both", "kind": "sink",
        "purpose": "the per-head sink is one extra logit with no value: it can raise the running max and adds exp(sink - max) to the denominator only",
        "inputs": s, "expected": _round(good),
        "contrasts": [
            _contrast("no_sink", "softmax without the sink", _attend(s["logits"], s["values"]), good),
            _contrast("sink_carries_first_value", "sink treated as an extra key whose value is the first key's value", _attend(s["logits"] + [s["sink"]], s["values"] + [s["values"][0]]), good),
            _contrast("sink_in_max_only", "sink raises the max but is left out of the denominator", in_max_only, good)],
        "facts": ["m15", "h17"]})

    # -- gate / VHA order ------------------------------------------------------------
    g = {"heads": 2, "dim": 2, "o": [[1.0, 2.0], [3.0, 4.0]], "m": [[1.1, 0.2], [-0.1, 0.9]], "gate_logits": [[0.5, -0.5], [1.0, -1.0]]}
    def mix(o):
        return [[sum(g["m"][h][j] * o[j][d] for j in range(2)) for d in range(2)] for h in range(2)]
    def gate(o, gl):
        return [[o[h][d] * _sigmoid(gl[h][d]) for d in range(2)] for h in range(2)]
    good = gate(mix(g["o"]), g["gate_logits"])
    fixtures.append({
        "id": "gate-vha-order", "family": "both", "kind": "gate-vha",
        "purpose": "attention output -> VHA head mix m = I + V U^T -> element sigmoid gate (the gate multiplies after the mix); on HCA the same order applies with the gate on the wo_a output",
        "inputs": g, "expected": _round(good),
        "contrasts": [
            _contrast("gate_before_vha", "gate applied to the heads first, then mixed", mix(gate(g["o"], g["gate_logits"])), good),
            _contrast("per_head_scalar_gate", "one gate logit per head (first element reused) as ref_mla_attn_paged / ref_head_gate do", gate(mix(g["o"]), [[r[0], r[0]] for r in g["gate_logits"]]), good)],
        "facts": ["m16", "m17", "h16", "h18"]})

    # -- HCA union softmax -------------------------------------------------------------
    u = {"window_logits": [0.4, 1.1, -0.2], "window_values": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
         "comp_logits": [0.9, -0.6], "comp_values": [[2.0, 0.0], [0.0, 2.0]], "sink": 0.3}
    good = _attend(u["window_logits"] + u["comp_logits"], u["window_values"] + u["comp_values"], u["sink"])
    ow = _attend(u["window_logits"], u["window_values"], u["sink"])
    oc = _attend(u["comp_logits"], u["comp_values"], u["sink"])
    fixtures.append({
        "id": "hca-union-softmax", "family": "hca", "kind": "union-softmax",
        "purpose": "one softmax (with the sink) over the window keys and the complete compressed keys together; two separate attentions added or averaged are not the same function",
        "inputs": u, "expected": _round(good),
        "contrasts": [
            _contrast("separate_then_sum", "window attention and compressed attention each normalised, outputs added", [a + b for a, b in zip(ow, oc)], good),
            _contrast("separate_then_mean", "the same two outputs averaged", [(a + b) / 2 for a, b in zip(ow, oc)], good)],
        "facts": ["h04", "h08"]})

    # -- rope pairing ----------------------------------------------------------------------
    rp = {"vector": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "rope_dim": 4, "position": 5, "theta": 10000.0}
    v, rd, pos, th = rp["vector"], rp["rope_dim"], rp["position"], rp["theta"]
    n_ = len(v)
    lead = _rot(v, 0, rd, pos, th, "neox")
    tail = _rot(v, n_ - rd, rd, pos, th, "gptj")
    fixtures.append({
        "id": "rope-pairing", "family": "both", "kind": "rope-split",
        "purpose": "the indexer rotates the LEADING rope dims split-half (NeoX); the DSA main path and HCA rotate the TRAILING rope dims interleaved (GPT-J); rotating everything or mixing the pairings gives different vectors",
        "inputs": rp, "expected": {"indexer_lead_neox": _round(lead), "main_tail_gptj": _round(tail)},
        "contrasts": [
            _contrast("tail_neox_pairing", "trailing slice with split-half pairing", _rot(v, n_ - rd, rd, pos, th, "neox"), tail),
            _contrast("lead_gptj_pairing", "leading slice with interleaved pairing", _rot(v, 0, rd, pos, th, "gptj"), lead),
            _contrast("rope_everywhere_gptj", "all dims rotated interleaved (no partial rope)", _rot(v, 0, n_, pos, th, "gptj"), tail)],
        "facts": ["m08", "m09", "h10"]})

    # -- HCA compressed rope position ------------------------------------------------------------
    rg = {"vector": [0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 1.0, 2.0], "rope_dim": 4, "ratio": 4, "last_position": 7, "theta": 160000.0}
    start = (rg["last_position"] // rg["ratio"]) * rg["ratio"]
    nv = len(rg["vector"])
    good = _rot(rg["vector"], nv - rg["rope_dim"], rg["rope_dim"], start, rg["theta"], "gptj")
    fixtures.append({
        "id": "hca-compressed-rope-position", "family": "hca", "kind": "rope-position",
        "purpose": "a compressed entry completed at position 7 (group 1, ratio 4) is rotated at its group START (4), not at the completing token (7) nor the next group's start (8)",
        "inputs": rg, "expected": {"rope_position": start, "rotated": _round(good)},
        "contrasts": [
            _contrast("group_end_position", "rotated at the position of the completing token", _rot(rg["vector"], nv - rg["rope_dim"], rg["rope_dim"], rg["last_position"], rg["theta"], "gptj"), good),
            _contrast("next_group_start_position", "rotated at the next group's start", _rot(rg["vector"], nv - rg["rope_dim"], rg["rope_dim"], start + rg["ratio"], rg["theta"], "gptj"), good)],
        "facts": ["h22", "h05"]})

    # -- shape fixtures -----------------------------------------------------------------------
    nope, vd, heads = cfg["hybrid_mla_qk_nope_head_dim"], cfg["hybrid_mla_v_head_dim"], cfg["hybrid_mla_num_attention_heads"]
    rows = heads * (nope + vd)
    legal = None
    for d in range(1, 4097):
        if heads * 2 * d == rows and d == nope and d == vd:
            legal = d
    fixtures.append({
        "id": "kv-b-unequal-widths", "family": "mla_dsa", "kind": "shape",
        "purpose": "ref_mla_project needs w.shape[0] == heads * 2 * d with one width d for keys and values; the pinned kv_b_proj has heads * (k_dim + v_dim) rows with k_dim != v_dim",
        "inputs": {"heads": heads, "k_dim": nope, "v_dim": vd, "kv_lora": cfg["hybrid_mla_kv_lora_rank"]},
        "expected": {"baidu_kv_b_rows": rows, "ref_mla_project_rows_if_d_is_k": heads * 2 * nope,
                     "ref_mla_project_rows_if_d_is_v": heads * 2 * vd, "legal_d": legal},
        "contrasts": [
            {"name": "d_is_k", "description": "d set to the key width", "result": {"rows": heads * 2 * nope}, "differs": heads * 2 * nope != rows},
            {"name": "d_is_v", "description": "d set to the value width", "result": {"rows": heads * 2 * vd}, "differs": heads * 2 * vd != rows}],
        "facts": ["m12"]})
    fixtures.append({
        "id": "glm-geometry-vs-ernie", "family": "mla_dsa", "kind": "shape",
        "purpose": "GLM's indexer block fixes top-k 512 over pooled ratio-4 pages without rope or window; ERNIE selects index_topk tokens (ratio 1) after rope-first NeoX rotation, excluding the trailing window",
        "inputs": {"index_topk": cfg["index_topk"], "sliding_window": cfg["sliding_window"]},
        "expected": {"ernie_topk": cfg["index_topk"], "glm_topk": 512, "ernie_index_ratio": 1, "glm_index_ratio": 4,
                     "ernie_window_exclusion": cfg["sliding_window"], "glm_window_exclusion": 0},
        "contrasts": [{"name": "glm_values", "description": "GLM block constants", "result": {"topk": 512, "ratio": 4, "window_exclusion": 0}, "differs": True}],
        "facts": ["m04", "m05", "m09"]})
    pool_ratio, pool_n = HCA_MODE, cfg["head_dim"]
    fixtures.append({
        "id": "compressor-geometry-limits", "family": "hca", "kind": "shape",
        "purpose": "ref_dsa_pool refuses ratio > 16 and n that is not a power of two <= 256; the HCA compressor has ratio 128 and head_dim 512",
        "inputs": {"ratio": pool_ratio, "n": pool_n},
        "expected": {"ref_dsa_pool_max_ratio": 16, "ref_dsa_pool_max_n": 256,
                     "ref_dsa_pool_accepts_ratio": 2 <= pool_ratio <= 16,
                     "ref_dsa_pool_accepts_n": 1 <= pool_n <= 256 and (pool_n & (pool_n - 1)) == 0},
        "contrasts": [{"name": "ref_limits", "description": "limits of ref_dsa_pool", "result": {"ratio_max": 16, "n_max": 256},
                       "differs": not (2 <= pool_ratio <= 16 and 1 <= pool_n <= 256 and (pool_n & (pool_n - 1)) == 0)}],
        "facts": ["h05", "h06"]})
    for f in fixtures:
        for c in f["contrasts"]:
            if not c["differs"]:
                raise AssertionError(f"fixture {f['id']} contrast {c['name']} does not discriminate")
    return fixtures


# ----------------------------------------------------------------------------- report
def build_report(cfg: dict, lengths: list[int] | None, chunks: list[tuple[int, int]] | None) -> dict:
    here = pathlib.Path(__file__).resolve().parent
    facts = json.loads((here / FACTS_NAME).read_text(encoding="utf-8"))
    sched = schedule(cfg)
    hca_entries = sched["hca_main_layers"]
    ratio = cfg["compress_ratios"][hca_entries[0]] if hca_entries else None
    w, k = cfg["sliding_window"], cfg["index_topk"]
    lens = lengths if lengths is not None else default_lengths(cfg, ratio)
    chs = chunks if chunks is not None else default_chunks(ratio)
    entries = sched["entries"]
    mla = {
        "layers": sched["mla_dsa_main_layers"],
        "draft_entries": [e["entry"] for e in entries if e["kind"] == "draft" and e["family"] == "mla_dsa"],
        "facts": facts["mla_facts"],
        "derived": mla_derived(cfg),
        "window_table": [mla_row(n, w, k) for n in lens],
        "production": mla_production(cfg),
    }
    hca = {
        "layers": hca_entries,
        "draft_entries": [e["entry"] for e in entries if e["kind"] == "draft" and e["family"] == "hca"],
        "facts": facts["hca_facts"],
        "derived": hca_derived(cfg, ratio),
        "window_table": [hca_row(n, w, ratio) for n in lens] if ratio else [],
        "group_table": group_rows(ratio, cdiv(cfg["max_position_embeddings"], ratio)) if ratio else [],
        "chunk_table": [chunk_row(s, q, ratio) for s, q in chs] if ratio else [],
        "production": hca_production(cfg, ratio),
    }
    keys = ["num_hidden_layers", "num_nextn_predict_layers", "compress_ratios", "sliding_window", "index_topk",
            "index_n_heads", "index_head_dim", "max_position_embeddings", "head_dim", "qk_rope_head_dim",
            "num_attention_heads", "o_groups", "o_lora_rank", "hybrid_mla_num_attention_heads",
            "hybrid_mla_q_lora_rank", "hybrid_mla_kv_lora_rank", "hybrid_mla_qk_nope_head_dim",
            "hybrid_mla_qk_rope_head_dim", "hybrid_mla_v_head_dim", "compress_rope_theta", "rope_theta",
            "rope_scaling", "rms_norm_eps"]
    return {
        "report": "ernie55-attention-reference",
        "schema_version": 1,
        "pins": facts["pins"],
        "sources": facts["sources"],
        "config": {k2: cfg[k2] for k2 in keys},
        "schedule": sched,
        "mla": mla,
        "hca": hca,
        "reuse": facts["reuse"],
        "fixtures": build_fixtures(cfg, ratio),
        "unresolved": facts["unresolved"],
        "header_uniformity": facts["header_uniformity"],
    }


# ----------------------------------------------------------------------------- verification
def _all_provenance(report: dict):
    for fam in ("mla", "hca"):
        for f in report[fam]["facts"]:
            for p in f.get("provenance", []):
                yield f["id"], p
    for r in report["reuse"]:
        if r.get("candidate"):
            yield r["id"] + "/candidate", r["candidate"]
        for p in r.get("radiance_evidence", []) + r.get("upstream", []):
            yield r["id"], p


def verify_sources(report: dict, tree: pathlib.Path, repo_root: pathlib.Path, config_path: pathlib.Path | None) -> dict:
    shas = {(s["pin"], s["file"]): s.get("sha256") for s in report["sources"]}
    cache: dict = {}
    failures, checked = [], 0

    def text_of(pin, file):
        key = (pin, file)
        if key in cache:
            return cache[key]
        if pin == "ernie_repo":
            data = subprocess.run(["git", "-C", str(repo_root), "show", f"{report['pins']['ernie_repo']['commit']}:{file}"],
                                  capture_output=True, check=True).stdout
        elif pin == "hf_config":
            data = (config_path or tree / "hf_config" / file).read_bytes()
        else:
            data = (tree / pin / file).read_bytes()
        cache[key] = data
        return data

    for owner, p in _all_provenance(report):
        if p["pin"] == "hf_headers":
            continue
        checked += 1
        try:
            data = text_of(p["pin"], p["file"])
        except (OSError, subprocess.CalledProcessError) as exc:
            failures.append({"fact": owner, "citation": f"{p['pin']}:{p['file']}", "problem": f"cannot read: {exc}"})
            continue
        want = shas.get((p["pin"], p["file"]))
        if want and hashlib.sha256(data).hexdigest() != want:
            failures.append({"fact": owner, "citation": f"{p['pin']}:{p['file']}", "problem": "sha256 differs from the pinned content"})
            continue
        lines = data.decode("utf-8").split("\n")
        span = "\n".join(lines[p["line"] - 1:p["line_end"]])
        if p["quote"] not in span:
            failures.append({"fact": owner, "citation": f"{p['pin']}:{p['file']}:{p['line']}-{p['line_end']}",
                             "problem": "quote not found in the cited lines", "quote": p["quote"]})
    return {"verified": "sources", "citations_checked": checked, "files": len(cache), "failures": failures}


def _header(model_dir: pathlib.Path, shard: str) -> dict:
    with open(model_dir / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def verify_headers(report: dict, model_dir: pathlib.Path) -> dict:
    failures, checked, cache = [], 0, {}
    for owner, p in _all_provenance(report):
        if p["pin"] != "hf_headers":
            continue
        checked += 1
        shard = p["file"]
        if shard not in cache:
            cache[shard] = _header(model_dir, shard)
        t = cache[shard].get(p["tensor"])
        if t is None or t["dtype"] != p["dtype"] or t["shape"] != p["shape"]:
            failures.append({"fact": owner, "tensor": p["tensor"], "problem": f"header says {t and (t['dtype'], t['shape'])}"})
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    sigs = {}
    for name, shard in index.items():
        m = re.match(r"model\.layers\.(\d+)\.((self_attn|input_layernorm)\..*)$", name)
        if not m:
            continue
        if shard not in cache:
            cache[shard] = _header(model_dir, shard)
        t = cache[shard][name]
        sigs.setdefault(int(m.group(1)), {})[m.group(2)] = (t["dtype"], tuple(t["shape"]))
    sched = report["schedule"]
    for fam, ids in (("hca", sched["hca_main_layers"]),
                     ("mla_dsa", [e["entry"] for e in sched["entries"] if e["family"] == "mla_dsa"])):
        for i in ids:
            if sigs.get(i) != sigs.get(ids[0]):
                failures.append({"family": fam, "entry": i, "problem": "attention tensor signature differs from the family's first entry"})
    return {"verified": "headers", "citations_checked": checked, "shards_read": len(cache),
            "entries_compared": sum(len(v) for v in (sched["hca_main_layers"],)) + sum(1 for e in sched["entries"] if e["family"] == "mla_dsa"),
            "failures": failures}


# ----------------------------------------------------------------------------- main

def dumps(o, ind=0):
    """Deterministic, reviewable JSON: citations and short scalar lists stay on one line."""
    pad = " " * ind
    flat = lambda v: json.dumps(v, separators=(", ", ": "))
    if isinstance(o, dict):
        if not o:
            return "{}"
        if "pin" in o and ("quote" in o or "tensor" in o):
            return flat(o)
        items = [f"{pad} {json.dumps(k)}: {dumps(v, ind + 1)}" for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(o, list):
        if not o:
            return "[]"
        if len(flat(o)) <= 110 and not any(isinstance(x, dict) for x in o):
            return flat(o)
        items = [f"{pad} {dumps(v, ind + 1)}" for v in o]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return flat(o)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ERNIE 5.5 attention reference report (offline).")
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--sample-lengths")
    ap.add_argument("--chunks")
    ap.add_argument("--verify-sources")
    ap.add_argument("--verify-headers")
    ap.add_argument("--repo-root", default=".")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 2 if exc.code else 0
    try:
        cfg = load_config(pathlib.Path(args.metadata_dir))
        lengths = parse_lengths(args.sample_lengths) if args.sample_lengths else None
        chunks = parse_chunks(args.chunks) if args.chunks else None
        report = build_report(cfg, lengths, chunks)
    except RefError as exc:
        print(f"attention_reference: {exc}", file=sys.stderr)
        return 2
    if args.verify_sources or args.verify_headers:
        receipts = []
        if args.verify_sources:
            receipts.append(verify_sources(report, pathlib.Path(args.verify_sources), pathlib.Path(args.repo_root),
                                           pathlib.Path(args.metadata_dir) / CONFIG_NAME))
        if args.verify_headers:
            receipts.append(verify_headers(report, pathlib.Path(args.verify_headers)))
        sys.stdout.write(dumps(receipts) + "\n")
        return 1 if any(r["failures"] for r in receipts) else 0
    sys.stdout.write(dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
