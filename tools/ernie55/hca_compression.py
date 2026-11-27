#!/usr/bin/env python3
"""hca_compression.py --metadata-dir DIR -- the ERNIE 5.5 HCA compressor state and index-visibility report, offline.

    python3 tools/ernie55/hca_compression.py --metadata-dir /models/ERNIE-5.5-Flash-Base

Prints one deterministic JSON report for the HCA compressor of the pinned model: the legal compression periods
(128 production, 4 the only other legal one, with its overlap lanes), the compressor state, group completion,
incomplete-group masking, absolute group anchoring, the dtype and rounding of every stage, the cache layouts, the
traced cache-state API, the implementation code map for a later op story, an independent scalar fixture run and what
the pinned sources do not establish. Everything that depends on the model is computed here from the pinned
`config.json` in DIR (state geometry, visibility, group and chunk tables, overlap lane tables). Everything that
depends on the upstream sources (Baidu vLLM-ERNIE b8414da, vLLM v0.32.0) or on the repository is DATA in
hca_compression_facts.json: every fact carries provenance entries {pin, file, line, line_end, quote} (or a
safetensors header entry {pin: hf_headers, file, tensor, dtype, shape}); a fact the sources do not establish has
status `unresolved` and a resolver. The fixtures are produced by a small scalar reference of the compressor
(explicit fp32 / bf16 / FP8 rounding stages) with a seeded input generator.

    --positions a,b,...      positions for the visibility tables (default: group boundaries of each period)
    --chunks s:q,s:q,...     (start, query length) chunks for the chunk table (default: boundary-crossing chunks)
    --verify-sources DIR     re-read every cited line from DIR/<pin>/<file> (pins baidu, vllm), from `git show
                             <commit>:<file>` (ernie_repo, in --repo-root) and from DIR/hf_config or the config in
                             --metadata-dir; print a receipt and exit 1 if a citation, hash or quote does not hold
    --verify-headers MODEL   re-read the cited safetensors headers and check ALL compressor tensors of the checkpoint

Valid input exits 0 with the JSON on stdout. Invalid input exits 2 with a message naming the key or option on stderr
and nothing on stdout. See docs/ernie55/hca-compression.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import attention_reference as ar  # noqa: E402  (config validation and report printer of the 3-7 audit)

RefError = ar.RefError
FACTS_NAME = "hca_compression_facts.json"
LEGAL_RATIOS = (4, 128)
FP8_MAX = 448.0
QUANT_EPS = 1e-4
QUANT_BLOCK = 64


# ----------------------------------------------------------------------------- input validation
def load_config(directory: pathlib.Path) -> dict:
    cfg = ar.load_config(directory)
    ar._int(cfg, "hidden_size")
    return cfg


def parse_positions(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not re.fullmatch(r"\d+", part):
            raise RefError(f"--positions entries must be non-negative integers, got {part!r}")
        out.append(int(part))
    if not out:
        raise RefError("--positions is empty")
    return sorted(set(out))


def parse_chunks(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        m = re.fullmatch(r"\s*(\d+):(\d+)\s*", part)
        if not m or int(m.group(2)) < 1:
            raise RefError(f"--chunks entries must be start:qlen with qlen >= 1, got {part!r}")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ----------------------------------------------------------------------------- tables from the config
def coff_of(ratio: int) -> int:
    return 1 + (ratio == 4)


def periods(cfg: dict) -> list[dict]:
    D, hidden, rope = cfg["head_dim"], cfg["hidden_size"], cfg["qk_rope_head_dim"]
    nope = D - rope
    used = {r for r in cfg["compress_ratios"] if r in LEGAL_RATIOS}
    out = []
    for r in LEGAL_RATIOS:
        coff = coff_of(r)
        W = coff * D
        if coff == 1:
            lanes = [{"lane": "only", "window_index_range": [0, r - 1], "source": "current group", "channels": [0, D]}]
        else:
            lanes = [{"lane": "previous", "window_index_range": [0, r - 1], "source": "previous group", "channels": [0, D]},
                     {"lane": "current", "window_index_range": [r, 2 * r - 1], "source": "current group", "channels": [D, 2 * D]}]
        out.append({
            "ratio": r, "role": "production" if r == 128 else "small legal period",
            "used_by_checkpoint": r in used, "overlap": coff == 2, "coff": coff, "indexer": r == 4,
            "window_rows": coff * r, "state_block_rows": 4 if r == 4 else 8,
            "state_width": W, "state_row_floats": 2 * W, "state_row_bytes": 8 * W,
            "ape_shape": [r, W], "projection_out_widths": [W, W], "projection_in_width": hidden,
            "lanes": lanes, "first_group_masked_rows": r if coff == 2 else 0,
            "compressed_row_bytes": {"fp8_ds_mla": nope + 2 * rope + (nope // QUANT_BLOCK + 1),
                                     "bf16_full": 2 * D, "fp8_full_per_tensor": D},
            "state_retention_rows": coff * r,
        })
    return out


def default_positions(ratio: int, maxpos: int) -> list[int]:
    s = {0, 1, ratio - 2, ratio - 1, ratio, ratio + 1, 2 * ratio - 2, 2 * ratio - 1, 2 * ratio, 2 * ratio + 1, 3 * ratio - 1,
         3 * ratio, 4 * ratio - 1, 4 * ratio, maxpos - ratio - 1, maxpos - ratio, maxpos - 1}
    return sorted(p for p in s if 0 <= p < maxpos)


def default_chunks(ratio: int) -> list[tuple[int, int]]:
    if ratio == 128:
        return [(0, 1), (0, 127), (0, 128), (127, 1), (100, 60), (128, 128), (255, 2), (256, 1024)]
    return [(0, 1), (0, 3), (0, 4), (3, 1), (2, 6), (4, 4), (7, 2), (8, 16), (5, 40)]


def vis_row(p: int, r: int) -> dict:
    n = p + 1
    c = n // r
    return {"position": p, "seq_len": n, "visible_groups": c, "newest_visible_group": c - 1 if c else None,
            "newest_visible_group_last_position": c * r - 1 if c else None, "incomplete_start": c * r,
            "incomplete_tokens": n - c * r, "completes_here": n % r == 0}


def group_rows(r: int, maxpos: int) -> list[dict]:
    maxc = cdiv(maxpos, r)
    js = sorted({j for j in (0, 1, 2, 3, maxc - 1) if 0 <= j < maxc})
    return [{"group": j, "first_token": j * r, "last_token": j * r + r - 1, "completes_at_position": j * r + r - 1,
             "visible_from_seq_len": j * r + r, "anchor_position": j * r, "ape_rows": [0, r - 1]} for j in js]


def chunk_row(start: int, q: int, r: int) -> dict:
    W = coff_of(r) * r
    end = start + q
    first = next((p for p in range(start, end) if (p + 1) % r == 0), None)
    count = end // r - start // r
    earlier = 0 if first is None else max(0, start - max(0, first - W + 1))
    return {"start": start, "qlen": q, "end": end, "groups_before": start // r, "groups_after": end // r,
            "groups_completed": count, "boundary_flag": (start % r) + q >= r, "first_completing_position": first,
            "state_rows_from_earlier_chunks": earlier, "tail_tokens_after": end % r, "state_rows_needed_after": min(end, W - 1)}


def overlap_rows(r: int, D: int) -> list[dict]:
    """Per completing token of groups 0..3 at period 4, the window rows with their lane; empty at period 128."""
    if coff_of(r) == 1:
        return []
    out = []
    for g in range(4):
        p = g * r + r - 1
        rows = []
        for w in range(2 * r):
            pos = p - 2 * r + 1 + w
            lane_cur = w >= r
            rows.append({"w": w, "position": pos, "source_group": pos // r if pos >= 0 else None,
                         "lane": "current" if lane_cur else "previous", "channels": [D, 2 * D] if lane_cur else [0, D],
                         "masked": pos < 0})
        out.append({"group": g, "completing_position": p, "rows": rows})
    return out


# ----------------------------------------------------------------------------- scalar reference
def fround(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x: float) -> float:
    b = struct.unpack("<I", struct.pack("<f", fround(x)))[0]
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def _fp8_table() -> list[tuple[int, float]]:
    t = []
    for code in range(0x7F):  # 0x7F is NaN in e4m3fn
        e, m = code >> 3, code & 7
        t.append((code, (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)))
    return t


FP8_TABLE = _fp8_table()


def fp8_encode(y: float) -> int:
    """float -> E4M3FN code, clamped to +-448, round to nearest, ties to the even code; subnormals kept."""
    y = max(-FP8_MAX, min(FP8_MAX, y))
    sign = 0x80 if (y < 0 or (y == 0 and math.copysign(1.0, y) < 0)) else 0
    a = abs(y)
    best, best_d = 0, None
    for code, v in FP8_TABLE:
        d = abs(v - a)
        if best_d is None or d < best_d or (d == best_d and code % 2 == 0 and best % 2 == 1):
            best, best_d = code, d
    return best | sign


def fp8_decode(code: int) -> float:
    v = FP8_TABLE[code & 0x7F][1]
    return -v if code & 0x80 else v


def ceil_log2(x: float) -> int:
    m, k = math.frexp(x)
    return k - 1 if m == 0.5 else k


def scale_exponent(absmax: float, rule: str) -> int:
    """UE8M0 exponent of one 64-channel block: 'scale_floor' (CPU kernel, CuTeDSL) or 'absmax_floor' (Triton)."""
    if rule == "scale_floor":
        scale = max(fround(absmax / FP8_MAX), fround(QUANT_EPS))
    else:
        scale = fround(max(absmax, fround(QUANT_EPS)) * fround(1.0 / FP8_MAX))
    return ceil_log2(scale)


class Lcg:
    """Park-Miller minstd: s = s * 48271 mod 2^31-1, u = s / (2^31-1)."""

    def __init__(self, seed: int):
        self.s = seed

    def u(self) -> float:
        self.s = (self.s * 48271) % 2147483647
        return self.s / 2147483647.0


def make_inputs(spec: dict) -> dict:
    r, D = spec["ratio"], spec["head_dim"]
    W = coff_of(r) * D
    n = spec["tokens"]
    g = Lcg(spec["seed_kv"])
    kv = [[fround((g.u() - 0.5) * 4.0) for _ in range(W)] for _ in range(n)]
    g = Lcg(spec["seed_score"])
    score = [[fround((g.u() - 0.5) * 6.0) for _ in range(W)] for _ in range(n)]
    g = Lcg(spec["seed_ape"])
    ape = [[fround((g.u() - 0.5) * 2.0) for _ in range(W)] for _ in range(r)]
    g = Lcg(spec["seed_norm"])
    norm = [bf16(0.5 + g.u()) for _ in range(D)]
    return {"kv": kv, "score": score, "ape": ape, "norm": norm}


class ScalarCompressor:
    """Stage-rounded scalar statement of the compressor: state write, window gather with lanes, per-channel softmax
    pool, RMSNorm, nope bf16 -> FP8 with UE8M0 block scales, rope rotated at the group start and rounded to bf16."""

    def __init__(self, spec: dict, inputs: dict, rule="scale_floor", anchor="start", drop_state=False, prune_extra=0):
        self.r, self.D = spec["ratio"], spec["head_dim"]
        self.rope, self.qb = spec["rope_dim"], spec["quant_block"]
        self.nope = self.D - self.rope
        self.coff = coff_of(self.r)
        self.W = self.coff * self.r
        self.eps, self.theta = spec["eps"], spec["theta"]
        self.inp, self.rule, self.anchor, self.drop_state, self.prune_extra = inputs, rule, anchor, drop_state, prune_extra
        self.lenient = drop_state or prune_extra > 0
        self.state: dict[int, tuple[list, list]] = {}
        self.rows: list[dict] = []
        self.pos = 0

    def write(self, t: int) -> None:
        kv = list(self.inp["kv"][t])
        sc = [fround(s + a) for s, a in zip(self.inp["score"][t], self.inp["ape"][t % self.r])]
        self.state[t] = (kv, sc)

    def _cos_sin(self, pos: int):
        half = self.rope // 2
        ang = [pos * self.theta ** (-2.0 * i / self.rope) for i in range(half)]
        return [fround(math.cos(a)) for a in ang], [fround(math.sin(a)) for a in ang]

    def compress(self, p: int) -> dict:
        r, D, W = self.r, self.D, self.W
        pooled = []
        for c in range(D):
            es, ks = [], []
            for w in range(W):
                pos = p - W + 1 + w
                if pos < 0:
                    continue  # masked row: score -inf, kv 0
                off = D if (self.coff == 2 and w >= r) else 0
                if pos not in self.state:
                    if self.lenient:
                        continue  # contrast drivers only: a lost row silently drops out of the pool
                    raise KeyError(f"state row {pos} needed by the group completed at {p} is not in the state")
                kv, sc = self.state[pos]
                es.append(sc[off + c]), ks.append(kv[off + c])
            m = max(es)
            e = [math.exp(s - m) for s in es]
            pooled.append(fround(sum(a * k for a, k in zip(e, ks)) / sum(e)))
        ss = sum(x * x for x in pooled)
        mean = fround(ss / D)
        inv = fround(1.0 / math.sqrt(fround(mean + self.eps)))
        y = [fround(fround(x * inv) * w) for x, w in zip(pooled, self.inp["norm"])]
        nope_bf = [bf16(v) for v in y[:self.nope]]
        exps, codes = [], []
        for b in range(self.nope // self.qb):
            blk = nope_bf[b * self.qb:(b + 1) * self.qb]
            e = scale_exponent(max(abs(v) for v in blk), self.rule)
            exps.append(e)
            inv_s = 2.0 ** -e
            codes.extend(fp8_encode(fround(v * inv_s)) for v in blk)
        start = (p // r) * r
        pos = {"start": start, "end": p, "next_start": start + r}[self.anchor]
        cs, sn = self._cos_sin(pos)
        rope_out = []
        for i in range(self.rope // 2):
            ev, od = y[self.nope + 2 * i], y[self.nope + 2 * i + 1]
            rope_out += [bf16(fround(ev * cs[i] - od * sn[i])), bf16(fround(ev * sn[i] + od * cs[i]))]
        return {"group": p // r, "position": p, "anchor_position": pos, "pooled": pooled, "normed": y,
                "nope_scale_bytes": [e + 127 for e in exps], "nope_codes": codes, "rope_bf16": rope_out,
                "bf16_full": nope_bf + rope_out}

    def step(self, start: int, q: int) -> list[dict]:
        if self.drop_state:
            self.state = {}
        end = start + q
        for t in range(start, end):
            self.write(t)
        done = [self.compress(p) for p in range(start, end) if (p + 1) % self.r == 0]
        self.rows += done
        self.pos = end
        keep = end - self.W + 1 + self.prune_extra  # rows older than coff*ratio - 1 behind the next token are never read
        self.state = {t: v for t, v in self.state.items() if t >= keep}
        return done


def run_partition(spec, inputs, partition, **kw):
    c = ScalarCompressor(spec, inputs, **kw)
    progress, s = [], 0
    for q in partition:
        done = c.step(s, q)
        s += q
        progress.append({"start": s - q, "qlen": q, "visible_groups_after": len(c.rows), "completed_here": [d["group"] for d in done],
                         "state_rows_after": len(c.state)})
    return c.rows, progress


def _num(x):
    return float(f"{x:.9g}")


def _pack(row: dict) -> dict:
    out = dict(row)
    for k in ("pooled", "normed", "rope_bf16", "bf16_full"):
        out[k] = [_num(v) for v in row[k]]
    return out


FIXTURE_SPECS = {
    "period4": dict(ratio=4, head_dim=16, rope_dim=4, quant_block=4, eps=1e-6, theta=10000.0, tokens=23,
                    seed_kv=11, seed_score=12, seed_ape=13, seed_norm=14,
                    partitions={"one_shot": [23], "irregular": [1, 2, 5, 3, 7, 5], "decode": [1] * 23, "split_group": [2, 6, 15]}),
    "period128": dict(ratio=128, head_dim=16, rope_dim=4, quant_block=4, eps=1e-6, theta=10000.0, tokens=300,
                      seed_kv=21, seed_score=22, seed_ape=23, seed_norm=24,
                      partitions={"one_shot": [300], "irregular": [100, 27, 1, 128, 44], "split_at_boundary": [127, 1, 172],
                                  "lopsided": [255, 45]}),
}


def chunk_fixture(name: str, spec: dict) -> dict:
    inp = make_inputs(spec)
    base_rows, base_prog = run_partition(spec, inp, spec["partitions"]["one_shot"])
    n_groups = spec["tokens"] // spec["ratio"]
    parts, same = {}, True
    for pname, part in spec["partitions"].items():
        rows, prog = run_partition(spec, inp, part)
        eq = rows == base_rows
        same = same and eq
        parts[pname] = {"partition": part, "rows_identical_to_one_shot": eq, "progress": prog}
    tail_start = n_groups * spec["ratio"]
    pert = {"kv": [list(r) for r in inp["kv"]], "score": [list(r) for r in inp["score"]], "ape": inp["ape"], "norm": inp["norm"]}
    for t in range(tail_start, spec["tokens"]):
        pert["kv"][t] = [fround(v + 3.0) for v in pert["kv"][t]]
        pert["score"][t] = [fround(v - 2.0) for v in pert["score"][t]]
    pert_rows, _ = run_partition(spec, pert, spec["partitions"]["irregular"])
    irr = spec["partitions"]["irregular"]
    dropped_rows, _ = run_partition(spec, inp, irr, drop_state=True)
    short_rows, _ = run_partition(spec, inp, irr, prune_extra=1)
    if not same:
        raise RefError(f"fixture {name}: chunked rows differ from one-shot (reference bug)")
    if pert_rows != base_rows:
        raise RefError(f"fixture {name}: the incomplete tail changed a complete row (reference bug)")
    if short_rows == base_rows:
        raise RefError(f"fixture {name}: pruning one state row too many does not change the rows (not discriminating)")
    if dropped_rows == base_rows:
        raise RefError(f"fixture {name}: dropping the state between chunks does not change the rows (not discriminating)")
    return {"id": f"chunk_invariance_{name}", "topic": "group_completion",
            "spec": {k: v for k, v in spec.items() if k != "partitions"},
            "generator": "Park-Miller minstd (s = s*48271 mod 2147483647, u = s/2147483647); kv=(u-.5)*4, score=(u-.5)*6, ape=(u-.5)*2, "
                         "norm=bf16(.5+u), each rounded to fp32, order: kv rows, score rows, ape rows, norm; seeds in spec",
            "complete_groups": n_groups, "incomplete_tail_tokens": spec["tokens"] - tail_start,
            "partitions": parts, "all_partitions_identical": same,
            "tail_perturbation_leaves_rows_unchanged": pert_rows == base_rows,
            "dropped_state_changes_rows": dropped_rows != base_rows,
            "pruning_one_row_too_many_changes_rows": short_rows != base_rows,
            "rows": [_pack(r) for r in base_rows]}


def anchor_fixture() -> dict:
    spec = FIXTURE_SPECS["period4"]
    inp = make_inputs(spec)
    res = {}
    for a in ("start", "end", "next_start"):
        rows, _ = run_partition(spec, inp, [spec["tokens"]], anchor=a)
        res[a] = rows
    base = res["start"]
    contrast = {}
    for a in ("end", "next_start"):
        if all(x["rope_bf16"] == y["rope_bf16"] for x, y in zip(base[1:], res[a][1:])):
            raise RefError(f"anchor fixture does not discriminate {a}")
        contrast[a] = [x["rope_bf16"] != y["rope_bf16"] for x, y in zip(base, res[a])]
    return {"id": "anchor_contrast_period4", "topic": "anchoring",
            "statement": "the rope part of group g is rotated at position g*ratio (group start); anchoring at the completing token or at the next start gives "
                         "different rope values for every group except where the angle is zero",
            "anchors": [{"group": r["group"], "anchor_position": r["anchor_position"], "completing_position": r["position"]} for r in base],
            "differs_when_anchored_at_end": contrast["end"], "differs_when_anchored_at_next_start": contrast["next_start"],
            "rope_bf16_by_group": [[_num(v) for v in r["rope_bf16"]] for r in base]}


def scale_fixture() -> dict:
    samples = [0.001, 0.01, 0.02, 0.03, 0.0448, 0.05, 0.5, 1.0, 100.0, 447.0, 448.0, 449.0, 1000.0]
    rows = []
    for a in samples:
        a = bf16(a)
        e1, e2 = scale_exponent(a, "scale_floor"), scale_exponent(a, "absmax_floor")
        rows.append({"absmax_bf16": _num(a), "scale_floor_exponent": e1, "absmax_floor_exponent": e2, "rules_agree": e1 == e2,
                     "scale_byte": e1 + 127})
    if all(r["rules_agree"] for r in rows) or not any(r["rules_agree"] for r in rows):
        raise RefError("scale fixture does not discriminate the two floor rules")
    return {"id": "scale_rule", "topic": "nope_quant", "threshold_absmax": 0.0448, "exponent_differs_at_or_below_absmax": 0.02734375,
            "rule_a": "CPU kernel and CuTeDSL: scale = max(absmax/448, 1e-4); exponent = ceil(log2(scale)); byte = exponent + 127",
            "rule_b": "Triton: scale = max(absmax, 1e-4)/448; exponent = ceil(log2(scale))", "rows": rows}


def fp8_fixture() -> dict:
    vals = [0.0, 1.0, 1.0625, 1.1875, 0.001953125, 0.0009765625, 0.0029296875, 0.015625, 17.0, 447.0, 448.0, 460.0, 1000.0, -1.0625, -448.0, -0.3]
    return {"id": "fp8_e4m3_rounding", "topic": "nope_quant",
            "statement": "round to nearest, ties to the even code, subnormals kept, clamp to +-448; code 0x7F (NaN) never produced",
            "rows": [{"input": v, "code": fp8_encode(v), "decoded": fp8_decode(fp8_encode(v))} for v in vals]}


def layout_fixture() -> dict:
    """Synthetic rounding cost of the fp8_ds_mla layout against the fp32 normalised row (k20)."""
    D, rope, qb, rows = 512, 64, 64, 32
    nope = D - rope
    g = Lcg(31)
    w = [bf16(0.5 + g.u()) for _ in range(D)]
    num = {"fp8_nope": 0.0, "bf16_nope": 0.0, "bf16_rope": 0.0}
    den = {"nope": 0.0, "rope": 0.0}
    for _ in range(rows):
        x = [fround((g.u() + g.u() + g.u() + g.u() - 2.0) * 3.0) for _ in range(D)]
        ss = sum(v * v for v in x)
        inv = fround(1.0 / math.sqrt(fround(fround(ss / D) + 1e-6)))
        y = [fround(fround(a * inv) * b) for a, b in zip(x, w)]
        for b in range(nope // qb):
            blk = [bf16(v) for v in y[b * qb:(b + 1) * qb]]
            e = scale_exponent(max(abs(v) for v in blk), "scale_floor")
            for yy, bb in zip(y[b * qb:(b + 1) * qb], blk):
                dec = fp8_decode(fp8_encode(fround(bb * 2.0 ** -e))) * 2.0 ** e
                num["fp8_nope"] += (dec - yy) ** 2
                num["bf16_nope"] += (bb - yy) ** 2
                den["nope"] += yy * yy
        for yy in y[nope:]:
            num["bf16_rope"] += (bf16(yy) - yy) ** 2
            den["rope"] += yy * yy
    rel = {k: _num(math.sqrt(v / den["rope" if k == "bf16_rope" else "nope"])) for k, v in num.items()}
    return {"id": "layout_error", "topic": "cache_dtype", "rows": rows, "head_dim": D, "rope_dim": rope, "quant_block": qb,
            "statement": "relative L2 error against the fp32 normalised row on synthetic unit-RMS rows; the rope channels cost one bf16 rounding in every layout",
            "rel_l2": rel, "layouts": {"fp8_ds_mla_nope": rel["fp8_nope"], "bf16_full_nope": rel["bf16_nope"], "rope_bf16": rel["bf16_rope"]},
            "ratio_fp8_over_bf16_nope": _num(rel["fp8_nope"] / rel["bf16_nope"]), "unmeasured": "per-tensor fp8 full row (needs the fp8_scale of a running model)"}


def build_fixtures() -> list[dict]:
    return [chunk_fixture("period4", FIXTURE_SPECS["period4"]), chunk_fixture("period128", FIXTURE_SPECS["period128"]),
            anchor_fixture(), scale_fixture(), fp8_fixture(), layout_fixture()]


# ----------------------------------------------------------------------------- static sections
ROUNDING_STAGES = [
    {"stage": "projection", "dtype_in": "bf16", "dtype_out": "fp32", "operation": "hidden @ [Wkv; Wgate]^T, output fp32 (kv first, score second)", "facts": ["k02"]},
    {"stage": "state_write_kv", "dtype_in": "fp32", "dtype_out": "fp32", "operation": "store the kv row unchanged", "facts": ["k03"]},
    {"stage": "state_write_score", "dtype_in": "fp32", "dtype_out": "fp32", "operation": "score + APE[position % ratio], fp32 add, stored fp32", "facts": ["k03"]},
    {"stage": "window_gather", "dtype_in": "fp32", "dtype_out": "fp32", "operation": "rows position-coff*ratio+1 .. position; rows before 0 masked (score -inf, kv 0); lane offset D for rows w >= ratio", "facts": ["k06", "k07"]},
    {"stage": "channel_softmax_pool", "dtype_in": "fp32", "dtype_out": "fp32", "operation": "per channel softmax over the window, weighted sum of kv; CPU exp, CuTeDSL exp2 fastmath", "facts": ["k08", "k12"]},
    {"stage": "rmsnorm", "dtype_in": "fp32", "dtype_out": "fp32", "operation": "x * rsqrt(mean(x^2) + eps) * weight (bf16 weight widened); CPU 1/sqrt, CuTeDSL fastmath rsqrt", "facts": ["k08", "k12"]},
    {"stage": "nope_round", "dtype_in": "fp32", "dtype_out": "bf16", "operation": "channels [0, head_dim - rope_dim) rounded to bf16 (feeds only the quantiser in the fp8_ds_mla layout)", "facts": ["k09"]},
    {"stage": "nope_scale", "dtype_in": "bf16", "dtype_out": "ue8m0 byte", "operation": "per block of 64: scale = max(absmax/448, 1e-4), exponent = ceil(log2(scale)), byte = exponent + 127", "facts": ["k09", "k10"]},
    {"stage": "nope_fp8", "dtype_in": "bf16", "dtype_out": "fp8 e4m3fn", "operation": "value * 2^-exponent, clamp +-448, round to nearest even", "facts": ["k09"]},
    {"stage": "rope_rotate", "dtype_in": "fp32 (un-rounded)", "dtype_out": "fp32", "operation": "GPT-J interleaved rotation of the trailing rope channels at the group start, fp32 cos/sin", "facts": ["k11"]},
    {"stage": "rope_store", "dtype_in": "fp32", "dtype_out": "bf16", "operation": "one bf16 rounding of the rotated rope channels", "facts": ["k11"]},
    {"stage": "alt_layout_bf16_full", "dtype_in": "fp32", "dtype_out": "bf16", "operation": "nope channels rounded to bf16 and stored, rope as above; no FP8, no scale bytes", "facts": ["k13"]},
    {"stage": "alt_layout_fp8_full", "dtype_in": "fp32", "dtype_out": "fp8 e4m3fn", "operation": "whole row (rope rotated) to bf16, times 1/fp8_scale, clamp +-448, convert; one per-tensor scale", "facts": ["k13"]},
]

CACHE_LAYOUTS = [
    {"id": "fp8_ds_mla", "selected_when": "cache dtype uint8", "nope": "448 x fp8 e4m3fn, 7 ue8m0 scale bytes (block 64) + 1 pad byte", "rope": "64 x bf16",
     "bytes_at_head_512": 584, "facts": ["k13", "k09"]},
    {"id": "bf16_full", "selected_when": "cache dtype not uint8 and not fp8", "nope": "448 x bf16", "rope": "64 x bf16", "bytes_at_head_512": 1024, "facts": ["k13"]},
    {"id": "fp8_full_per_tensor", "selected_when": "cache dtype float8_e4m3fn", "nope": "448 x fp8", "rope": "64 x fp8 (one per-tensor scale for all 512)", "bytes_at_head_512": 512, "facts": ["k13"]},
]

AGREEMENT = [
    {"aspect": "boundary gate (position + 1) % ratio == 0", "cpu": "yes", "cutedsl": "yes", "triton": "yes", "verdict": "agree", "facts": ["k04"]},
    {"aspect": "window rows and masking of rows before position 0", "cpu": "slot -1 skipped", "cutedsl": "C4 pos >= 0 test, C128 none needed", "triton": "-inf score, 0 kv", "verdict": "agree", "facts": ["k06"]},
    {"aspect": "overlap lane channel offset (rows w >= ratio use channels [D, 2D))", "cpu": "head_offset", "cutedsl": "C4 head_offset = (row // ratio) * D", "triton": "head_offset", "verdict": "agree", "facts": ["k07"]},
    {"aspect": "per-channel softmax pool", "cpu": "max-subtract exp", "cutedsl": "online softmax, exp2 fastmath, warp combine", "triton": "tl.softmax", "verdict": "same algorithm, different fp32 summation order and exp", "facts": ["k08", "k12"]},
    {"aspect": "RMSNorm", "cpu": "1/sqrt", "cutedsl": "fastmath rsqrt, butterfly sums", "triton": "tl.rsqrt", "verdict": "same formula, different rounding", "facts": ["k08", "k12"]},
    {"aspect": "nope bf16 round before the quantiser", "cpu": "yes", "cutedsl": "yes", "triton": "yes", "verdict": "agree", "facts": ["k09"]},
    {"aspect": "FP8 scale floor", "cpu": "scale >= 1e-4", "cutedsl": "scale >= 1e-4", "triton": "absmax >= 1e-4", "verdict": "CPU = CuTeDSL; Triton differs: scale below absmax 0.0448, exponent byte at or below absmax 0.02734375", "facts": ["k09", "k10"]},
    {"aspect": "FP8 subnormals", "cpu": "exact scalar patch", "cutedsl": "hardware conversion", "triton": "hardware conversion", "verdict": "agree where representable", "facts": ["k10"]},
    {"aspect": "rope anchor", "cpu": "(p // ratio) * ratio", "cutedsl": "p - (ratio - 1) on fp8_ds_mla, (p // ratio) * ratio on full rows", "triton": "(p // ratio) * ratio", "verdict": "identical at boundary tokens", "facts": ["k11"]},
    {"aspect": "rope applied to the un-rounded row, one bf16 rounding", "cpu": "yes", "cutedsl": "yes", "triton": "yes", "verdict": "agree", "facts": ["k11"]},
]

IMPLEMENTATION_RULES = [
    "state rows: one fp32 row per absolute position, [kv | score + APE[pos % ratio]], only the newest coff*ratio positions are ever read",
    "a chunk writes all its rows first, then compresses every token with (pos + 1) % ratio == 0, reading earlier-chunk rows from the state",
    "the compressed row of the group completed at pos is row pos // ratio of the compressed cache; no other token writes a compressed row",
    "queries at position pos read compressed rows 0 .. (pos + 1)//ratio - 1; the group completed by pos itself is included",
    "replayed tokens (PAD slot) neither rewrite state nor compressed rows",
]


def state_api(per: list[dict]) -> dict:
    return {
        "state_cache": {"rows": "one per absolute token position", "row_layout": "[kv: coff*D fp32 | score+APE: coff*D fp32]", "dtype": "fp32",
                        "retention": "coff*ratio positions", "facts": ["k02", "k14"]},
        "operations": [
            {"name": "state_write", "per": "token", "effect": "state[slot] = [kv, score + APE[pos % ratio]]", "skip": "slot < 0", "facts": ["k03"]},
            {"name": "compress", "per": "token with (pos + 1) % ratio == 0", "effect": "gather window rows, per-channel softmax pool, RMSNorm, nope fp8/bf16, rope at group start, store at the compressed slot", "skip": "kv slot < 0 (PAD)", "facts": ["k04", "k06", "k08", "k09", "k11"]},
            {"name": "compressed_slot", "per": "token", "effect": "block_table[(pos//ratio)//block_size]*block_size + (pos//ratio)%block_size at boundary tokens, PAD elsewhere", "facts": ["k04", "k18"]},
            {"name": "visibility", "per": "query", "effect": "compressed rows 0..(pos+1)//ratio - 1", "facts": ["k05"]},
        ],
        "rules": IMPLEMENTATION_RULES,
        "periods": [{"ratio": p["ratio"], "window_rows": p["window_rows"], "state_row_floats": p["state_row_floats"], "state_block_rows": p["state_block_rows"]} for p in per],
    }


# ----------------------------------------------------------------------------- facts, report
def load_facts() -> dict:
    path = pathlib.Path(__file__).resolve().parent / FACTS_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RefError(f"{FACTS_NAME} cannot be read: {exc}")


def build_report(cfg: dict, positions: list[int] | None, chunks: list[tuple[int, int]] | None) -> dict:
    facts = load_facts()
    per = periods(cfg)
    D, maxpos = cfg["head_dim"], cfg["max_position_embeddings"]
    n = cfg["num_hidden_layers"]
    hca_layers = [i for i in range(n) if cfg["compress_ratios"][i] == 128]
    vis = []
    for p in per:
        r = p["ratio"]
        ps = positions if positions is not None else default_positions(r, maxpos)
        if any(x >= maxpos for x in ps):
            raise RefError(f"--positions must be below max_position_embeddings ({maxpos}), got {max(ps)}")
        cs = chunks if chunks is not None else default_chunks(r)
        vis.append({"ratio": r, "rule": "visible_groups = (position + 1) // ratio; group j completes at position j*ratio + ratio - 1 and is visible from sequence length j*ratio + ratio; the incomplete tail is invisible",
                    "position_table": [vis_row(x, r) for x in ps], "group_table": group_rows(r, maxpos),
                    "chunk_table": [chunk_row(s, q, r) for s, q in cs], "overlap_rows": overlap_rows(r, D)})
    unresolved = [{"id": f["id"], "statement": f["statement"], "why": f["why"], "impact": f["impact"], "resolver": f["resolver"]}
                  for f in facts["facts"] if f["status"] == "unresolved"]
    rep = {
        "report": "ernie55-hca-compression", "schema_version": 1, "pins": facts["pins"], "sources": facts["sources"],
        "config": {"hidden_size": cfg["hidden_size"], "head_dim": D, "rope_dim": cfg["qk_rope_head_dim"], "nope_dim": D - cfg["qk_rope_head_dim"],
                   "rms_norm_eps": cfg["rms_norm_eps"], "compress_rope_theta": cfg["compress_rope_theta"],
                   "max_position_embeddings": maxpos, "hca_layers": hca_layers, "hca_layer_count": len(hca_layers),
                   "max_compressed_rows_at_128": cdiv(maxpos, 128), "compressor_tensors_expected": 4 * len(hca_layers)},
        "periods": per, "visibility": vis, "rounding_stages": ROUNDING_STAGES, "cache_layouts": CACHE_LAYOUTS,
        "agreement": AGREEMENT, "state_api": state_api(per), "code_map": facts["code_map"], "facts": facts["facts"],
        "resolves_3_7": facts["resolves"], "fixtures": build_fixtures(), "unresolved": unresolved,
    }
    return rep


# ----------------------------------------------------------------------------- verification
def _all_provenance(report: dict):
    for f in report["facts"]:
        for p in f.get("provenance", []):
            yield f["id"], p
    for m in report["code_map"]:
        for p in m.get("citations", []):
            yield m["id"], p


def verify_sources(report: dict, tree: pathlib.Path, repo_root: pathlib.Path, config_path: pathlib.Path | None) -> dict:
    shas = {(s["pin"], s["file"]): s.get("sha256") for s in report["sources"]}
    cache: dict = {}
    failures, checked = [], 0

    def text_of(pin, file):
        key = (pin, file)
        if key not in cache:
            if pin == "ernie_repo":
                cache[key] = subprocess.run(["git", "-C", str(repo_root), "show", f"{report['pins']['ernie_repo']['commit']}:{file}"],
                                            capture_output=True, check=True).stdout
            elif pin == "hf_config":
                cache[key] = (config_path or tree / "hf_config" / file).read_bytes()
            else:
                cache[key] = (tree / pin / file).read_bytes()
        return cache[key]

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
        if p["file"] not in cache:
            cache[p["file"]] = _header(model_dir, p["file"])
        t = cache[p["file"]].get(p["tensor"])
        if t is None or t["dtype"] != p["dtype"] or t["shape"] != p["shape"]:
            failures.append({"fact": owner, "tensor": p["tensor"], "problem": f"header says {t and (t['dtype'], t['shape'])}"})
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    cfgr = report["config"]
    p128 = next(p for p in report["periods"] if p["ratio"] == 128)
    want = {"ape": ("F32", p128["ape_shape"]), "linear_wkv.weight": ("BF16", [cfgr["head_dim"], cfgr["hidden_size"]]),
            "linear_wgate.weight": ("BF16", [cfgr["head_dim"], cfgr["hidden_size"]]), "norm.weight": ("BF16", [cfgr["head_dim"]])}
    seen: dict[int, set] = {}
    total = 0
    for name, shard in index.items():
        m = re.match(r"model\.layers\.(\d+)\.self_attn\.core_attention\.compressor\.(.+)$", name)
        if not m:
            continue
        total += 1
        layer, kind = int(m.group(1)), m.group(2)
        seen.setdefault(layer, set()).add(kind)
        if shard not in cache:
            cache[shard] = _header(model_dir, shard)
        t = cache[shard][name]
        if kind not in want:
            failures.append({"tensor": name, "problem": "unexpected compressor tensor"})
        elif (t["dtype"], t["shape"]) != want[kind]:
            failures.append({"tensor": name, "problem": f"header says {(t['dtype'], t['shape'])}, expected {want[kind]}"})
    if sorted(seen) != cfgr["hca_layers"]:
        failures.append({"problem": "layers with compressor tensors differ from the HCA layers", "found": sorted(seen)})
    for layer, kinds in seen.items():
        if kinds != set(want):
            failures.append({"layer": layer, "problem": f"tensor kinds {sorted(kinds)}"})
    if total != cfgr["compressor_tensors_expected"]:
        failures.append({"problem": f"{total} compressor tensors, expected {cfgr['compressor_tensors_expected']}"})
    return {"verified": "headers", "citations_checked": checked, "shards_read": len(cache), "compressor_tensors": total,
            "layers": len(seen), "failures": failures}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ERNIE 5.5 HCA compressor state and visibility report (offline).")
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--positions")
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
        positions = parse_positions(args.positions) if args.positions is not None else None
        chunks = parse_chunks(args.chunks) if args.chunks is not None else None
        report = build_report(cfg, positions, chunks)
    except RefError as exc:
        print(f"hca_compression: {exc}", file=sys.stderr)
        return 2
    if args.verify_sources or args.verify_headers:
        receipts = []
        if args.verify_sources:
            receipts.append(verify_sources(report, pathlib.Path(args.verify_sources), pathlib.Path(args.repo_root),
                                           pathlib.Path(args.metadata_dir) / ar.CONFIG_NAME))
        if args.verify_headers:
            receipts.append(verify_headers(report, pathlib.Path(args.verify_headers)))
        sys.stdout.write(ar.dumps(receipts) + "\n")
        return 1 if any(r["failures"] for r in receipts) else 0
    sys.stdout.write(ar.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
