#!/usr/bin/env python3
"""ERNIE 5.5 Flash memory plan at the IQR-V4 target (story 5-0-iqr-v4-target): arithmetic over the MEASURED converted size, never a GPU run.

  iqr_memplan.py --layers docs/ernie55/iqr-v4-layers.json --memguard /var/lib/memguard/host.json --out-json docs/ernie55/iqr-v4-memory-plan.json [--out-md ...]

Scenarios: (1) TP8 over the eight 32 GiB cards (weights sharded evenly: experts by width, 2048 = 16 blocks of 128, 2 a rank; attention by head); (2) the four-card variant: one socket's four cards,
the experts that do not fit streamed from the host pool; (3) host RAM against memguard. The FP8 baseline (268.6 GiB, the figure the earlier plan used) is carried for comparison.
Every assumption is a named field (reserve, KV bytes a token from the config, link bandwidth, host use); none of them is a measurement.
"""
import argparse, json, re, sys
from pathlib import Path

GIB = 2 ** 30
CARD_GIB = 32.0
CFG = Path.home() / "models/ernie55-flash-fp8/config.json"
RUNTIME_RESERVE_GIB = 3.0       # engine arena, graph capture, collectives scratch: an assumption (a served model budgets ~2-4 GiB a card)
HOST_OTHER_USE_GIB = 16.0       # pinned staging, the file-tier cache and the engine's own host memory: an assumption
NODE2_CARDS = 4                 # four cards of one socket: the half-node variant
ALL_CARDS = 8


def kv_bytes_per_token(cfg):
    """From the config: the 7 MLA-DSA layers (6 main + the ECHO layer) cache a bf16 latent (512 + 64 rope) and an FP8 index key (128 codes + an f32 scale) a token; the 37 HCA layers cache one
    512-wide bf16 entry per 128 tokens (plus a fixed 128-token window and the compressor state, not per token)."""
    ratios = cfg["compress_ratios"]
    mla = sum(1 for r in ratios if r == -2)
    hca = sum(1 for r in ratios if r == 128)
    per_mla = (cfg["kv_lora_rank"] + cfg["qk_pos_emb_head_dim"]) * 2 + (cfg["index_head_dim"] + 4)
    per_hca = cfg["head_dim"] * 2 / 128
    window = hca * cfg["sliding_window"] * cfg["head_dim"] * 2
    return dict(mla_layers=mla, hca_layers=hca, bytes_per_token=mla * per_mla + hca * per_hca, fixed_window_bytes_per_sequence=window)


def build(layers, memguard, cfg):
    total, experts = layers["total_bytes"], layers["experts_bytes"]
    nonexp = total - experts
    kv = kv_bytes_per_token(cfg)
    ctx = {str(c): round(kv["bytes_per_token"] * c / GIB + kv["fixed_window_bytes_per_sequence"] / GIB, 3) for c in (131072, 262144, 1048576)}
    # scenario 1: TP8 over all eight cards
    w_card = total / ALL_CARDS / GIB
    replicated = 0.0
    for s in layers["size_by_class"]:
        if any(k in s["class_name"] for k in ("block_sparse_moe.gate.weight", "mhc", "hyper_connection", "hc_head", "indexer.wk", "indexer.weights_proj", "latent_norm", "compressor.ape")):
            replicated += s["total_bytes"]
    rep_gib = replicated / GIB * (1 - 1 / ALL_CARDS)      # a replicated tensor costs its full size on every rank, not 1/8
    reserve = RUNTIME_RESERVE_GIB + rep_gib
    free = CARD_GIB - w_card - reserve
    kv1m = ctx["1048576"]
    tp8 = dict(id="tp8_all_cards", cards=ALL_CARDS, vram_per_card_gib=CARD_GIB, vram_total_gib=CARD_GIB * ALL_CARDS, weights_per_card_gib=w_card, reserve_per_card_gib=reserve,
               runtime_reserve_gib=RUNTIME_RESERVE_GIB, replicated_overhead_per_card_gib=rep_gib, free_per_card_gib=free,
               fits_weights_resident=total / GIB <= ALL_CARDS * (CARD_GIB - reserve), host_ram_for_experts_gib=0.0 if total / GIB <= ALL_CARDS * (CARD_GIB - reserve) else max(0.0, total / GIB - ALL_CARDS * (CARD_GIB - reserve)),
               kv_gib_per_sequence=ctx, kv_replicated_per_card_gib_at_1m=kv1m,
               note="TP8, the latent and index caches replicated on every rank (as glm5next does: one latent head); with DCP (story 6) they shard 1/8. free_per_card is what is left for KV, speculative buffers and prefill arena.",
               kv_fits_1m_one_sequence=kv1m <= free,
               socket_note="all eight cards means a production model already on the other socket's cards must be unloaded first: a planned maintenance window")
    # scenario 2: four cards of one socket, experts streamed from host
    vram = NODE2_CARDS * CARD_GIB
    rep_total_gib = replicated / GIB
    n_reserve = NODE2_CARDS * (RUNTIME_RESERVE_GIB + rep_total_gib * (1 - 1 / NODE2_CARDS))
    kv_ctx = ctx["131072"]
    kv_all = kv_ctx * NODE2_CARDS
    avail_for_weights = vram - n_reserve - kv_all
    nonexp_gib = nonexp / GIB
    exp_gib = experts / GIB
    exp_resident = max(0.0, min(exp_gib, avail_for_weights - nonexp_gib))
    streamed = exp_gib - exp_resident
    ex_tensors = [t for t in layers["tensors"] if t["precision"] == "iqr_v4"]
    by_w = {}
    for t in ex_tensors:
        by_w.setdefault(re.search(r"\.(w[123])\.weight$", t["name"]).group(1), []).append(t["bytes_dst"])
    per_expert_mib = sum(sum(v) / len(v) for v in by_w.values()) / 2 ** 20 if by_w else 0.0     # one expert = the mean w1 + the mean w3 + the mean w2
    moe_layers = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
    active_gib_per_token = 10 * moe_layers * per_expert_mib / 1024
    frac = streamed / exp_gib if exp_gib else 0.0
    n2 = dict(id="four_card_expert_streaming", cards=NODE2_CARDS, max_cards_busy=NODE2_CARDS, vram_per_card_gib=CARD_GIB, vram_total_gib=vram, reserve_gib=n_reserve,
              kv_context_tokens=131072, kv_gib_all_ranks=kv_all, nonexpert_resident_gib=nonexp_gib, expert_resident_gib=exp_resident, expert_streamed_gib=streamed,
              experts_total_gib=exp_gib, streamed_fraction=frac,
              active_expert_gib_per_decode_token_if_all_streamed=active_gib_per_token, expected_streamed_gib_per_decode_token_uniform_routing=active_gib_per_token * frac,
              traffic_bound_gib_per_decode_token_all_streamed=active_gib_per_token,
              link_note="CONDITIONAL ESTIMATE, not a bound: it assumes routing spreads uniformly over the experts and a link throughput that is NOT measured here (25 and 50 GB/s are stated assumptions); a token can pick more streamed experts than the expectation. The only hard bound is the all-streamed traffic figure (10 experts a layer, every one from host); the hot-expert profile (expert-tiered placement) is what would tighten it, and no routing distribution of the real model has been measured",
              ms_per_token_at_25_gbs=active_gib_per_token * frac * 1.0737 / 25 * 1000, ms_per_token_at_50_gbs=active_gib_per_token * frac * 1.0737 / 50 * 1000,
              note="the four cards of one socket are free and at most 4 cards may be busy: this variant fits such a window and is the CHEAPEST way to run ERNIE for a first real-weight measurement")
    # scenario 3: host RAM against memguard
    avail_gib = memguard["host_available_gib"]
    window = memguard["gpu_window_ok_gib"]
    h = dict(id="host_ram_budget", memguard=dict(host_total_gib=memguard["host_total_gib"], host_available_gib=avail_gib, gpu_window_ok_gib=window, kill_below_gib=memguard.get("kill_below_gib"), read_at=memguard.get("time")),
             other_host_use_gib=HOST_OTHER_USE_GIB, streamed_gib=streamed, fits_gpu_window=streamed <= window - HOST_OTHER_USE_GIB,
             host_cap_gib=96, host_pool_rule_gib=32,
             exceeds_host_pool_rule=streamed > 32,
             tp8_all_cards_host_gib=tp8["host_ram_for_experts_gib"], note="the host pool is capped at 32 GiB unless the machine's free memory says otherwise; the GPU window allowance is memguard's gpu_window_ok_gib. At TP8 on all cards no expert is streamed and host RAM is not on the path.")
    fp8 = dict(total_gib=layers["source_fp8_gib"], vram_total_gib=CARD_GIB * ALL_CARDS, vram_overflow_gib=layers["source_fp8_gib"] - CARD_GIB * ALL_CARDS,
               note="the figure of the earlier plan: block FP8 is not a Radiance 2.x serving format, so this configuration cannot exist; it is kept to show what the target changes")
    return dict(schema="ernie55-iqr-v4-memory-plan/1", model=dict(total_bytes=total, total_gib=total / GIB, experts_bytes=experts, experts_gib=exp_gib, nonexpert_bytes=nonexp, nonexpert_gib=nonexp_gib,
                                                                 per_expert_mib=per_expert_mib, source="docs/ernie55/iqr-v4-layers.json (measured per-class bytes x checkpoint counts)"),
                kv=dict(per_token_bytes=kv["bytes_per_token"], mla_layers=kv["mla_layers"], hca_layers=kv["hca_layers"], gib_one_sequence_by_context=ctx), scenarios=[tp8, n2, h],
                fp8_baseline=fp8, assumptions=dict(card_gib=CARD_GIB, runtime_reserve_per_card_gib=RUNTIME_RESERVE_GIB, host_other_use_gib=HOST_OTHER_USE_GIB, kv_formula="7 MLA layers x ((512+64) x 2 + 132) bytes + 37 HCA layers x 1024/128 bytes, per token, plus a fixed 128-token window a sequence in each HCA layer; bf16 cache; replicated tensors counted: routers, mHC, indexer wk and weights_proj, latent norm, compressor APE (the compressor kv/gate projections are treated as sharded)"))


def render_md(m):
    mo = m["model"]
    t, n, h = m["scenarios"]
    f = m["fp8_baseline"]
    L = ["# ERNIE 5.5 Flash memory plan at the IQR-V4 target (story 5-0-iqr-v4-target)", "",
         "Generated by `tools/ernie55/iqr_memplan.py` from the measured converted size (`iqr-v4-layers.json`). Arithmetic only: nothing here ran on a GPU; every assumption is a field of the JSON.", "",
         f"Model at IQR-V4 + IQ8R: **{mo['total_gib']:.1f} GiB** ({mo['experts_gib']:.1f} routed experts, {mo['nonexpert_gib']:.1f} everything else). The FP8 plan was {f['total_gib']:.1f} GiB against {f['vram_total_gib']:.0f} GiB of VRAM ({f['vram_overflow_gib']:.1f} GiB overflow); it cannot exist (FP8 is not a serving format).", "",
         "## TP8 over all eight cards", "",
         f"- weights {t['weights_per_card_gib']:.2f} GiB a card, reserve {t['reserve_per_card_gib']:.2f} (runtime {t['runtime_reserve_gib']:.1f} + replicated tensors {t['replicated_overhead_per_card_gib']:.2f}), free {t['free_per_card_gib']:.2f} GiB of {t['vram_per_card_gib']:.0f}. Everything resident, no host spill.",
         f"- KV for one sequence, replicated per rank: {', '.join(f'{k}: {v} GiB' for k, v in t['kv_gib_per_sequence'].items())}. One 1M-token sequence {'fits' if t['kv_fits_1m_one_sequence'] else 'does not fit'} in the free space.", "",
         "## four cards of one socket, experts streamed from host", "",
         f"- {n['vram_total_gib']:.0f} GiB VRAM; non-expert weights {n['nonexpert_resident_gib']:.1f} GiB resident; KV at 131072 tokens {n['kv_gib_all_ranks']:.1f} GiB over four ranks; experts resident {n['expert_resident_gib']:.1f} of {n['experts_total_gib']:.1f} GiB, **{n['expert_streamed_gib']:.1f} GiB streamed** ({n['streamed_fraction']*100:.0f}%).",
         f"- conditional estimate under uniform routing: {n['expected_streamed_gib_per_decode_token_uniform_routing']:.2f} GiB of streamed experts a decode token, {n['ms_per_token_at_25_gbs']:.0f} ms at an assumed 25 GB/s, {n['ms_per_token_at_50_gbs']:.0f} ms at an assumed 50 GB/s (link and routing distribution not measured; the only hard traffic bound is all-streamed: {n['traffic_bound_gib_per_decode_token_all_streamed']:.2f} GiB a token).", "",
         "## Host RAM against memguard", "",
         f"- memguard ({h['memguard']['read_at']}): {h['memguard']['host_total_gib']} GiB total, {h['memguard']['host_available_gib']} available, GPU-window allowance {h['memguard']['gpu_window_ok_gib']} GiB. Streamed experts {h['streamed_gib']:.1f} GiB + other host use {h['other_host_use_gib']:.0f} GiB: {'fits' if h['fits_gpu_window'] else 'does NOT fit'} the window; {'exceeds' if h['exceeds_host_pool_rule'] else 'within'} the 32 GiB host-pool rule (96 GiB host cap).", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="docs/ernie55/iqr-v4-layers.json"); ap.add_argument("--memguard", default="/var/lib/memguard/host.json")
    ap.add_argument("--config", default=str(CFG)); ap.add_argument("--out-json", required=True); ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    try:
        m = build(json.loads(Path(a.layers).read_text()), json.loads(Path(a.memguard).read_text()), json.loads(Path(a.config).read_text()))
    except (OSError, ValueError, KeyError) as e:
        print(f"iqr_memplan: {e}", file=sys.stderr)
        return 2
    Path(a.out_json).write_text(json.dumps(m, indent=1) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(render_md(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
