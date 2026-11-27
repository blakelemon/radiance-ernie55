# ERNIE 5.5 attention reference (`tools/ernie55/attention_reference.py`)

A source-backed account of the two attention families of ERNIE 5.5 Flash, written before any Radiance
attention op is specified: **MLA-DSA** (`compress_ratios` entry `-2`) and **HCA** (entry `128`). It answers,
for each family, the questions stories 3-1 to 3-4 and 6-1 must not guess: where the trailing window starts
and ends, which tokens are candidates, when a compressed group becomes visible, how positions are
transformed, what the KV rows look like, what the scale, sink, VHA and gate are and in which order they
act, where precision changes, and which existing Radiance op can carry each piece.

```
python3 tools/ernie55/attention_reference.py --metadata-dir /models/ERNIE-5.5-Flash-Base
python3 tools/ernie55/attention_reference.py --metadata-dir DIR --sample-lengths 127,128,129 --chunks 0:128,127:1
python3 tools/ernie55/attention_reference.py --metadata-dir DIR \
    --verify-sources SRC_TREE --repo-root . --verify-headers /models/ernie55-flash-fp8
```

`--metadata-dir` needs only `config.json` (the pinned Base config). The committed report is
`docs/ernie55/attention-reference.json` (the tool's output for the pinned config, byte for byte). Valid input
exits `0` with the JSON on stdout; invalid input (wrong schedule length, a ratio other than 128 or -2, a
non-positive window, odd rope width, ...) exits `2` naming the key and prints nothing on stdout.

## What the report contains

| Section | Content |
| --- | --- |
| `pins`, `sources` | the exact upstream pins and, for every cited upstream file, its raw URL and SHA-256 |
| `schedule` | all 44 entries: kind (main or draft), ratio, family |
| `mla`, `hca` | `facts` (id, topic, status, statement, provenance), `derived` constants, `window_table`, and for HCA `group_table`, `chunk_table`, `production` |
| `reuse` | one source-backed comparison per Radiance candidate op, with mismatch records |
| `fixtures` | discriminating fixtures, each with the correct result and tempting wrong variants |
| `unresolved` | every fact the pinned sources do not settle, with its resolver |

Window, group, chunk and production tables are computed from `config.json`, never copied: change
`sliding_window`, `compress_ratios`, `index_topk` or `max_position_embeddings` and they change.

## Pins

| Pin | Value |
| --- | --- |
| Baidu | `baidu/vLLM-ERNIE` commit `b8414da6b703579814c2009c5a63bac101916bbe`, `src/vllm_ernie/models/ernie5_5/` (`nvidia/attention.py`, `config.py`, `nvidia/model.py`, `nvidia/mtp.py`) |
| vLLM | tag `v0.32.0` (`5df63a7203cdb9328d824a27b682e77c7b9928be`): `models/deepseek_v4`, `v1/attention/backends/mla`, `model_executor/layers/{sparse_attn_indexer,rotary_embedding,attention}`, `csrc/cpu/sgl-kernels/compressor.cpp` |
| FlashMLA | `vllm-project/FlashMLA` `0eee43b12f034b657133cf2afca6a72ebb6efccf` (named by vLLM's cmake; not read) |
| Config | `baidu/ERNIE-5.5-Flash-Base` revision `028544b2ffceea189b2d8768ed2b9d09b06f2343`, `config.json` |
| Checkpoint | safetensors **headers** (names, dtypes, shapes) of the same revision; no tensor data |
| Radiance | this repository at `f900e7d21f0edac9029529d536a4bd146e10ba23` |

Every provenance entry is `{pin, file, line, line_end, quote}` (checkpoint facts: `{pin: hf_headers, file,
tensor, dtype, shape}`). The quote is an exact substring of the cited lines. `--verify-sources` re-reads
every citation from fetched copies of the pinned files (and `git show` for the Radiance pin), checks the file
hashes and quotes, and `--verify-headers` re-reads the cited headers and checks that all 37 HCA entries and all
7 MLA-DSA entries (6 main + draft 43) have identical attention tensor signatures. The receipt is in the story
evidence.

## Schedule

`compress_ratios` has 44 entries: 43 main layers plus one draft entry. An entry is DSA when its ratio is
`<= 0` and its index is below the list length (`Ernie5_5Config.is_dsa_layer`); every other entry is HCA.

| Family | Entries |
| --- | --- |
| MLA-DSA, main | 7, 14, 21, 28, 35, 42 (zero based) |
| HCA, main | the other 37 main layers, ratio 128 |
| Draft (MTP) | entry 43, ratio -2, so a DSA layer with its own buffers; the draft is a separate artifact stage |

A draft entry whose ratio were 128 would be SWA-only in the pinned vLLM code (its index is not below
`num_hidden_layers`); this checkpoint has none (fact `h01`). The ratio-4 "C4A" variant with overlap lanes and a
lightning indexer exists in the vLLM code but is not selected by this model: no overlap lane applies at
ratio 128 (`h05`).

## The trailing window (both families)

With `seq = position + 1` and `window = sliding_window = 128`, **the window is token indices
`seq - w .. seq - 1` inclusive, `w = min(seq, window)`**: positions `p-127 .. p`, 128 tokens, the query's own
token last. Proven in `merge_swa_topk` (`m03`, DSA) and the SWA index kernel plus the prefill combine kernel
(`h02`, HCA); the Radiance `ref_attn_paged_sink` window test (`pos - j >= window` hides a key) is the same
rule. The first full window is at `seq = 128` (`[0, 127]`), at `seq = 129` it is `[1, 128]`.

| seq | window (inclusive) | DSA indexer range (half open) | DSA selected | DSA row length | HCA complete groups | HCA incomplete tokens | HCA candidates | window/group overlap |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0..0 | [0, 0) | 0 | 1 | 0 | 1 | 1 | 0 |
| 127 | 0..126 | [0, 0) | 0 | 127 | 0 | 127 | 127 | 0 |
| 128 | 0..127 | [0, 0) | 0 | 128 | 1 | 0 | 129 | 128 |
| 129 | 1..128 | [0, 1) | 1 | 129 | 1 | 1 | 129 | 127 |
| 256 | 128..255 | [0, 128) | 128 | 256 | 2 | 0 | 130 | 128 |
| 2176 | 2048..2175 | [0, 2048) | 2048 | 2176 | 17 | 0 | 145 | 128 |
| 2177 | 2049..2176 | [0, 2049) | 2048 | 2176 | 17 | 1 | 145 | 127 |
| 1048576 | 1048448..1048575 | [0, 1048448) | 2048 | 2176 | 8192 | 0 | 8320 | 128 |

(The report holds the full table for 15 default lengths and for any `--sample-lengths`.)

## MLA-DSA

* **Candidates.** The indexer scores only tokens older than the window, `[0, max(seq - window, 0))`
  (`m04`): prefill clamps its exclusive end to `max(ke - window, ks)`, decode subtracts the window from
  `seq_lens`. It returns `n = min(max(seq - window, 0), index_topk)` ids; the merge writes
  `[top-k ids | window ids ascending | -1]`, width `index_topk + sliding_window = 2176` (`m05`). Candidates and
  window never overlap in a DSA layer, so nothing is attended twice; truncation by top-k begins at
  `seq = 2177`. The order inside the selected ids and tie handling belong to the compiled top-k kernels
  (`m06`, unresolved).
* **Positions.** q is `[nope 192 | rope 64]` and k_pe a single shared 64-wide head; RoPE is GPT-J
  interleaved, on the trailing slice, plain at `rope_theta` 8,000,000, **no YaRN** (`m08`). The cos/sin table is built from f32 angles and cast to the model dtype, so **bf16 cos and sin** (`get_rope` default dtype, `base.py` `cache.to(dtype)`; corrected in story 3-2b); only the HCA YaRN class keeps an f32 cache. The **indexer** is
  different: q and k are split **rope first** `[rope 64 | nope 64]`, rotated NeoX split-half, then a Hadamard
  over all 128; k passes `LayerNorm(128, eps 1e-6)` with bias first (`m09`). Positions are absolute
  request-local, so chunked prefill sees what a one-shot prefill sees (`m24`).
* **Scale** `(192 + 64)^-0.5 = 0.0625`, no YaRN magnitude (`m11`). The indexer logit is
  `sum_h relu(q_h . k) * weight_h`, weights scaled by `head_dim^-0.5 * n_head^-0.5` (`m10`).
* **Latent MQA.** Decode and prefill both run absorbed MQA over the 576-wide latent row; `kv_b_proj` is
  `[W_UK 192 ; W_UV 256]` per head, so key and value widths differ (`m12`).
* **KV layout.** One 576-element row per token (normed latent, then rotated k_pe) in 64-token pages: 1152
  bytes bf16, or 656 bytes `fp8_ds_mla` (512 FP8, 4 fp32 scales, 64 bf16 rope). The indexer keeps a per-token
  key row of 132 bytes (128 FP8 + fp32 scale; MXFP4: 68) (`m13`, `m14`, `m26`). The cache dtype is a serving
  choice (`m25`).
* **Sink, VHA, gate.** Sink: one fp32 logit per head, in the max and the denominator, no value (`m15`).
  VHA: `o[h] <- sum_g m[h,g] o[g]`, `m = I + V U^T`, rank 16, on the head axis (`m17`). Order: attention,
  then VHA, then element `sigmoid(gate)` (gate from the normed q latent, width `64 x 256`), then `wo_b`
  (`m16`).
* **Draft entry 43** is the same layer with its own buffers (`m23`).

## HCA

* **Compressed candidates.** A query at position `p` sees compressed entries `0 .. (p+1)//128 - 1`: **only
  complete groups**; group `j` covers tokens `128j .. 128j+127` and becomes visible at `seq = 128(j+1)`
  (`h03`, `h05`). There is no indexer at ratio 128, so every complete entry is attended; the list width is
  not capped by `index_topk` (`h03`).
* **Group completion.** An entry is produced at the **last** token of its group: the compress step is skipped
  unless `(position + 1) % 128 == 0`; every token first stores its raw kv and `score + APE[position % 128]` in
  an fp32 state cache (`h05`, `h15`). Chunked prefill keeps that state across steps; a chunk completes at
  least one group iff `start % 128 + chunk_len >= 128` (`h07`), which the `chunk_table` reproduces.
* **Compression arithmetic.** Per **channel** softmax over the 128 rows of `score + APE`, weighted sum of the
  128 fp32 kv rows, fp32 RMSNorm over 512 with the compressor weight, then nope (448) to bf16 and FP8 in
  blocks of 64 (UE8M0), rope (64) rotated GPT-J from the fp32 value and stored bf16 (`h06`). The rope position
  is the group **start** `(position // 128) * 128` (`h22`).
* **Incomplete group.** At most 127 trailing tokens exist only in the state cache and are invisible as
  entries; the 128-token window always covers them, so every token is attended raw or through a complete
  entry (`h07`).
* **Overlap.** The window and the newest complete group overlap (128 tokens at the completing token, one
  fewer per further token); the pinned source does **not** de-duplicate (`combined_len = topk_len + swa_len`),
  so the duplicate is part of the reference (`h08`). Story 3-4's "duplicate exclusion" wording must be read as
  "the window and compressed sets are kept distinct", not "tokens are removed" (concern recorded in the
  evidence).
* **Union softmax.** One softmax over window and compressed keys together, with the sink (`h04`); the bf16 FlashInfer path builds the same union as one index matrix (its kernel was not read).
* **Positions and scaling.** q: weightless per-head RMSNorm over 512, then GPT-J RoPE on the **trailing** 64;
  kv: `kv_norm` then the same rotation; V is the rotated row, so the output projection applies the **inverse**
  rotation at the query position before `wo_a` (`h10`). At ratio 128 RoPE uses `compress_rope_theta` 160,000
  and YaRN (factor 16, beta 32/1, original 65,536 -> correction range 15..25 for 32 pairs, magnitude exactly
  1, fp32 cos/sin cache of 1,048,576 rows) (`h11`); whether the YaRN branch is reached for this config's
  `type` key is `h12` (unresolved). Scale `512^-0.5` (`h09`).
* **Output.** Attention (512-wide heads) -> VHA -> inverse RoPE + grouped `wo_a` (8 groups of 8 heads,
  4096 -> 1024) -> `* sigmoid(gate)` (gate width 8192) -> `wo_b` (`h16`, `h18`). Sink: fp32, same semantics
  (`h17`).
* **KV layout.** Compressed rows: 584 bytes `fp8_ds_mla` (448 FP8 | 64 bf16 | 8 scale bytes) or 1024 bytes
  bf16; window cache in 64-token pages; compressor state 128 rows x 1024 fp32 per request and layer
  (`h13`, `h14`, `h15`).

### Group and chunk tables (ratio 128)

| group | tokens | visible from seq | rope position |
| ---: | --- | ---: | ---: |
| 0 | 0..127 | 128 | 0 |
| 1 | 128..255 | 256 | 128 |
| 8191 | 1048448..1048575 | 1048576 | 1048448 |

| chunk (start:len) | groups completed | boundary flag | state tokens left |
| --- | ---: | --- | ---: |
| 0:127 | 0 | false | 127 |
| 127:1 | 1 | true | 0 |
| 100:60 | 1 | true | 32 |
| 255:2 | 1 | true | 1 |
| 256:1024 | 8 | true | 0 |

Production arithmetic: at most `cdiv(1048576, 128) = 8192` compressed entries, sparse list width 8192.

## Reuse table

One row per candidate Radiance op (`reuse` section; `verdict` is reuse, reuse_with_adapter, mismatch or
new_op_required). Each row cites the Radiance source and the upstream behaviour it was compared with; every
mismatch names the Radiance symbol, the upstream symbol, the legal tensor shapes, the order, and an
independent contrasting fixture.

| Candidate | Verdict | Why (short) |
| --- | --- | --- |
| `ref_layernorm`, `ref_rmsnorm`, `ref_dsa_query`, `RadYarn` | reuse | same arithmetic as the pinned indexer k_norm, the norms, the index-query quantiser and the YaRN blend |
| `ref_mla_attn_paged` | reuse_with_adapter | closest to the DSA main attention (576-wide row, rope part, value = latent) but attends all keys, no sink, per-head scalar gate |
| `ref_attn_paged_sink` | reuse_with_adapter | window and sink semantics match exactly; no id list, no second pool, K then V slot layout |
| `ref_rope` | reuse_with_adapter | leading-slice rotary fits the indexer; HCA needs trailing slice and an inverse |
| `ref_row_topk` | reuse_with_adapter | valid selection; tie order of the pinned kernel unresolved |
| `RAD_KV_GROUPS` | reuse_with_adapter | FULL, WINDOW and LINEAR kinds cover the caches; ratio growth for compressed rows to confirm |
| `ref_mla_project`, `ref_mla_attn`, `MlaLatent` | mismatch | equal K/V widths, dense no-rope no-sink, GLM no-RoPE pooled pages |
| `ref_dsa_score`, `ref_dsa_pool`, `DsaCache` | mismatch | pooled-page scoring and ratio-4 pooling, top-k 512, no window; pool limits ratio <= 16, n <= 256 |
| `ref_seg_softmax_pool`, `ref_qk_norm_rope`, `ref_head_gate`, `ref_mix_rows`, `ref_attn_paged_anchor_window` | mismatch | scalar-per-row softmax, rounding order and slice, factor-2 per-head gate, row mixing, block-anchored window |
| `hca_compressor`, `hca_union_attention`, `swa_topk_merge`, `vha_mix`, `inverse_rope_grouped_oproj` | new_op_required | nothing in Radiance computes them |

## Fixtures

Pure-Python references of each equation with its tempting wrong variants; every contrast must differ from the
correct result (the tool refuses to run otherwise) and the sealed checker recomputes them independently:
`dsa-window-merge` (window endpoint, overlap and width slips), `hca-group-pool` (partial group counted, mean
pool, one score per row, no APE), `sink-denominator`, `gate-vha-order`, `hca-union-softmax`, `rope-pairing`,
`hca-compressed-rope-position`, `kv-b-unequal-widths`, `glm-geometry-vs-ernie`, `compressor-geometry-limits`.
They use reduced sizes (window 4, ratio 4) where a legal reduced period shows the same equations; a reduced
fixture proves only the paths it exercises, and production boundaries come from the tables above.

## What is unresolved (and what settles it)

| Id | Open fact | Resolver |
| --- | --- | --- |
| `m06` | order and tie handling of the indexer top-k kernels; slot fill beyond the valid count | read `top_k_per_row_prefill/decode` in vLLM v0.32.0 csrc; pin with a tied-score fixture in 3-1 |
| `m18` | dtype and rounding of the folded VHA matrix `m` | the size is measured (real layers 7 and 42: bf16 rounding moves entries by up to 7e-3, about one bf16 ulp; off-diagonals reach 0.47 to 0.62); which choice the reference makes needs vLLM's dtype context (construct the layer, or read the loader) |
| `m22` | bit-level numerics of FlashMLA's sparse forward, DeepGEMM logits, hadacore | read FlashMLA `0eee43b` and the hadacore source; measure tolerances against a reference run |
| `m25` | DSA latent cache dtype (bf16, FP8, fp8_ds_mla) | design decision in 3-2/3-5 after measuring agreement |
| `h12` | whether the YaRN branch is taken for the config's `type` key | CPU probe: instantiate `Ernie5_5Config` under the pinned transformers and print `rope_parameters` |
| `h21` | agreement of the CPU reference compressor with the CUDA CuTeDSL compressor | read `sparse_attn_compress_cutedsl.py` or compare both on one real group |
| `h23` | HCA window/compressed cache dtype | design decision in 3-3/3-4 |

Also outside the evidence: the sources were read at the pins named above; a different vLLM or Baidu revision
is a different audit. The compiled kernels were not read; semantic statements about them rest on in-tree
Python or CPU references (the ROCm Triton sink kernel, the XPU Triton q/kv kernel, the CPU compressor). The
report proves nothing about quality; real-weight numerics are for stories 5-x.

## For later stories

* 3-1: window `[seq-w, seq-1]`, indexer range `[0, seq-w)`, merged row layout, `swa_topk_merge` op, tie order `m06`.
* 3-2: positional split orders (`rope-pairing`), 576-wide latent row, absorbed MQA with unequal K/V widths,
  sink, VHA then element gate, absolute positions across chunks.
* 3-3: group completion at `(position + 1) % 128 == 0`, per-channel softmax with APE, group-start rope
  position, state rows, chunk-boundary rule, incomplete-group invisibility, no overlap lanes at 128.
* 3-4: single softmax over window plus complete compressed keys with the sink; the overlap is kept; inverse
  rope and grouped `wo_a` before the gate.
* 6-1: DSA identities are absolute tokens; HCA window is local tokens and compressed entries are group ids
  `(p+1)//128` complete at `seq = 128(j+1)`; one sink contribution per (query token, head); MXFP4 indexer needs
  PCP = DCP = 1 (`m26`), so initial parity uses the FP8 indexer cache.
