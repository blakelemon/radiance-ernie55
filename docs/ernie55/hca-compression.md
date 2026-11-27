# ERNIE 5.5 HCA compressor: state and index visibility (story 3-3)

This is a discovery report, not an implementation. It records how the pinned reference compresses the KV stream of
the 37 HCA layers: when a compressed row exists, what state it needs, what a query can see, and with which dtype and
rounding each stage runs. The machine-readable form is `docs/ernie55/hca-compression.json`, written by
`tools/ernie55/hca_compression.py` from the pinned `config.json` (every fact carries its source as data:
pin, file, line, quote). No Radiance op, ABI field or kernel is added by this story; ABI 40 and the stock path are untouched.

Sources: Baidu vLLM-ERNIE `b8414da6b703579814c2009c5a63bac101916bbe`, vLLM `v0.32.0` (commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`), Hugging Face revision `028544b2ffceea189b2d8768ed2b9d09b06f2343`, this repository at
`a767783`. Regenerate with `python3 tools/ernie55/hca_compression.py --metadata-dir <dir with config.json> > docs/ernie55/hca-compression.json`;
`--verify-sources <tree>` re-reads all 148 upstream and repository citations, `--verify-headers <model dir>` checks all 148 compressor tensors of the checkpoint.

## Periods (k01, k02, k16, k17)

The compressor accepts exactly two periods, **128** and **4** (`assert compress_ratio in [4, 128]`). The checkpoint uses
only 128 (config entries 128 and -2). Period 4 is the only other legal one and the only one with **overlap** lanes and an indexer.

| | period 128 (production) | period 4 (small legal) |
|---|---|---|
| lanes per state row (coff) | 1 | 2 |
| window rows pooled per group | 128 | 8 |
| state row | kv 512 + score 512 fp32 = 4096 B | kv 1024 + score 1024 fp32 = 8192 B |
| APE | fp32 [128, 512] | fp32 [4, 1024] |
| state page rows | 8 | 4 |
| compressed row (fp8_ds_mla / bf16 full / fp8 full) | 584 / 1024 / 512 B | same |
| in the checkpoint | yes, 37 layers | no |

The checkpoint holds per HCA layer `compressor.ape` F32 [128, 512], `linear_wkv.weight` and `linear_wgate.weight` BF16 [512, 4096]
and `norm.weight` BF16 [512]: 148 tensors in 37 layers, none in the six DSA layers or the draft layer (checked for all of them).
The loader maps `linear_wkv` to shard 0 (kv) and `linear_wgate` to shard 1 (score) of the fused projection (k15).

## State, group completion and masking (k03 to k06, k14)

Every token first writes its fp32 state row `[kv | score + APE[position % ratio]]`; the APE row is selected by the **absolute**
position, so pooling does not depend on the chunking. Only the token with `(position + 1) % ratio == 0` compresses (k04); it
pools the `coff * ratio` rows at positions `position - coff*ratio + 1 .. position` (its own row included, written just before).
Rows before position 0 are masked (score -inf, kv 0): only the first group of a period-4 stream has any.
A chunk that completes a group reads rows written by earlier chunks (127 rows at period 128 for a chunk `127:1`; 7 at period 4),
so the result is chunk invariant exactly when those rows persist; nothing older than `coff*ratio - 1` rows behind the next token is read.

## Overlap lanes (k07)

At period 4 the window has 8 rows. Rows with window index `w < 4` belong to the **previous** group and read channels [0, D) of their
state row; rows `w >= 4` belong to the **current** group and read channels [D, 2D). The softmax is per channel over all 8 rows. Each
token row therefore carries two lanes (lane 1 for its own group, lane 0 for the next group). Group 0 has no previous rows. At period 128
there is one lane and a group never reads the previous group's rows. The report tabulates the rows of groups 0 to 3 at period 4.

## Visibility (k05, k18)

A query at position `pos` sees compressed rows `0 .. (pos + 1)//ratio - 1`: group `j` becomes visible at the step of its last token,
`pos = j*ratio + ratio - 1` (sequence length `j*ratio + ratio`), including to that token itself, because the compressor runs before the
sparse attention of the same layer. The incomplete tail is invisible (the window covers it). Boundary table:

| position | 126 | 127 | 128 | 254 | 255 | 256 |
|---|---|---|---|---|---|---|
| visible groups, period 128 | 0 | 1 | 1 | 1 | 2 | 2 |

| position | 2 | 3 | 4 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| visible groups, period 4 | 0 | 1 | 1 | 1 | 2 | 2 |

The compressed row written by the group completed at `pos` is row `pos // ratio` of the compressed cache; the slot mapping marks only
boundary tokens valid (PAD elsewhere and for replayed tokens). The CUDA path skips the compress kernels for a step when no request crosses a
boundary (`start % 128 + query_len >= 128`), the same predicate as the report's `boundary_flag`.

## Absolute group anchoring (k11)

The rope part (trailing 64 channels, GPT-J interleaved) of group `g` is rotated at position `g * ratio`, the group **start**, not at the completing
token and not at the next start. The CPU and Triton kernels compute `(position // ratio) * ratio`; the fp8_ds_mla CuTeDSL kernel computes
`position - (ratio - 1)`; the two are equal wherever the kernel runs. The rotation uses the un-rounded fp32 normalised row and is rounded to bf16 once.
The fixtures use plain RoPE: they prove the anchor, not the YaRN angles (k22).

## Rounding and dtype at each stage (k08 to k13)

1. projection: bf16 hidden states times the fused weight, **fp32** output (kv first, score second);
2. state write: kv unchanged fp32, score plus APE in fp32;
3. window gather with lanes and masks (no rounding);
4. per-channel softmax pool in fp32 (CPU `exp`, CuTeDSL `exp2` with fastmath and an online combine over 8 warps);
5. RMSNorm over all 512 channels in fp32 with the compressor's own bf16 weight widened, `rms_norm_eps` (CPU `1/sqrt`, CuTeDSL fastmath `rsqrt`);
6. nope channels [0, 448) rounded to **bf16**; per block of 64: `scale = max(absmax/448, 1e-4)`, `exponent = ceil(log2(scale))`, byte `exponent + 127`,
   value times `2^-exponent`, clamp +-448, **FP8 E4M3** round to nearest even (the 8th scale byte is zero padding);
7. rope channels rotated in fp32 from the un-rounded row, one rounding to **bf16**.

Layouts (k13): `fp8_ds_mla` 584 B per row (448 FP8 + 128 B bf16 rope + 8 scale bytes); `bf16_full` 1024 B (nothing FP8; the nope is only rounded to bf16);
`fp8_full_per_tensor` 512 B plus one scale. The layout follows the layer's KV cache dtype.

## Agreement of the three implementations (h21 narrowed)

The 3-7 audit left h21 open: does the CPU kernel (used as the statement of the compressor) agree with the CUDA CuTeDSL kernel? Read
`sparse_attn_compress_cutedsl.py`, `fused_compress_quant_cache.py` (Triton) and `compressor.cpp`: they agree on the boundary gate,
window and masking, the lane offsets, the per-channel softmax, the bf16 round before the quantiser, the UE8M0 scale rule (CPU = CuTeDSL), the
rope anchor and the single bf16 rope rounding. They differ in rounding detail: `exp2` fastmath and summation order in the pool, fastmath `rsqrt`, and
**the Triton path floors the absmax at 1e-4 where CPU and CuTeDSL floor the scale**: the scale differs for blocks with bf16 absmax below 0.0448 and the
stored exponent byte for absmax at or below 0.02734375 (448/2^14). Probe of the real norm weights of layers 0, 1, 41 (per-block minimum |weight| 0.71 to 1.20) makes
such blocks implausible for non-degenerate rows (k21). Bit-level agreement with CUDA is not established (k19).

## Fixtures (scalar reference, seeded)

The tool runs a stage-rounded scalar reference (double arithmetic with explicit fp32, bf16 and FP8 rounding between stages) on seeded inputs
(head_dim 16, rope 4, block 4) and refuses to run if a contrast does not discriminate:

* `chunk_invariance_period4` (23 tokens: 5 complete groups, 3 in the incomplete tail) and `chunk_invariance_period128` (300 tokens: 2 groups, 44 tail)
  with one-shot, irregular, decode-sized and group-splitting chunkings: identical complete rows; the incomplete tail never changes a row;
  dropping the state between chunks or pruning one row too many changes rows;
* `anchor_contrast_period4`: anchoring at the completing token or the next start changes the rope values of every group after the first;
* `scale_rule`, `fp8_e4m3_rounding`: the two floor rules and tie rounding;
* `layout_error`: relative L2 error against the fp32 normalised row on synthetic unit-RMS rows: fp8_ds_mla nope 2.66e-2, bf16 nope 1.67e-3, bf16 rope 1.71e-3
  (FP8 costs about 16 times the bf16 rounding). Per-tensor fp8 rows are unmeasured.

The sealed test `tests/3-3-hca-compression.acceptance.cjs` runs its own implementation of the same semantics (group-wise lane form, other widths and seeds) and compares it
with the tool's numbers.

## Cache-state API (k18)

Per HCA layer: (1) `state_write` per token, (2) `compress` per boundary token, (3) `compressed_slot` per token, (4) `visibility` per query; rules in the report's
`state_api`. The state cache and the compressed cache are separate caches with separate block tables.

## Code map for the implementation story (3-3b)

`stories/3-3b-hca-compressor-op.md` (draft) schedules two host ops in a new `libref/ref_ernie_hca.cpp` following the pattern of 3-2a: `ernie_hca_state_write` and
`ernie_hca_compress`, registered in `libref/ref_ops.h`, `libref/ref_registry.cpp` and `libref/ref_shapes.cpp`, documented in `docs/OPS.md`, tested in `tests/ref_test.cpp`; the
plugin wiring in `arch/ernie55/ernie55.cpp` is a later story. Gap: the ABI has no KV group that stores one row per `ratio` tokens (the compressed rows),
see the `gap` field of the code map.

## What is not established (unresolved)

| id | what | resolver |
|---|---|---|
| k19 | bit-level numerics CPU vs CuTeDSL (h21 remainder) | runtime-probe on an NVIDIA machine |
| k20 | cache layout of compressed rows (h23); proposed default full bf16 | design-decision in 3-3b with real rows from 3-6 |
| k21 | FP8 scale-floor rule for non-CUDA targets | runtime-probe with real hidden states |
| k22 | whether the YaRN branch is taken (h12, inherited) | source-read / probe of the rope parameters |

Real weights were read only as safetensors headers (148 compressor tensors) and the three 1 KiB `norm.weight` tensors above; no compressor row was computed from real data.

## Fact index

Every fact below is in the report with its citations (`facts[]`); `unresolved` ones carry `why`, `impact` and a resolver.

| id | topic | status |
|---|---|---|
| k01 | periods | established |
| k02 | state_geometry | established |
| k03 | state_write | established |
| k04 | completion | established |
| k05 | visibility | established |
| k06 | window_gather | established |
| k07 | overlap_lanes | established |
| k08 | arithmetic | established |
| k09 | nope_quant | established |
| k10 | nope_quant_variant | established |
| k11 | rope_anchor | established |
| k12 | cuda_path | established |
| k13 | cache_layouts | established |
| k14 | chunk_persistence | established |
| k15 | subclass | established |
| k16 | kernel_constants | established |
| k17 | weights | established |
| k18 | state_api | established |
| k19 | numerics | unresolved |
| k20 | cache_dtype | unresolved |
| k21 | scale_rule | unresolved |
| k22 | yarn_branch | unresolved |
