# DSA score-plane tiling (ernie/dsa-score-tiling)

Status: CPU reference (libref + plugin) and the device op (libr4d, section 5) implemented; the device op is tested under the SIMT emulator only, never on a card.
Provenance tags: [CODE] read in source, [DERIVED] arithmetic from code facts, [UNVERIFIED] not measured.

## 1. What is being removed

`m.sc` is declared {T, Lmax} f32 in the activation arena (`arch/ernie55/ernie55.cpp`, `attn.idx_scores`), T = max-num-batched-tokens,
Lmax = max-model-len. It costs 4 B x T x Lmax per card (8 KiB per context position at T = 2048) and is checked against 2^30 scores
at declare (convert time and serve time use the same code path). The plane exists only because `ernie_idx_logits` writes one score
per (row, logical key) and `ernie_dsa_select` reads the whole row. Nothing else reads it.

## 2. Existing ops: is there a range or offset form? [CODE]

| op | width | offset/range form | consequence |
|---|---|---|---|
| `ernie_idx_logits` | L = `scores.shape[1]`, columns are logical ids 0..L-1 | none; column j is always id j | cannot score ids [t0, t0+tile) |
| `ernie_dsa_select` | L = `scores.shape[1]`, reads ids 0..end-1 | none; refuses `end > L` | cannot merge a tile into a running list |
| `row_topk` / `row_topk_merge` | per-row top-R of a dense row; the merge takes gathered per-rank candidate sets, ties to the LOWER id | `vocab_off` per rank (vocabulary sharding), no id offset for a context tile | the right tie rule; the TP form is a vocabulary shard, not usable as-is because ids are logical token ids with a page filter and `-inf` mask |

The indexer is replicated on every rank (one index key per token, KV unsharded, indexer weights `RAD_SHARD_NONE`), so there is no
rank sharding of the index to merge across; the only merge is across context tiles on one rank. [CODE: `ernie55.cpp` indexer
declarations, `docs/ernie55/indexer-scores.md`]

A per-issue column offset cannot be passed as an op parameter (parameters are fixed at declare; the one ranged parameter is `M`), and a
per-tile `rad_issue` loop in the plugin would make the number of issued ops depend on the step's context, which recorded passes
(graph replay, `abi/rad_runtime.h` "a recorded pass replays it") cannot do. So the tile loop belongs inside one op.

## 3. Design: one fused op, `ernie_dsa_topk_tiled`

Operands: qf, qs, w, idx cache, block_table_idx, req, positions, block_table_sel, scores (scratch), ids, slots, kept (scratch, see 5), and optionally seqused. It is
`ernie_idx_logits` followed by `ernie_dsa_select` with the plane cut in tiles:

```
for tile t0 = 0, tile, 2*tile, ... < max over rows of end            (end = max(p + 1 - window, 0))
    scores[m, jj] = idx_logit(m, t0 + jj)   for jj < min(tile, end_m - t0)       same per-column arithmetic as ernie_idx_logits
    for each row m:  for each column in range:
        NaN or +inf        -> refuse the launch (E_SHAPE) after the last tile, nothing stored
        candidate if score > -inf and the id's page in block_table_sel is initialised
        append (score, id) to the row's running list; when the list reaches 2 x topk keep the topk best
finally per row: if more than topk candidates in all: sort the kept list (score descending, id ascending) and keep topk,
                 else the list (never cut) is in ascending id order; then the window ids, -1 padding, slots.
```

Exactness (bit for bit), argued and tested:
* Scores: each (row, id) is computed by the same expression in the same order as `ernie_idx_logits`, independent of the tile it falls in.
* Order: candidates are ranked by the strict total order "score descending, id ascending" (ids are unique), the rule `ernie_dsa_select`
  cuts by (`libref/ref_dsa.cpp` `better`: higher score, or equal and lower id; HIP: `f32_order_key`, -0.0 canonicalised, then lowest id).
  The top-k of a union is the top-k of the union of the parts' top-k, so cutting to the `topk` best after every tile never loses a member of
  the global top-k. One pass of the merge needs only a strict comparison against the current k-th best: tiles run in ascending id
  order, so a later candidate EQUAL to the k-th best score has a higher id and loses.
* Output order: untiled select lists all candidates in ascending id order when there are at most `topk`, and the cut set in descending
  score order otherwise. The tiled op counts candidates over all tiles (`ncand`) and applies the same two cases; a list is never cut
  while `ncand <= topk`.
* Refusals: untiled select validates every row before the first store. The tiled op stores only after the last tile, and (device) only if no row of the launch was refused: a NaN or +inf score in a row's candidate range, a bad position or request, a candidate range past the `tokens` / context bound (section 5).
* Window and `-1` padding and slot translation are untouched code (same loops).

Scratch plane: [M, tile] rank 2, or flat rank 1 [numel >= M]. With a flat buffer a launch with M rows scores `numel / M` columns a tile:
the plugin declares one flat buffer of T x tile scores, so a prefill chunk (M = T rows) scores `tile` columns a tile and a decode step
(M = 1..64) scores `T x tile / M` columns, e.g. 2048 x 16384 / 8 = 4,194,304 columns, which is more than 1,048,576: ONE tile, the same
shape as untiled. [DERIVED] That is the property that keeps the dispatch count of decode unchanged.

Plugin switch: `ernie55_idx_tile=N` (container key, or `RADIANCE_ERNIE55_IDX_TILE` at serve time), default 0 = today's two ops and
plane. With N > 0 the arena holds T x min(N, Lmax) f32, the 2^30 cap applies to T x tile, and max-model-len itself is limited only to
2^30 (i32 positions); a log line `indexer score plane tiled: {T, tile} f32 = X MiB per card (untiled {T, Lmax} would be Y MiB)`
reports both.

Arena arithmetic [DERIVED]: tile 16,384 at T = 2048 = 128 MiB (vs 8,192 MiB at 1M); at T = 1024: 64 MiB (vs 4,096 MiB); tile 4,096:
32 MiB at T = 2048.

## 4. Cost

* FLOPs: identical. Every (row, key) is scored once.
* Memory traffic of selection [DERIVED from `r4d_ernie_mla.hip` `ernie_dsa_select_kernel`]: the untiled select kernel reads the row about
  6 times (candidate count, 4 radix passes, the greater-than gather, the tie pass), one block per row. Tiled with a threshold merge reads
  each tile's scores once to filter candidates `> k-th best` (equal never displaces, above) and runs the small 2048 + survivors selection
  only on the survivors. Expected survivors per tile fall as the running k-th best rises. Net: no more passes than untiled, probably fewer. [UNVERIFIED on GPU]
* Dispatches [DERIVED]: untiled = 2 per DSA layer per step (6 layers = 12). Tiled, HIP, one score kernel and one merge kernel per tile:
  2 x ceil(max_end / tile_eff) per layer. Decode with the flat scratch: tile_eff = T x tile / M, one tile up to 1M context for M <= 32 at
  T = 2048 and tile = 16,384 (2048 x 16384 / 32 = 1,048,576) = same 2 dispatches. Prefill chunk (M = 2048): 64 tiles at 1M context =
  128 dispatches a layer, 768 a chunk against a chunk that scores 2048 x up to 1M x 64 heads x 128 dims FP8 (about 17 TFLOP a layer at the
  deepest chunk), so the gaps are noise there. Dispatch gaps (about 29% of a decode step) therefore do not grow in decode. If a launch must bake a
  tile count (graph replay) the bound comes from the step's `max_ctx_len` band (host-side bound in `RadBatch`), not from the declared max.
* Interaction with the window: the window (the last `window` = 128 positions) is excluded from scoring (`end = p + 1 - window`) and
  appended after selection by the unchanged loop; tiles cover only [0, end). Prefill chunks: row m of a chunk has its own `end`; rows
  with smaller `end` just have fewer columns in range (`-inf` beyond), the loop bound is the maximum over rows.
* Host RAM: none. CPU reference only: the running lists, up to 2 x topk entries per row.
* Device memory: the scratch plane plus the kept-list scratch, T x (2 x topk + 3) i32 (32 MiB at T = 2048, topk = 2048; 16 MiB at T = 1024).

## 5. The device side (libr4d), as built
Built and verified under the SIMT emulator only (never on a card).
* `ernie_idx_logits` takes an optional `col_start` (libref and libr4d): column j is logical id `col_start + j`. The per-key arithmetic of the WMMA and fallback kernels is unchanged for every finite input, so tile scores equal the untiled plane's columns bit for bit. The relu is `relu_keep_nan` (a NaN stays a NaN, as libref's `std::max`); `fmaxf` returned 0 for a NaN, which hid a NaN key scale from every refusal. This is shared with the untiled op.
* `ernie_dsa_merge_kernel` (one block per row, the select kernel's radix threshold, strictly-greater gather, lowest-id tie take and bitonic sort over kept list + tile candidates) and `ernie_dsa_finalize_kernel` (slots, window, -1 padding).
* The running top-k lives in the op's scratch operand 11, `kept` (flat i32, at least M x (2 x topk + 3)): per row keys, ids, the kept count and the candidates seen, then one verdict word a row (contiguous after the M rows). `ids` and `slots` have one writer, the finalize kernel.
* Refusal contract (libref's: NaN or +inf score in [0, end), bad position or request, a candidate range past the bound; RAD_E_SHAPE, nothing stored). The merge kernel scans each tile's scores in the pass that counts candidates and starts the row's verdict with the first tile; the finalize kernel reads all M verdict words and every block takes the same decision, so one bad row anywhere stores nothing for any row. The verdict is published in `g_dsa_bad` (a synchronous launcher reads it) and raised as ONE event into `g_dsa_log` (bits: 1 non-finite score, 2 bad position or request, 4 range past the bound), the event log every ERNIE validator uses (`libr4d/r4d_ernie_verdict.h`: monotonic, a take never clears device state), read by `r4d_ernie_dsa_take_error()` and folded into `rad_kernel_poll_error`. Host builds (the emulator) also scan positions, requests and the bound in the launcher before the first launch (`R4D_DSA_VALIDATE=0` or `R4D_ERNIE_CARD_FORM=1` turn that scan off, the card situation) and return RAD_E_SHAPE; on a card (and in the emulator's card form) the launch returns RAD_OK, stores nothing, and the event is the report: the engine polls it before it commits the step's tokens and fails the step (`RADIANCE_POLL_KERNEL_ERRORS`, `R4D_ERNIE_VALIDATE=0` switch the poll off).
* `r4d_rad_ernie_dsa_topk_tiled` loops tiles from a host-side context bound (`RadArgs::kv_used_max_h`, set by the engine when the optional `seqused` operand is passed; else the `tokens` parameter; the smaller of the two when both exist; neither: refused). The row declares `RAD_KERNEL_CAP_DYNAMIC_PREFILL_SPLIT` so prefill/mixed passes are not recorded.
* Dispatches per DSA layer: `2 x tiles + 1` (untiled 2), at least 3. With the flat scratch a decode step is one tile. The refusal mechanism adds no launch; it adds the verdict scan in the finalize kernel (M words a block, from L2) and one NaN / +inf compare on the value the merge's counting loop already loads.

## 6. Evidence on the CPU reference

* `ref_test`: `op_ernie_dsa_topk_tiled_matches_untiled` (26 cases, bitwise ids and slots against `ernie_idx_logits` + `ernie_dsa_select` over a plane as wide as the context: tile not dividing the context, ctx a multiple of the tile, ctx = tile * k + 1, ctx = tile * k - 1, tile wider than ctx, tile 1, tile = block, tile < block and not dividing it, fewer candidates than topk, ctx shorter than topk + window and than window, three distinct keys (heavy ties), all keys equal (the lowest ids win), uninitialised pages, block 16, ctx 1000, flat rank-1 scratch with M = 1 and M > 1) and `..._refuses_like_untiled` (NaN key scale refuses with E_SHAPE and nothing stored, bad request, `tokens` bound). Red log `red-ref_test-tiled.log` (32 failures, all the new tests plus the op vocabulary list), green `green-ref_test-tiled.log`.
* End to end on the synthetic target (`tests/ernie55/score-tiling-e2e.cjs`): single-sequence greedy ids and top-5 logprobs, 18 runs / 626 steps, tiled (tile 1 and 8) against untiled at TP1 and TP2: 0 differing runs, prefix counters equal (`e2e/compare.txt`). The synthetic model has `index_topk` 4, so the cut path (more candidates than topk) is exercised on every prompt longer than about 12 tokens.
* Not covered: libr4d, any GPU, the real checkpoint, the batch-invariance caveat (all identity gates are single-sequence).
