# MLA-DSA global selection (story 3-1)

Operator: `ernie_dsa_select` (`libref/ref_dsa.cpp`, host only, ABI 40 untouched). It builds the
selection row one MLA-DSA query attends to, for a single rank. Operand and parameter spelling is in
`docs/OPS.md`; the facts it implements are `m03` to `m06` of `docs/ernie55/attention-reference.md`.

## The row

For the query at position `p` (`seq = p + 1`), `window = 128`, `index_topk = 2048` (checkpoint
`config.json`: `index_topk` 2048, `sliding_window` 128):

* candidates: token ids in `[0, seq - 128)`, half open, empty when `seq <= 128`;
* selected: `min(max(seq - 128, 0), 2048)` of them; truncation starts at `seq = 2177`;
* window: ids `seq - min(seq, 128) .. seq - 1`, ascending, the query's own token last;
* row: `[selected | window | -1 ...]`, width `2048 + 128 = 2176`.

Selection runs on logical token ids. The indexer scores arrive logically indexed (`ref_dsa_score`
already follows the block table to find a key's page). The physical slots are produced last, from
the selected ids, so a reversed or shuffled page table moves `slots` and never `ids`.

Beyond the pinned row, this op also treats these as not selectable: a score of `-inf` (the scorer's
mask) and a token whose page is uninitialised (table entry negative or at or past `num_pages`, or a
block past the table). An illegal window token is dropped, not translated, so the window and the
candidates never name an uninitialised page. Columns at or past `seq - 128` of `scores` are never
read (future tokens, never-written scores). A NaN or `+inf` score inside the candidate range, a bad
position or request, or scores narrower than the candidate range refuse the launch before any
store; the op is also refused above its row band.

## m06: order and ties of the indexer top-k (resolved from source, with one assumption)

Read at vLLM `v0.32.0`: `csrc/sampler.cu` (`top_k_per_row_prefill`, `top_k_per_row_decode`,
`topKPerRowJob`, sha256 `2e8cf659...3a90`) and `csrc/persistent_topk.cuh` (the decode selector,
sha256 `e41fda5b...20ac2`).

* Rows with at most `topk` entries: the kernels write ids `0 .. len-1` in ascending order and `-1`
  after them, not sorted by score (`sampler.cu` `topKPerRowJob` shortcut, "Indices are not sorted by
  their corresponding logit", lines 403-420; `persistent_topk.cuh` 936-938 and 1056-1058,
  `(i < seq_len) ? i : -1`). The op reproduces this: a candidate set that fits is listed ascending.
* Rows longer than `topk`: ids strictly above the threshold bin are written in `atomicAdd` arrival
  order (`sampler.cu` line 281), the threshold bin is ranked by an insertion sort or a block radix
  sort over items whose slot order itself came from an `atomicAdd` (lines 488, 541), and in the last
  radix step the first `topk` arrivals win (`dstIdx < topK`, line 309). So **the order inside the
  selected ids and which of several equal scores make the cut are not specified by the kernels**;
  they depend on thread arrival. Attention is invariant to candidate order up to summation order,
  and only a tie at the k-th place can change the selected set.
* Slots past the valid count: `-1` (kernel) and, after the merge, `-1` (`merge_swa_topk` masks
  every indexer slot at or beyond `n`, `m05`).

Decision (an assumption, stated as one): for a truncated row the op lists the best `topk`
candidates in descending score with ties to the **lower id**, exactly `row_topk`'s order and
Radiance's convention, which is one legal outcome of the kernel. A real-GPU run may pick another of
the tied ids; only the real model (epic 5) can show whether any checkpoint input ties at the k-th
place in practice. The sealed test pins the choice with tied-score fixtures
(`ties at the k-th place go to the lower id ...`).

## Reuse verdicts (from the 3-7 table)

* `ref_dsa_score` stays the indexer score op; its audited mismatch (it masks by live blocks per
  request, not per token by `[ks, ke)`) is not fixed here. The selection op does its own masking
  from `positions`, which is the per-token rule, and ignores scores outside the candidate range. The
  contract composes the two ops (`dsa_score` output into `ernie_dsa_select`, reversed/shuffled
  tables, non-unit key scales).
* `ref_row_topk` semantics are reused (descending, ties to the lower id); the op reimplements the
  comparator because it needs a masked, per-row-ranged input. `tests/ref_test.cpp`
  (`op_ernie_dsa_select`) cross-checks the two on an unmasked row.
* Not covered here: the indexer's score math beyond `dsa_score` (q/k split, RoPE, Hadamard,
  LayerNorm: `m08`..`m10`), cache dtypes (`m25`), prefill chunk metadata (`cu_seqlen_ks/ke`) and
  the DCP merge. Those belong to later stories.
