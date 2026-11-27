# ERNIE device dispatch inventory (story 4-1, CPU and inventory part)

`tools/ernie55/device_dispatch.py` turns the REAL declared graph (`radiance --debug-graph` on a converted ERNIE container) and the libr4d row table into
`docs/ernie55/device-dispatch.json`: per distinct operation the target (gfx1201), ABI (43), parameters (shape), dtype, cache layout, the host kernel the registry
chose, the device result (libr4d rows that list the operation and dtype, or the missing kernel named with its own bounded follow-up story, host parity and code map)
and the fallback (the libref kernel). gfx1201 compile-only commands are listed, not run (no hipcc on the CPU host). Row matching is by operation, dtype in the activation role (a whitelist of role words; kv ops match their cache storage strictly), and the shape clauses of the row that name a declared graph parameter (multiple-of, <=, >=, a..b, =, exactly; evaluated lexically): `kernel` means "a candidate row accepts this declared shape"; a shape every row rejects is `missing` with the host fallback and a `dev-*` follow-up. Every candidate keeps its own constraint text, checks and eligibility in `device.candidates`; clauses about parameters the graph does not carry (M, tiles, pitches) stay text.

Final matching rules (supersedes the first-round wording above): storage roles are matched, not words: a plain bf16 want accepts only rows that name no quantised storage (IQR, int8, fp8, i4), so a bf16 cache lists kv_store_bf16 only and a bf16 table lists no IQR row. Split-K rows (K divisible by SK*16, K/SK by 256) are checked by their necessary condition on K, so the tiny fixture GEMMs (K of 4 to 12) have no launching row and gemm_nt is listed missing at those shapes. Each ready entry says `verified`: true only when every constraint clause of an accepting row was evaluated; the summary splits `device_ready_verified` from `device_ready_unverified` (rows with M, pitch or prose clauses the graph cannot resolve). `dtype_source` says where an entry's dtype came from, or that the graph dump carries none (operand storage types and shapes of those operations are in their libref schema, named in the follow-up code map).

The committed JSON is generated from the tiny 3-6 fixture (4 layers: dense and routed, MLA-DSA and HCA). The production model declares the same operation set with larger
parameters; the schedule is 6 MLA-DSA + 37 HCA layers (3-6).

Result: every `ernie_*` operation (20 distinct) and `situ_mul` has no libr4d row: each is listed as missing with a `dev-*` follow-up story. Generic operations
(rmsnorm, embed_lookup, kv_store, the sampler family, ...) have a device row that accepts the fixture shapes; moe_gemm is listed missing at the fixture shapes (N, K of 4 or 5 against its multiple-of-64 rule: the production geometry is re-evaluated by running the tool on the production graph). The GPU stories 4-2, 6-8 and 7-3 stay on hold.

Three decisions, recorded in `device-dispatch-decisions.json` and pinned by the sealed 4-1 contract:

1. Indexer key quantiser rule: the CPU reference takes `quant_rule` = nvidia (default: floor 1e-10, /448) | rocm (floor 1e-4, /448) | rocm_fnuz (floor 1e-4, /224). A device kernel
   names its rule and is judged against that rule only. Which rule matches the shipped checkpoint on ROCm is decided by the first real-weights comparison. The FNUZ byte encoding is not emulated.
2. Whole-model near-tie indexer fixture: all indexer scores tie exactly; the served model must follow the lowest-index tie rule (and the highest-index rule is measurably different).
3. Parity band: `|device - reference| <= rope_angle_term(position) |q||k| + bf16_half_ulp |reference|`, rope terms 2e-5 (4k), 6e-5 (16k), 2.4e-4 (64k), 3.5e-4 (131k), 2e-3 (1M) (plain theta, enforced); the YaRN-measured terms 2.3e-4 (131k) and 1.2e-3 (1M) are carried explicitly beside them.

Hazard recorded: upstream f9dd69db (window-group positions relative to first_block_pos) needs a recycled-window fixture before the production RAD_KV_WINDOW HCA layout ships.

Round 7 notes (review): (1) the committed report uses the tiny fixture's shapes, so `gemm_nt` and `moe_gemm` read `missing` there although their rows launch at production geometry: `device-dispatch-production-probe.json` (graph `device-dispatch-production-probe.graph.txt`, hidden 4096, moe 2048, ffn 14336, head_dim 512) shows the same tool accepting them, and a shape-only `missing` carries `follow_up.kind` "rows exist, shape rejected: re-run at production geometry" instead of a new-kernel story. (2) libr4d rows the text parser cannot read (macro-built `attn_*_h#HD` names, `ar_*` all_reduce rows) are listed in `targets[gfx1201].unparsed_rows` and make their op read `missing` with that reason, never silently absent. (3) The parity band's bf16 storage term is scaled by |q||k| like the rope term, not by |reference|.
