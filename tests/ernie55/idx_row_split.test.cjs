'use strict';
// The indexer row split (RADIANCE_ERNIE55_IDX_ROW_SPLIT=1; arch/ernie55/ernie55.cpp, commit 4b4cdb34), on the
// tiny synthetic ERNIE + DSpark checkpoint (support/dspark_ckpt.cjs) on the CPU reference kernels. Under
// tensor parallelism (world >= 2) each rank runs the DSA indexer chain (ernie_idx_logits + ernie_dsa_select,
// or ernie_dsa_topk_tiled when the model is converted with ernie55_idx_tile) for its own run of a step's rows
// and an all_gather joins the runs' slot lists for ernie_mla_sparse_attn; the key side and the query-side
// projections stay whole on every rank, so each row's selection is bit-identical to the replicated one's. A
// step splits when T >= 256 rows, or T >= 16 and T*max_ctx_len >= 2^18 (idx_split_step). The switch is off by
// default and inert at TP 1. The collectives overlap is declined when the row split is on (ov_declare).
//
// The fixture is tiny (max-model-len 1024), so the "T >= 16 and T*max_ctx_len >= 2^18" rule cannot trigger
// here: with max_ctx_len <= 1024 that bound needs T >= 256, which the first rule already covers. The split is
// therefore exercised only through the T >= 256 (prefill-sized chunk) rule; the second rule is NOT forced.
//
// Cases:
//   (a) declare at TP 2, switch off (unset and '0'): no indexer all_gather, and the declared graph equals the
//       switch-unset one (graphText drops the engine log lines that float between runs).
//   (b) declare at TP 2, switch on: the log says "indexer row split on:" and the graph gains the f32 slot-list
//       all_gather(s); at TP 1 the switch is inert (no "on" line, graph unchanged).
//   (c) identity at TP 2, on vs off, tokens and top-3 logprobs bit-identical, for: (i) 256-row prefill chunks
//       (a ~700-token prompt -> 3 chunks, ragged last); (ii) an odd row count (the last rank's run is short);
//       (iii) two concurrent requests of different lengths (a multi-sequence step); (iv) the DSpark drafter on
//       (spec 7), so the verify rows pass through the indexer; (v) score tiling on (ernie55_idx_tile), the
//       ernie_dsa_topk_tiled path.
//   (d) both RADIANCE_ERNIE55_IDX_ROW_SPLIT=1 and RADIANCE_ERNIE55_AR_OVERLAP=1 at TP 2: the server starts, the
//       overlap is declined with a reason naming the indexer row split, and the output equals the plain run.
//   In (c-i..c-v) and (d) the ON arm also sets RADIANCE_ERNIE55_IDX_ROW_SPLIT_TRACE=1 and the per-rank trace is
//   checked (a T >= 256 step; r0 = rank*run and per-rank rows summing to T; in (c-ii) an odd T >= 257 with unequal
//   rows); the OFF arm sets the trace env too and must emit no trace line.
//
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:
//   nosys node --test tests/ernie55/idx_row_split.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ck = require('./support/dspark_ckpt.cjs');
const {h} = ck;

const SW = 'RADIANCE_ERNIE55_IDX_ROW_SPLIT';
const AR = 'RADIANCE_ERNIE55_AR_OVERLAP';
const TR = 'RADIANCE_ERNIE55_IDX_ROW_SPLIT_TRACE';    // per-rank, per-step row-split trace (arch/ernie55/ernie55.cpp)
const NT = Number(process.env.IRS_NT || 40);          // decode rows a request
// The trace env is set in BOTH arms: the ON arm must emit per-rank trace lines, the OFF arm must emit none (the
// split branch is not taken with the switch off). The switch in OFF is set to '0' explicitly, never inherited.
const OFF = {[SW]: '0', [TR]: '1'};
const ON = {[SW]: '1', [TR]: '1'};

let _dir, _model, _tiled;
function getModel() {
  if (!_model) {
    const c = ck.buildCheckpoint('idx-row-split');
    const conv = ck.convertDspark(c.dir, 7, 'idxrs');   // max-spec 7: the drafter is available to case (c-iv)
    assert.equal(conv.status, 0, 'convert must work: ' + (conv.stderr || '').slice(-600));
    _dir = c.dir;
    _model = conv.out;
  }
  return _model;
}
// The same checkpoint converted with the score-tiling key (ernie55_idx_tile=8, as score-tiling-e2e.cjs sets it),
// so ernie_dsa_topk_tiled is declared and the row split drives the tiled op (case (c-v)). Mirrors convertDspark's
// keys/rules plus the tile key.
function getTiledModel() {
  if (!_tiled) {
    getModel();
    const conv = h.convert(_dir, {
      keys: ['ernie55_quant_forward=1', 'ernie55_dspark=1', 'ernie55_attention_block=8', 'ernie55_hca_period=4', 'ernie55_idx_tile=8'],
      rules: [...h.RULES, ...ck.DSPARK_RULES], extra: ['--max-spec', '7'],
    });
    assert.equal(conv.status, 0, 'the tiled convert must work: ' + (conv.stderr || '').slice(-600));
    _tiled = conv.out;
  }
  return _tiled;
}

// `radiance --config` with debug-graph at the given tp: rank 0's declared program, printed instead of served
// (Engine::dump prints ranks_[0]->program; the host backend needs no card). `env` is merged into the spawn
// environment; the switch is removed when the caller does not pass it, so "unset" is really unset and an
// ambient RADIANCE_ERNIE55_IDX_ROW_SPLIT cannot turn an OFF run on. Memoised per (tp, env).
const _dg = new Map();
let _dgSeq = 0;
function debugGraph(tp, env = {}) {
  const key = `${tp}|${JSON.stringify(env)}`;
  if (_dg.has(key)) return _dg.get(key);
  const b = h.ensureBuilt();
  const y = path.join(h.tmpRoot, `idxdg${tp}-${_dgSeq++}.yaml`);
  fs.writeFileSync(y, ['server:', '  host: 127.0.0.1', '  port: 0', '  log: info', `  radiance-home: ${b.home}`, '  kernels: libref',
    '  debug-accept-reference-kernels: true', 'cards: all', 'models:', '  ern:', `    model: ${getModel()}`, `    tp: ${tp}`,
    '    debug-graph: true', '    max-num-seqs: 2', '    max-num-batched-tokens: 256', '    max-model-len: 1024',
    '    num-speculative-tokens: 0', ''].join('\n'));
  const spawnEnv = {...process.env, ROCR_VISIBLE_DEVICES: '99', RADIANCE_HOST_DEVICES: String(tp), RADIANCE_HOST_VRAM_MIB: '512', ...env};
  if (!(SW in env)) delete spawnEnv[SW];
  const r = h.sh(b.radiance, ['--config', y], {env: spawnEnv});
  const out = {status: r.status, text: r.stdout + r.stderr};
  _dg.set(key, out);
  return out;
}

// The declared graph only (ck.graphText): the `==== declared graph:` section with engine log lines and the
// `radiance: ` plugin notes dropped (the notes are concatenated after stdout, so slicing from the banner does
// not drop them) and the run-dependent `---- unresolved:` table sorted. Shared with dspark_kv_window.test.cjs.
const graphText = ck.graphText;
// The row split's slot-list gather moves i32 rows as f32 bytes (make_ag_max(..., "f32")); it is the only f32
// all_gather the program declares, so counting f32 all_gather op headers counts the indexer gathers.
const nF32Gather = g => (g.match(/^all_gather\b[^\n]*dtype=f32/gm) || []).length;

const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
// Tokens and top-3 logprobs only (genFull's `top` entries are "id@logprob" strings, so this is bit for bit).
// The drafter's drafted/accepted counters are wall-clock dependent and are not compared here.
function sameTokens(a, b, what) {
  assert.ok(eq(a.ids, b.ids), `${what}: tokens differ\n  off ${a.ids.join(',')}\n  on  ${b.ids.join(',')}`);
  assert.ok(eq(a.top, b.top), `${what}: top-3 logprobs differ`);
}
// One engine step's shape from RADIANCE_LOG_STEPS: `step N <phase> n_seq=K n_tok=T ...`.
const steps = log => [...log.matchAll(/step \d+ (?:prefill|decode|MIXED) n_seq=(\d+) n_tok=(\d+)/g)]
  .map(m => ({nseq: Number(m[1]), ntok: Number(m[2])}));
// Teeth that a splittable (>= 256-row) step really ran, so the ON arm exercised the split and not the serial path.
function assertSplitRan(log, what) {
  const st = steps(log);
  assert.ok(st.length > 0, `${what}: no step lines captured (RADIANCE_LOG_STEPS)`);
  assert.ok(st.some(s => s.ntok >= 256), `${what}: no step reached the 256-row split threshold (max n_tok ${Math.max(...st.map(s => s.ntok))})`);
}

// The per-rank row-split trace (RADIANCE_ERNIE55_IDX_ROW_SPLIT_TRACE=1), one line per rank per split step:
//   radiance: ernie55: idx row split step: T=.. run=.. r0=.. rows=.. rank=.. world=..
const TRACE_RE = /idx row split step: T=(\d+) run=(\d+) r0=(\d+) rows=(\d+) rank=(\d+) world=(\d+)/g;
const traceLines = log => [...log.matchAll(TRACE_RE)].map(m => ({T: +m[1], run: +m[2], r0: +m[3], rows: +m[4], rank: +m[5], world: +m[6]}));

// Assert the trace teeth for an ON arm. Both ranks' stderr reaches srv.log: the host backend runs one thread per
// rank in a single process (core/device/host.cpp), so every rank's fprintf(stderr) is captured, and RADIANCE_LOG_STEPS
// (engine.cpp) is captured the same way. The per-step indexer all_gather keeps a step's `world` lines consecutive, so
// the trace chunks into per-step groups of `world`. Each group shares T and run; rank r has r0 = r*run and the rows
// sum to T across ranks 0..world-1. At least one group must reach T >= 256. With {odd}, also require an odd T >= 257
// group whose per-rank rows are unequal (the last rank's run is short).
function assertTrace(log, what, {odd = false} = {}) {
  const tl = traceLines(log);
  assert.ok(tl.length > 0, `${what}: no idx-row-split trace line captured (RADIANCE_ERNIE55_IDX_ROW_SPLIT_TRACE=1)`);
  const world = tl[0].world;
  assert.ok(world >= 2, `${what}: trace world ${world} < 2 (the split is inert at TP 1)`);
  assert.ok(tl.every(t => t.world === world), `${what}: trace lines disagree on world (${[...new Set(tl.map(t => t.world))]})`);
  assert.equal(tl.length % world, 0, `${what}: ${tl.length} trace lines is not a multiple of world ${world}`);
  let big = 0, oddBig = 0;
  for (let i = 0; i < tl.length; i += world) {
    const g = tl.slice(i, i + world), T = g[0].T, run = g[0].run;
    assert.ok(g.every(t => t.T === T && t.run === run), `${what}: a step's per-rank lines disagree on T/run: ${JSON.stringify(g)}`);
    const ranks = g.map(t => t.rank).sort((a, b) => a - b);
    assert.ok(eq(ranks, Array.from({length: world}, (_, j) => j)), `${what}: a step's ranks are not 0..${world - 1}: ${ranks}`);
    for (const t of g) assert.equal(t.r0, t.rank * run, `${what}: rank ${t.rank} r0 ${t.r0} != rank*run ${t.rank * run} (T ${T})`);
    assert.equal(g.reduce((s, t) => s + t.rows, 0), T, `${what}: per-rank rows do not sum to T ${T}: ${JSON.stringify(g)}`);
    if (T >= 256) big++;
    if (odd && (T % 2) === 1 && T >= 257) {
      oddBig++;
      assert.ok(new Set(g.map(t => t.rows)).size > 1, `${what}: odd T ${T} split into equal per-rank rows: ${JSON.stringify(g)}`);
    }
  }
  assert.ok(big > 0, `${what}: no trace step reached T >= 256 (max T ${Math.max(...tl.map(t => t.T))})`);
  if (odd) assert.ok(oddBig > 0, `${what}: no odd trace step with T >= 257 (the odd-T case did not exercise an odd split)`);
}
// The OFF arm sets the trace env too but must emit no trace line (the split branch is not taken with the switch off).
const assertNoTrace = (log, what) => assert.equal(traceLines(log).length, 0, `${what}: the switch-OFF arm emitted an idx-row-split trace line`);

// (a) declare at TP 2 with the switch off: no indexer all_gather, and the declared program is the switch-unset one.
test('(a) TP2 declare, switch off (unset and 0): no indexer all_gather, graph equals unset', () => {
  const unset = debugGraph(2, {});
  const zero = debugGraph(2, {[SW]: '0'});
  assert.equal(unset.status, 0, unset.text.slice(-800));
  assert.equal(zero.status, 0, zero.text.slice(-800));
  assert.equal(graphText(zero.text), graphText(unset.text), 'the declared program changed with the switch set to 0');
  // The program already declares f32 all_gathers at TP 2 (the drafter's candidate gathers), so the indexer's gather is checked as a delta in case (b), not as an absolute count here.
});

// (b) declare with the switch on: TP 2 turns it on (the log line, the extra f32 gather); TP 1 is inert.
test('(b) declare, switch on: TP2 on (log line + indexer all_gather), TP1 inert and unchanged', () => {
  const off = debugGraph(2, {});
  const on = debugGraph(2, {[SW]: '1'});
  assert.equal(on.status, 0, on.text.slice(-800));
  assert.match(on.text, /^radiance: ernie55: indexer row split on: /m, 'no "indexer row split on:" line at TP2: ' + on.text.slice(-800));
  const nOff = nF32Gather(graphText(off.text)), nOn = nF32Gather(graphText(on.text));
  assert.ok(nOn > nOff, `the ON graph has no extra indexer (f32) all_gather (${nOn} on vs ${nOff} off)`);
  // At TP 1 the switch turns itself off ("it needs tensor parallel 2 or more"): no "on" line, and the declared
  // program is byte-identical to the switch unset (the "requested but off" note is before the graph banner).
  const t1off = debugGraph(1, {});
  const t1on = debugGraph(1, {[SW]: '1'});
  assert.equal(t1on.status, 0, t1on.text.slice(-800));
  assert.doesNotMatch(t1on.text, /^radiance: ernie55: indexer row split on: /m, 'the row split turned on at TP1');
  assert.equal(graphText(t1on.text), graphText(t1off.text), 'the declared program changed at TP1 with the switch on');
});

// (c-i) a ~700-token prompt at max-num-batched-tokens 256: prefill chunks 256, 256, 188 (three chunks, ragged
// last). The two 256-row chunks split (>= 256 rows); on equals off in tokens and top-3 logprobs.
test('(c-i) TP2, 256-row prefill chunks (3 chunks, ragged last): on == off', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = ck.rnd(101, 700);
  assert.ok(P.length > 2 * 256 && P.length % 256 !== 0, 'the prompt must span >= 3 chunks with a ragged last (256-row budget)');
  const run = async env => {
    const srv = await ck.serveX(rad, 0, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 1, tokens: 256, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try { return {res: await ck.genFull(srv, P, NT), log: srv.log}; }
    finally { srv.stop(); }
  };
  const off = await run(OFF), on = await run(ON);
  sameTokens(off.res, on.res, '256-row chunks');
  assertSplitRan(off.log, 'off'); assertSplitRan(on.log, 'on');
  assertTrace(on.log, 'on (c-i)'); assertNoTrace(off.log, 'off (c-i)');
});

// (c-ii) an odd row count (257 rows, a single >= 256 chunk): rank 0 runs 129 rows, rank 1 the short 128. on == off.
test('(c-ii) TP2, odd T (last rank run short): on == off', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = ck.rnd(202, 257);                          // odd, >= 256 -> a single 257-row chunk that splits
  const run = async env => {
    const srv = await ck.serveX(rad, 0, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 1, tokens: 512, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try { return {res: await ck.genFull(srv, P, NT), log: srv.log}; }
    finally { srv.stop(); }
  };
  const off = await run(OFF), on = await run(ON);
  sameTokens(off.res, on.res, 'odd T');
  assertSplitRan(off.log, 'off'); assertSplitRan(on.log, 'on');
  // odd T: a trace step with odd T >= 257 whose two ranks' rows are unequal (rank 0 the long run, rank 1 the short)
  assertTrace(on.log, 'on (c-ii)', {odd: true}); assertNoTrace(off.log, 'off (c-ii)');
});

// (c-iii) two concurrent requests of different lengths (a multi-sequence step): each request on equals the same
// request off. RADIANCE_LOG_STEPS shows a step that carried two sequences AND reached the 256-row threshold, so
// the split ran over a multi-sequence step. (Whether the two prefills share a step depends on arrival timing; the
// assertion that they did, in both arms, is what makes the on-vs-off comparison meaningful.)
test('(c-iii) TP2, two concurrent requests of different lengths (multi-sequence step): on == off', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = [ck.rnd(303, 300), ck.rnd(404, 500)];      // different lengths, both > 256 rows
  const run = async env => {
    const srv = await ck.serveX(rad, 0, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 2, tokens: 512, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try {
      const res = await Promise.all(P.map(p => ck.genFull(srv, p, NT)));
      return {res, log: srv.log};
    } finally { srv.stop(); }
  };
  const off = await run(OFF), on = await run(ON);
  for (let i = 0; i < P.length; i++) sameTokens(off.res[i], on.res[i], `concurrent request ${i}`);
  for (const [tag, r] of [['off', off], ['on', on]]) {
    const st = steps(r.log);
    assert.ok(st.length > 0, `${tag}: no step lines captured (RADIANCE_LOG_STEPS)`);
    assert.ok(st.some(s => s.nseq >= 2 && s.ntok >= 256),
      `${tag}: no step carried two sequences at the 256-row threshold (n_seq/n_tok: ${st.map(s => s.nseq + '/' + s.ntok).join(',')})`);
  }
  assertTrace(on.log, 'on (c-iii)'); assertNoTrace(off.log, 'off (c-iii)');
});

// (c-iv) the DSpark drafter on (spec 7): on equals off in tokens and top-3 logprobs, and the drafter really
// drafted. The 256-row prefill chunks split; the per-step verify rows (T = 1 + spec = 8) are below the 256-row
// threshold and take the serial indexer path, so this case shows the split does not disturb a drafter run.
test('(c-iv) TP2, spec 7 (DSpark drafter on): on == off in tokens and top-3 logprobs', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = ck.rnd(505, 700);
  const run = async env => {
    const srv = await ck.serveX(rad, 7, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 1, tokens: 256, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try { return {res: await ck.genFull(srv, P, NT), log: srv.log}; }
    finally { srv.stop(); }
  };
  const off = await run(OFF), on = await run(ON);
  sameTokens(off.res, on.res, 'spec 7');
  assertSplitRan(off.log, 'off'); assertSplitRan(on.log, 'on');
  assertTrace(on.log, 'on (c-iv)'); assertNoTrace(off.log, 'off (c-iv)');
  assert.ok(off.res.drafted > 0, 'spec 7 off drafted nothing');
  assert.ok(on.res.drafted > 0, 'spec 7 on drafted nothing');
});

// (c-v) score tiling on (ernie55_idx_tile=8 -> ernie_dsa_topk_tiled): on equals off, and the serve log reports a
// tiled indexer score plane, so the tiled path was the one the split drove.
test('(c-v) TP2, score tiling on (ernie55_idx_tile): on == off on the ernie_dsa_topk_tiled path', {timeout: 2400000}, async () => {
  const rad = getTiledModel();
  const P = ck.rnd(606, 700);
  const run = async env => {
    const srv = await ck.serveX(rad, 0, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 1, tokens: 256, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try { return {res: await ck.genFull(srv, P, NT), log: srv.log}; }
    finally { srv.stop(); }
  };
  const off = await run(OFF), on = await run(ON);
  sameTokens(off.res, on.res, 'tiled');
  for (const [tag, r] of [['off', off], ['on', on]]) {
    assert.match(r.log, /indexer score plane[^\n]*tiled/i, `${tag}: the serve log does not report a tiled indexer score plane, so the tiled path was not taken: ` + r.log.slice(-2000));
    assertSplitRan(r.log, tag);
  }
  assertTrace(on.log, 'on (c-v)'); assertNoTrace(off.log, 'off (c-v)');
});

// (d) both switches at TP 2: the overlap is declined because the row split is on, and the output equals the plain
// (row-split-on) run -- the declined overlap changes nothing.
test('(d) TP2, row split + AR overlap: overlap declined (reason names the row split), output == plain', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = ck.rnd(707, 700);
  const run = async env => {
    const srv = await ck.serveX(rad, 0, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 1, tokens: 256, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try { return {res: await ck.genFull(srv, P, NT), log: srv.log}; }
    finally { srv.stop(); }
  };
  const plain = await run(ON);                          // row split on, no overlap requested
  const both = await run({...ON, [AR]: '1'});           // row split on, overlap requested (and declined)
  sameTokens(plain.res, both.res, 'row split + overlap vs plain');
  assert.match(both.log, /collectives overlap requested but off: the indexer row split is on/,
    'the overlap was not declined with the row-split reason: ' + both.log.slice(-2000));
  assert.doesNotMatch(plain.log, /collectives overlap requested but off/,
    'the plain run (no overlap requested) printed an overlap-declined line');
  // (d) has no OFF arm -- both runs have the switch on -- so both must carry the trace (T >= 256 teeth).
  assertTrace(plain.log, 'plain (d)'); assertTrace(both.log, 'both (d)');
});
