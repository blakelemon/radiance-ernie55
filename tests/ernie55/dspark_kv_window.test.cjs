'use strict';
// The ERNIE DSpark drafter's KV group (kv_dspark) under RADIANCE_ERNIE55_DSPARK_KV_WINDOW, on a tiny synthetic checkpoint with the drafter
// (MXFP4 routed experts, block-fp8 projections), on the CPU reference kernels. The switch is a declare-time choice of the group's kind:
//   * off (unset or '0'): kv_dspark is a full-context group and the declared program is byte-identical to the default;
//   * on ('1'): kv_dspark is a RAD_KV_WINDOW of the drafter's attention window (ds.window), its layers/block/page unchanged, its block
//     count below the full group's, every other group and every op block unchanged (the program does not change, only the group);
//   * on with no drafter (num-speculative-tokens 0): there is no kv_dspark group and the switch is inert.
// The allocator's group table is printed only under --debug-placement, and configure_kv (which prints it) runs only on a real serve
// (--debug-graph returns before it), so the group lines are read from a server's startup log; the program text and the one-shot
// "kv_dspark is a window group of N tokens" log line come from --debug-graph.
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:
//   nosys node --test tests/ernie55/dspark_kv_window.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ck = require('./support/dspark_ckpt.cjs');
const {groups, blocks, prefixCounters} = require('./support/kv_ckpt.cjs');
const {h} = ck;

const WIN = 'RADIANCE_ERNIE55_DSPARK_KV_WINDOW';

let model;
function getModel() {
  if (!model) {
    const c = ck.buildCheckpoint('dspark-kv-window');
    const conv = ck.convertDspark(c.dir, 7, 'declare');
    assert.equal(conv.status, 0, 'convert must work: ' + (conv.stderr || '').slice(-600));
    model = conv.out;
  }
  return model;
}

// `radiance --config` with debug-graph: the declared program, printed instead of served (host kernels, no card). `env` is merged
// into the spawn environment. The result is memoised: the cases below ask for the same (spec, env) more than once.
const _dg = new Map();
let _dgSeq = 0;
function debugGraph(spec, env = {}) {
  const key = `${spec}|${JSON.stringify(env)}`;
  if (_dg.has(key)) return _dg.get(key);
  const b = h.ensureBuilt();
  const y = path.join(h.tmpRoot, `dg${spec}-${_dgSeq++}.yaml`);
  fs.writeFileSync(y, ['server:', '  host: 127.0.0.1', '  port: 0', '  log: info', `  radiance-home: ${b.home}`, '  kernels: libref',
    '  debug-accept-reference-kernels: true', 'cards: all', 'models:', '  ern:', `    model: ${getModel()}`, '    tp: 1', '    debug-graph: true',
    '    max-num-seqs: 2', '    max-num-batched-tokens: 64', '    max-model-len: 256', `    num-speculative-tokens: ${spec}`, ''].join('\n'));
  // The window switch is removed from the spawn env when the caller does not pass it, so "unset" is
  // really unset and an ambient RADIANCE_ERNIE55_DSPARK_KV_WINDOW cannot turn an OFF run on.
  const spawnEnv = {...process.env, ROCR_VISIBLE_DEVICES: '99', RADIANCE_HOST_DEVICES: '1', RADIANCE_HOST_VRAM_MIB: '512', ...env};
  if (!(WIN in env)) delete spawnEnv[WIN];
  const r = h.sh(b.radiance, ['--config', y], {env: spawnEnv});
  const out = {status: r.status, text: r.stdout + r.stderr};
  _dg.set(key, out);
  return out;
}

// The declared graph only, with the engine's interleaved log lines dropped. debugGraph returns
// stdout+stderr merged, and an engine log line (e.g. the "I kernel libraries on the search path ...
// not loaded" note) is emitted at a slightly different position between two identical runs, so the raw
// text of two runs differs by line order even when the declared program is byte-identical -- the flake
// this helper removes. The engine prefixes every log line with a single capital letter and a space
// (`I `, `W `, `E `, `D `); dropping those lines leaves the `==== declared graph:` section and only it,
// which is what "the program is unchanged" means. Assertions that read a log line (the window-group
// note, the pool line, a warning) stay on the raw text and are not passed through here.
function graphText(text) {
  return text.split('\n').filter(l => !/^[IWED] /.test(l)).join('\n');
}

// The geometry serveGroups serves every group report with. The pool is sized against the WORST CASE:
// SG_SEQS sequences each at the model's max-model-len (serveX's default len, ck.MAXLEN). The full
// groups' block counts are checked against SG_SEQS * ck.MAXLEN in the ON case below.
const SG_SEQS = 2, SG_TOKENS = 64;
// The allocator's group table (name -> {kind, layers, block, pageBytes, blocks}), read from a real server's startup log: the report
// is printed only under --debug-placement, which configure_kv emits after --debug-graph has already returned. Memoised per (spec, env).
const _sg = new Map();
async function serveGroups(spec, env = {}) {
  const key = `${spec}|${JSON.stringify(env)}`;
  if (_sg.has(key)) return _sg.get(key);
  const srv = await ck.serveX(getModel(), spec, env, {seqs: SG_SEQS, tokens: SG_TOKENS, extraYaml: ['    debug-placement: true']});
  assert.ok(!srv.failed, `server did not start: ` + (srv.log || '').split('\n').slice(-12).join('\n'));
  try {
    const g = groups(srv.log);
    // Carry the startup log alongside the groups, non-enumerable so the ON case's name loop does not
    // mistake it for a group: the worst-case pool line is asserted from it.
    Object.defineProperty(g, 'log', {value: srv.log, enumerable: false});
    _sg.set(key, g);
    return g;
  } finally { srv.stop(); }
}

test('off: kv_dspark is a full group and the program is unchanged', async () => {
  const a = debugGraph(7, {});
  const b = debugGraph(7, {[WIN]: '0'});
  assert.equal(a.status, 0, a.text.slice(-800));
  assert.equal(b.status, 0, b.text.slice(-800));
  assert.equal(graphText(a.text), graphText(b.text), 'the declared program changed with the switch off');
  const g = await serveGroups(7, {});
  assert.ok(g.kv_dspark, 'no kv_dspark group in the allocator report');
  assert.equal(g.kv_dspark.kind, 'full');
});

test('on: kv_dspark is a window group of the drafter window, nothing else changes', async () => {
  const off = debugGraph(7, {});
  const on = debugGraph(7, {[WIN]: '1'});
  assert.equal(on.status, 0, on.text.slice(-800));
  const goff = await serveGroups(7, {});
  const gon = await serveGroups(7, {[WIN]: '1'});
  assert.ok(goff.kv_dspark && gon.kv_dspark, 'no kv_dspark group in the allocator report');
  assert.equal(gon.kv_dspark.kind, 'window');
  for (const k of ['layers', 'block', 'pageBytes']) assert.equal(gon.kv_dspark[k], goff.kv_dspark[k], `kv_dspark ${k} changed`);
  assert.ok(gon.kv_dspark.blocks < goff.kv_dspark.blocks, `window blocks ${gon.kv_dspark.blocks} not below full ${goff.kv_dspark.blocks}`);
  // Every other group keeps its kind and geometry. A WINDOW group is capped the same way and keeps its
  // block count exactly. A FULL group's block count is NOT compared off vs on: on this tiny host the
  // pool is sized to the WORST CASE (SG_SEQS sequences all at max-model-len) and rounded up to a whole
  // MiB, not filled with every free byte, so freeing the drafter's bytes SHRINKS the pool (4 MiB off ->
  // 2 MiB on) and the full groups keep slack above the worst case in both runs -- i.e. blocks_on >=
  // blocks_off holds on a card (bytes-bound) but NOT here (demand-bound). The contract that holds in
  // BOTH regimes is that the pool still covers the worst case: every full group's blocks*block >=
  // SG_SEQS*max-model-len, asserted here, with the allocator's own "backs 100% of the worst case" line
  // checked in both startup logs below the loop.
  const worst = SG_SEQS * ck.MAXLEN;   // SG_SEQS sequences each at the model's max-model-len (serveX default len)
  for (const name of Object.keys(goff)) {
    if (name === 'kv_dspark') continue;
    assert.ok(gon[name], `group ${name} disappeared`);
    for (const k of ['kind', 'layers', 'block', 'pageBytes']) assert.equal(gon[name][k], goff[name][k], `group ${name} ${k} changed`);
    if (goff[name].kind === 'full') {
      for (const [tag, grp] of [['off', goff], ['on', gon]])
        assert.ok(grp[name].blocks * grp[name].block >= worst,
          `FULL group ${name} ${tag}: ${grp[name].blocks} blocks * ${grp[name].block} tok = ${grp[name].blocks * grp[name].block} < the worst case ${worst} (${SG_SEQS} seqs * max-model-len ${ck.MAXLEN})`);
    } else
      assert.equal(gon[name].blocks, goff[name].blocks, `WINDOW group ${name} blocks changed (${goff[name].blocks} off -> ${gon[name].blocks} on)`);
  }
  // The pool really covers the worst case in both runs, so the slack above `worst` the full groups keep
  // is real and not a pool that no longer backs the worst case.
  for (const [tag, grp] of [['off', goff], ['on', gon]])
    assert.ok(grp.log.includes('kv: the pool backs 100% of the worst case'),
      `${tag}: the pool does not back 100% of the worst case, so the full groups' slack above ${worst} tokens is unjustified`);
  const boff = blocks(graphText(off.text)), bon = blocks(graphText(on.text));
  assert.equal(bon.length, boff.length, 'the op-block count changed');
  for (let i = 0; i < boff.length; i++) assert.equal(bon[i][0], boff[i][0], `op block ${i} header changed`);
  const decl = /drafter window (\d+)/.exec(on.text);
  assert.ok(decl, 'no drafter window line in the program');
  const m = /kv_dspark is a window group of (\d+) tokens/.exec(on.text);
  assert.ok(m, 'no "kv_dspark is a window group" log line');
  assert.equal(m[1], decl[1], `window-group log ${m[1]} != drafter window ${decl[1]}`);
  // The declared window VALUE, checked against a number computed independently of both log lines: the
  // fixture writes HGEO.window (8) into config.json as sliding_window, which the engine reads back as the
  // drafter's ds.window (arch/ernie55/ernie55.cpp:1194, 2026). A wrong declared window fails here even
  // though the block geometry (ceil(window/block)) is the same for any window in (block, 2*block].
  assert.equal(Number(m[1]), ck.HGEO.window, `declared window ${m[1]} != the fixture drafter window HGEO.window=${ck.HGEO.window} (config sliding_window)`);
  // The allocator report prints a group's total blocks, not a per-sequence figure, so the implied
  // per-sequence window-block count (ceil(window/block)+1, kv.cpp:182) is not separately asserted here.
});

test('on with spec 0: no drafter, no group, the switch is inert', async () => {
  const g = debugGraph(0, {[WIN]: '1'});
  assert.equal(g.status, 0, g.text.slice(-800));
  assert.doesNotMatch(g.text, /kv_dspark is a window group/);
  const grp = await serveGroups(0, {[WIN]: '1'});
  assert.ok(!grp.kv_dspark, 'a kv_dspark group exists with no drafter');
});

/* ------------------------------------------------------------------ P4: the host end-to-end gate
 *
 * With the switch on, every request must give the same tokens, the same top-3 logprobs and the
 * same drafted/accepted counters as with it off. The group declaration is the only change, and a
 * WINDOW group's table is rebased to its first held block, so the op reads the same rows in the
 * same order. The counters are the strong check: verification is lossless, so a
 * drafter that read the wrong rows would still emit the same tokens and only the counters would
 * see it.
 *
 * Every case starts an OFF server and an ON server with identical yaml and env but for the switch
 * (and the debug-placement report the group lines are read from), and compares the two. The
 * prompts are long enough for the drafter's window to recycle many times: a prompt of P rows plus
 * NT decode rows is P + NT rows, and the fixture's window W is 8, so the shortest prompt's
 * 40 + 64 rows recycle it at least three times (asserted in case (a)).
 */
const NT = 64;
// The four prompts of dspark_spec_parity.test.cjs, and the packet's four (seed, length) pairs.
const PROMPTS4 = [[1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8], [17, 17, 17, 17, 2, 2, 2, 40, 41, 42, 17, 17, 60, 33, 21, 9, 9, 9, 50, 51, 52], ck.rnd(1234, 33), ck.rnd(77, 5)]
  .map(p => p.map(x => x % ck.VOCAB));
const P4 = [...[[11, 40], [12, 97], [13, 230], [14, 17]].map(([s, n]) => ck.rnd(s, n)), ...PROMPTS4];
const OFF = {ERNIE_A16_MAX: '0', [WIN]: '0'};                 // the row-independence pin dspark_spec_parity uses; the switch is set to '0' explicitly, never inherited
const ON = {ERNIE_A16_MAX: '0', [WIN]: '1'};
const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

// One request's every checked quantity: the tokens, the top-3 logprobs and the drafter's own two
// counters. The logprobs are strings (`id@logprob`), so the comparison is bit for bit.
function sameRequest(a, b, what) {
  assert.ok(eq(a.ids, b.ids), `${what}: tokens differ\n  off ${a.ids.join(',')}\n  on  ${b.ids.join(',')}`);
  assert.ok(eq(a.top, b.top), `${what}: top-3 logprobs differ`);
  assert.equal(a.drafted, b.drafted, `${what}: drafted differs (${a.drafted} off, ${b.drafted} on)`);
  assert.equal(a.accepted, b.accepted, `${what}: accepted differs (${a.accepted} off, ${b.accepted} on)`);
}

// The tokens and top-3 logprobs only, skipping the two drafter counters. Used where the number of
// drafts a sequence verifies is chosen from MEASURED step times (core/sched/draft.h: step_cost /
// step_time, "the cost of the steps before"), so drafted/accepted are wall-clock dependent and not
// reproducible even between two OFF runs. Verification is lossless, so the tokens and top-3 stay
// exact; only the counters are dropped.
function sameRequestTokens(a, b, what) {
  assert.ok(eq(a.ids, b.ids), `${what}: tokens differ\n  off ${a.ids.join(',')}\n  on  ${b.ids.join(',')}`);
  assert.ok(eq(a.top, b.top), `${what}: top-3 logprobs differ`);
}

// One named counter from /metrics. The exposition is `name{labels} value` (core/server/metrics.cpp),
// and a counter the process has not published yet reads as 0. kv_ckpt's prefixCounters filters its
// lines on "prefix", which drops radiance:attn_cached_tokens_total, so the named counters are read
// here and prefixCounters is used for the vllm:prefix_cache_* lines beside them.
async function metric(srv, name) {
  const r = await fetch(`http://127.0.0.1:${srv.port}/metrics`);
  assert.equal(r.status, 200, '/metrics status ' + r.status);
  const m = new RegExp('^' + name + '(?:\\{[^}]*\\})?\\s+(\\S+)', 'm').exec(await r.text());
  return m ? Number(m[1]) : 0;
}

// (a) one request at a time, at TP1 and TP2, at depths 0, 2 and 7: on equals off.
for (const tp of [1, 2]) {
  test(`(a) TP${tp}: spec 0/2/7, one request at a time -- on equals off in tokens, top logprobs and draft counters`, {timeout: 2400000}, async () => {
    const rad = getModel();
    const w = /drafter window (\d+)/.exec(debugGraph(7, {[WIN]: '1'}).text);
    assert.ok(w, 'no drafter window line in the program');
    assert.ok(40 + NT > Number(w[1]) + 3 * 8,
      `the shortest prompt (40) plus ${NT} decode rows must recycle the drafter window ${w[1]} at least three times`);
    for (const spec of [0, 2, 7]) {
      const run = {};
      for (const on of [false, true]) {
        const srv = await ck.serveX(rad, spec, on ? ON : OFF, {seqs: 1, tp, extraYaml: ['    debug-placement: true']});
        assert.ok(!srv.failed, `TP${tp} spec ${spec} ${on ? 'on' : 'off'}: server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
        try {
          const res = [];
          for (const p of P4) res.push(await ck.genFull(srv, p, NT));   // alone in the engine
          run[on ? 'on' : 'off'] = {res, g: groups(srv.log)};
        } finally { srv.stop(); }
      }
      const off = run.off, on = run.on;
      off.res.forEach((g, i) => sameRequest(g, on.res[i], `TP${tp} spec ${spec} prompt ${i}`));
      if (spec === 0) {
        assert.ok(!off.g.kv_dspark && !on.g.kv_dspark, `spec 0 has a kv_dspark group`);
      } else {
        assert.ok(off.g.kv_dspark && on.g.kv_dspark, 'no kv_dspark group in the allocator report');
        assert.equal(off.g.kv_dspark.kind, 'full', 'the OFF group is not full');
        assert.equal(on.g.kv_dspark.kind, 'window', 'the ON group is not a window group');
        assert.ok(on.g.kv_dspark.blocks < off.g.kv_dspark.blocks,
          `window blocks ${on.g.kv_dspark.blocks} not below full ${off.g.kv_dspark.blocks}`);
      }
      if (spec === 7) {
        assert.ok(off.res.reduce((a, g) => a + g.drafted, 0) > 0, 'spec 7 off drafted nothing');
        assert.ok(on.res.reduce((a, g) => a + g.drafted, 0) > 0, 'spec 7 on drafted nothing');
      }
      console.log(`TP${tp} spec ${spec}: kv_dspark blocks off ${off.g.kv_dspark ? off.g.kv_dspark.blocks : '-'} on ${on.g.kv_dspark ? on.g.kv_dspark.blocks : '-'}; drafted off ${off.res.reduce((a, g) => a + g.drafted, 0)} on ${on.res.reduce((a, g) => a + g.drafted, 0)}`);
    }
  });
}

// (b) four requests at once (the concurrency pattern of dspark_multiseq.test.cjs): each request on
// equals the same request off, counters included. RADIANCE_LOG_STEPS shows the requests really did
// share a step, so the case is about concurrency and not four serial runs.
test('(b) TP1, spec 7: 4 concurrent requests (seqs 4) -- each request on equals the same request off', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = P4.slice(0, 4);
  const run = async env => {
    const srv = await ck.serveX(rad, 7, {...env, RADIANCE_LOG_STEPS: '1'}, {seqs: 4});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try {
      const res = await Promise.all(P.map(p => ck.genFull(srv, p, NT)));
      const st = [...srv.log.matchAll(/step \d+ (?:prefill|decode|MIXED) n_seq=(\d+)/g)].map(m => Number(m[1]));
      return {res, st};
    } finally { srv.stop(); }
  };
  const off = await run(OFF), on = await run(ON);
  for (let i = 0; i < P.length; i++) sameRequest(off.res[i], on.res[i], `concurrent request ${i}`);
  for (const [tag, r] of [['off', off], ['on', on]]) {
    assert.ok(r.st.length > 0, `${tag}: no step lines captured (RADIANCE_LOG_STEPS)`);
    assert.ok(Math.max(...r.st) >= 2, `${tag}: no step carried two sequences (n_seq per step: ${r.st.join(',')})`);
    console.log(`(b) ${tag}: max n_seq ${Math.max(...r.st)} over ${r.st.length} steps`);
  }
});

// (c) a repeated prompt through the prefix cache, with checkpoints: the second request hits, and a
// short next turn (the prompt plus a few new tokens) resumes at the prompt-end snapshot and so reads
// the restored drafter window on its first draft. On equals off for every request. Two geometries are
// run, each OFF vs ON, differing in where a checkpoint lands relative to a prefill chunk.
//
// Geometry (read back from the engine's own logs, not assumed): the fixture's quantum/block is 8 and
// the drafter window W is 8. A window tail snapshot is taken a block short of the prompt's end
// (scheduler.cpp tail_checkpoint: (target - max(Q,16)) / Q * Q, floored at min(interval, 512)).
//   A `prefix-resume: nearest`: interval 48 (> the 40-token budget, a multiple of the quantum 8 but not
//     of the budget 40), prompt 96 tokens (= 2*48). The tail snapshot at 80 then falls STRICTLY INSIDE
//     the budget chunk [48,88), while the prompt-end interval snapshot at 96 is the deepest checkpoint
//     (the tip). This needs `nearest`: the DEFAULT (`exact`) rounds an interval past the budget up to a
//     multiple of the budget's chunk so every save is where a cold prefill's chunk ends
//     (geometry.cpp:181-189, "--checkpoint-interval 48 resolves to 80"), and only `nearest` keeps 48
//     and lets a save fall strictly inside a chunk.
//   B the DEFAULT (`exact`, no yaml key): interval 40 (= the budget, a multiple of it), prompt 120
//     tokens. Every save lands ON a chunk boundary by design (the cold prefill's grid), so none is
//     strictly inside a chunk.
// In both an APPENDING turn's hit is capped only at prompt.size()-1 (scheduler.cpp:1628), so it resumes
// at the tip (96 / 120) and leaves NEW=4 rows (< W-1=7): the first draft's window [total-W+1, total)
// reaches below the tip into restored rows. (An EXACT repeat is capped a block lower, at P-quantum, so
// it is not the request that shows the restored read -- F2's 112.)
//
// No log line or counter fires on a restore with its window rows and position: the restore path
// (engine.cpp:1580-1614) logs only on error. The restored read is shown by the strongest evidence that
// exists -- RADIANCE_DEBUG_CKPT_HASH logs each snapshot's position, whether it was taken inside the
// step's chunk, and a digest of the saved bytes (engine.cpp:2142); radiance:attn_cached_tokens_total
// gives the hit (resume) position; and the short-remainder arithmetic below ties the two to the window
// read. The seq-2 digest at the prompt end equals the seq-1 one: the window restored at the earlier hit
// and extended by the recompute reproduces the cold run's window bytes.
const C_NEW = 4;                 // the short next turn: fewer than W-1 new rows after the hit
const C_NT = 32;                 // decode rows per request (the exact repeat and the short turn)
const C_CASES = [
  {mode: 'nearest', tok: 40, iv: 48, plen: 96, inside: true, extraYaml: ['    prefix-resume: nearest']},
  {mode: 'exact', tok: 40, iv: 40, plen: 120, inside: false, extraYaml: []},
];
for (const cfg of C_CASES) {
  test(`(c) TP1, spec 7 [${cfg.mode}]: prefix cache with checkpoints -- a repeated prompt hits, a short next turn reads the restored drafter window, on equals off`, {timeout: 2400000}, async () => {
    const rad = getModel();
    const TOK = cfg.tok;                // --max-num-batched-tokens: the budget a prefill is chunked to
    const IV = cfg.iv;                  // --checkpoint-interval
    const P = ck.rnd(21, cfg.plen);     // the prompt: a multiple of the interval, so the prompt-end snapshot is the tip
    const w = /drafter window (\d+)/.exec(debugGraph(7, {[WIN]: '1'}).text);
    assert.ok(w, 'no drafter window line in the program');
    const W = Number(w[1]);
    if (cfg.inside)
      assert.ok(IV > TOK && IV % TOK !== 0, 'the interval must exceed the budget and not align with it, so a save can fall strictly inside a budget chunk');
    else
      assert.equal(IV % TOK, 0, 'the exact-mode interval must be a multiple of the budget, so every save lands on a chunk boundary');
    assert.equal(P.length % IV, 0, 'the prompt must be a multiple of the interval, so the prompt-end snapshot is the tip');
    assert.ok(C_NEW < W - 1, `the short turn (${C_NEW} new rows) must leave fewer than W-1=${W - 1} rows after the hit, or the first draft reads only new rows`);
    const run = async env => {
      const srv = await ck.serveX(rad, 7, {...env, RADIANCE_DEBUG_CKPT_HASH: '1'},
        {seqs: 1, tokens: TOK, prefix: true, ckptInterval: IV, extraYaml: cfg.extraYaml});
      assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
      try {
        const pc0 = await prefixCounters(srv);
        assert.ok(pc0.length > 0, 'the server publishes no prefix-cache counters');
        const m0 = await metric(srv, 'radiance:attn_cached_tokens_total');
        const r1 = await ck.genFull(srv, P, C_NT);                          // cold: fills the cache, writes the snapshots
        const m1 = await metric(srv, 'radiance:attn_cached_tokens_total');
        const r2 = await ck.genFull(srv, P, C_NT);                          // the exact repeat: a hit (capped a block short)
        const m2 = await metric(srv, 'radiance:attn_cached_tokens_total');
        const P3 = [...P, ...ck.rnd(22, C_NEW)];                            // the short next turn: P plus a few new tokens
        const r3 = await ck.genFull(srv, P3, C_NT);                         // resumes at the tip, reads the restored window
        const m3 = await metric(srv, 'radiance:attn_cached_tokens_total');
        return {res: [r1, r2, r3], m: [m0, m1, m2, m3], log: srv.log, P3};
      } finally { srv.stop(); }
    };
    const off = await run(OFF), on = await run(ON);
    for (let i = 0; i < 3; i++) sameRequest(off.res[i], on.res[i], `${cfg.mode} prefix request ${i + 1}`);
    for (const [tag, r] of [['off', off], ['on', on]]) {
      // The chunk geometry the engine resolved: the budget and the interval are the ones the case asked
      // for, and the resolved chunk is the budget (so a save can land strictly inside a chunk in mode A,
      // and on a boundary in mode B, where the interval equals the budget).
      const geom = /chunk geometry: quantum (\d+) \(attn block (\d+), state chunk (\d+)\), checkpoint interval (\d+), max_tok (\d+) -> prefill chunks of at most (\d+)/.exec(r.log);
      assert.ok(geom, `${tag}: no chunk geometry line in the log`);
      const Q = Number(geom[1]), iv = Number(geom[4]), maxTok = Number(geom[5]), eff = Number(geom[6]);
      assert.equal(maxTok, TOK, `${tag}: the token budget is not the one the case asked for`);
      assert.equal(iv, IV, `${tag}: the checkpoint interval is not the one the case asked for`);
      assert.equal(eff, maxTok, `${tag}: the resolved chunk (${eff}) must be the budget (${maxTok})`);
      // The hit positions, from the attn_cached deltas: the exact repeat resumes a quantum short of the
      // prompt end (P - Q), the short turn resumes at the prompt end (P).
      assert.equal(r.m[2] - r.m[1], P.length - Q,
        `${tag}: the exact repeat hit ${r.m[2] - r.m[1]} != P - quantum (${P.length} - ${Q}) (attn_cached ${r.m.join(' -> ')})`);
      assert.equal(r.m[3] - r.m[2], P.length,
        `${tag}: the short turn hit ${r.m[3] - r.m[2]} != the prompt end ${P.length} (attn_cached ${r.m.join(' -> ')})`);
      // A hit on this model resumes at a window snapshot, so the hit above is a restore and not only
      // adopted blocks (scheduler.cpp:142).
      assert.match(r.log, /a hit resumes from a window snapshot \(a checkpoint slot\)/,
        `${tag}: no windowed trunk group needs a snapshot, so no hit restores one`);
      // The ckpt-hash facts. RADIANCE_DEBUG_CKPT_HASH logs `seq`, `pos` (the absolute snapshot
      // position), `inside` (taken inside the step's chunk, not at its end) and a 16-hex digest of the
      // saved bytes (engine.cpp:2142).
      const hashes = [...r.log.matchAll(/ckpt-hash seq=(\d+) pos=(\d+) rank=\d+ inside=(\d+) bytes=\d+ ([0-9a-f]{16})/g)]
        .map(m => ({seq: Number(m[1]), pos: Number(m[2]), inside: Number(m[3]), hash: m[4]}));
      assert.ok(hashes.length > 0, `${tag}: RADIANCE_DEBUG_CKPT_HASH logged no snapshot`);
      if (cfg.inside) {
        // (ii) a save strictly inside a chunk: the window tail snapshot of a P-token prompt is at
        // (P - max(Q,16)) / Q * Q (tail_checkpoint); for this geometry that is 80, inside [48,88).
        const wtail = Math.floor((P.length - Math.max(Q, 16)) / Q) * Q;
        assert.ok(hashes.some(x => x.inside === 1 && x.pos === wtail),
          `${tag}: no snapshot taken strictly inside a chunk at the tail position ${wtail} (saw ${JSON.stringify(hashes)})`);
      } else {
        // Exact mode saves only on chunk boundaries by design (geometry.cpp:190-201): no snapshot is
        // ever taken strictly inside a chunk.
        assert.ok(hashes.every(x => x.inside === 0),
          `${tag}: exact mode took a snapshot inside a chunk (saw ${JSON.stringify(hashes.filter(x => x.inside === 1))})`);
      }
      // The prompt-end position is saved again by the later sequence (seq 2, the exact repeat, recomputes
      // from its hit to the prompt end): the restored window, extended by the recompute, must reproduce
      // the cold run's window bytes, so the two digests are equal.
      const pend = P.length;
      const h1 = hashes.find(x => x.seq === 1 && x.pos === pend);
      const h2 = hashes.find(x => x.seq === 2 && x.pos === pend);
      assert.ok(h1 && h2, `${tag}: no prompt-end checkpoint at ${pend} for both sequences (saw ${JSON.stringify(hashes)})`);
      assert.equal(h1.hash, h2.hash,
        `${tag}: the seq-2 checkpoint at the prompt end (${pend}) ${h2.hash} != seq-1's ${h1.hash} -- the restored window did not reproduce the cold run's bytes`);
      // (i)+(iii) the short next turn reads the restored drafter window: the hit position (the attn_cached
      // delta) is the prompt-end tip, and the turn leaves fewer than W-1 new rows, so the first draft's
      // window reaches below the hit into restored rows.
      const hit = r.m[3] - r.m[2];                  // tokens the short turn served from cache = resume position
      const remainder = r.P3.length - hit;          // new rows the context pass wrote after the restore
      assert.ok(remainder < W - 1, `${tag}: the short turn left ${remainder} new rows after the hit at ${hit}, not fewer than W-1=${W - 1}, so the first draft would not read restored window rows`);
      console.log(`(c) ${cfg.mode} ${tag}: attn_cached ${r.m.join(' -> ')}; quantum ${Q}, interval ${iv}, budget ${maxTok}, chunk ${eff}; short turn resumed at ${hit}, ${remainder} new rows (< W-1=${W - 1}) -> the first draft reads restored window rows`);
    }
    // The drafter's floor: the window group is not prefix-cached but rides in the window snapshot a
    // hit resumes from, so the ON line has one fewer prefix-cached group and one more snapshot group
    // than the OFF line, and neither leaves a hole.
    const floor = /draft floor: (\d+) paged group\(s\), (\d+) prefix-cached, (\d+) in the window snapshot -> a hit (leaves no hole|leaves a hole in a drafter's own KV)/;
    const fo = floor.exec(off.log), fn = floor.exec(on.log);
    assert.ok(fo, 'no draft floor line in the OFF log');
    assert.ok(fn, 'no draft floor line in the ON log');
    assert.equal(fn[1], fo[1], 'the paged group count changed');
    const exactOn = 'draft floor: 6 paged group(s), 3 prefix-cached, 3 in the window snapshot -> a hit leaves no hole';
    const exactOff = 'draft floor: 6 paged group(s), 4 prefix-cached, 2 in the window snapshot -> a hit leaves no hole';
    if (on.log.includes(exactOn) && off.log.includes(exactOff)) {
      console.log('(c) draft floor: the packet\'s counts (off 4 prefix-cached / 2 snapshot, on 3 / 3)');
    } else {
      console.log(`(c) draft floor off: ${fo[0]}\n(c) draft floor on:  ${fn[0]}`);
      assert.ok(off.log.includes('leaves no hole'), 'a hit leaves a hole in the drafter KV with the switch off');
      assert.ok(on.log.includes('leaves no hole'), 'a hit leaves a hole in the drafter KV with the switch on');
      assert.equal(Number(fn[2]), Number(fo[2]) - 1, 'the window group must be one fewer prefix-cached group');
      assert.equal(Number(fn[3]), Number(fo[3]) + 1, 'and one more group in the window snapshot');
      assert.equal(Number(fn[2]) + Number(fn[3]), Number(fn[1]), 'the ON groups are not all accounted for');
      assert.equal(Number(fo[2]) + Number(fo[3]), Number(fo[1]), 'the OFF groups are not all accounted for');
    }
  });
}

// (d) the pieced prefill (RADIANCE_ERNIE55_AR_OVERLAP=1) at TP2, with the window switch ON: pieced
// off vs on is bit-identical, and so is window on vs off with the pieced path on both. The env and
// the token budget are ar_overlap.test.cjs's (its serveTp default of 32 against a 28-row prompt, cut
// into two pieces of 14 with AR_OVERLAP_MIN 16); the prompt here is 230 rows, so it is many
// chunks and every one of them is pieced. Both pieced arms run with the window ON, so the case
// exercises the window group on the pieced path (Review F1 / Review major: the old case never set it).
test('(d) TP2, spec 7: pieced prefill (RADIANCE_ERNIE55_AR_OVERLAP=1), window on -- pieced off vs on and window on vs off', {timeout: 2400000}, async () => {
  const rad = getModel();
  const P = ck.rnd(13, 230);
  const TOK = 32;
  const AR = {RADIANCE_ERNIE55_AR_OVERLAP: '1', RADIANCE_ERNIE55_AR_OVERLAP_MIN: '16'};
  const run = async env => {
    const srv = await ck.serveX(rad, 7, env, {seqs: 1, tokens: TOK, tp: 2});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try {
      const res = await ck.genFull(srv, P, NT);
      return {res, log: srv.log};
    } finally { srv.stop(); }
  };
  const base = await run(ON);              // window on, no overlap: the control (must not piece)
  const onAr = await run({...ON, ...AR});  // window on, pieced
  const offAr = await run({...OFF, ...AR}); // window off, pieced
  sameRequest(base.res, onAr.res, 'pieced prefill (window on): pieced off vs on');
  sameRequest(onAr.res, offAr.res, 'pieced prefill: window on vs off');
  assert.doesNotMatch(base.log, /runs as two pieces/, 'the no-overlap control pieced a prefill');
  for (const [tag, r] of [['on+AR', onAr], ['off+AR', offAr]]) {
    assert.match(r.log, /collectives overlap on/, `${tag}: overlap not on`);
    assert.match(r.log, /prefill step runs as two pieces/, `${tag}: the pieced path must have run: ` + r.log.slice(-3000));
    assert.doesNotMatch(r.log, /overlap off for this program/, `${tag}: overlap off for this program`);
  }
});

// (e) --draft-window adaptive: on equals off. The adaptive policy changes how many drafts a
// sequence verifies, not the drafter's attention window (which stays the group's), so the rows the
// op reads are the same ones and the tokens and top-3 logprobs are bit-identical (sameRequestTokens).
// The drafted counter is wall-clock dependent (core/sched/draft.h: step_cost / step_time, "the cost
// of the steps before"), so it is not compared ON vs OFF directly. Instead an OFF-vs-OFF control
// establishes the premise: two OFF runs must draft DIFFERENT totals, or dropping the counter
// comparison is unjustified and the test FAILS (Review F3 / Review major). The accepted counter is
// compared ON vs OFF only when the OFF-vs-OFF control shows it equal too; otherwise both are printed.
// drafted > 0 and the log assertion hold in every run.
test('(e) TP1, spec 7: --draft-window adaptive -- on equals off, with an off-vs-off control', {timeout: 2400000}, async () => {
  const rad = getModel();
  const ADAPT = ['    draft-window: adaptive'];
  const run = async env => {
    const srv = await ck.serveX(rad, 7, env, {seqs: 1, extraYaml: ADAPT});
    assert.ok(!srv.failed, `server did not start:\n` + (srv.log || '').split('\n').slice(-20).join('\n'));
    try {
      const res = [];
      for (const p of P4) res.push(await ck.genFull(srv, p, NT));
      return {res, log: srv.log};
    } finally { srv.stop(); }
  };
  const sum = (r, k) => r.res.reduce((a, g) => a + g[k], 0);
  const off1 = await run(OFF), off2 = await run(OFF), on = await run(ON);   // off twice: the control
  off1.res.forEach((g, i) => sameRequestTokens(g, on.res[i], `adaptive prompt ${i}`));
  for (const [tag, r] of [['off1', off1], ['off2', off2], ['on', on]]) {
    assert.match(r.log, /--draft-window adaptive with a block drafter/,
      `${tag}: the adaptive window is not active, so the case proves nothing`);
    assert.ok(sum(r, 'drafted') > 0, `adaptive ${tag} drafted nothing`);
    console.log(`(e) ${tag}: drafted ${sum(r, 'drafted')} accepted ${sum(r, 'accepted')}`);
  }
  // The premise: the drafted total is wall-clock dependent, so two OFF runs must differ. If they do
  // not, dropping the ON-vs-OFF drafted comparison is unjustified -- fail loudly (do not skip).
  const d1 = sum(off1, 'drafted'), d2 = sum(off2, 'drafted');
  assert.notEqual(d1, d2, `(e) premise false: two OFF runs drafted the same total (${d1}); the drafted counter is not wall-clock dependent here, so the ON-vs-OFF counter comparison must not be dropped`);
  // Accepted: compared ON vs OFF per request only if the OFF-vs-OFF control shows it equal too;
  // otherwise the control does not justify an equality assertion, so both totals are printed.
  const a1 = sum(off1, 'accepted'), a2 = sum(off2, 'accepted'), an = sum(on, 'accepted');
  if (a1 === a2) {
    off1.res.forEach((g, i) => assert.equal(g.accepted, on.res[i].accepted, `adaptive prompt ${i}: accepted differs (${g.accepted} off, ${on.res[i].accepted} on)`));
    assert.equal(an, a1, `(e) ON accepted total ${an} != the equal OFF control ${a1}`);
    console.log(`(e) accepted: off-vs-off equal (${a1}), and on equals it (${an})`);
  } else {
    console.log(`(e) accepted: off-vs-off differ (${a1} vs ${a2}), so ON (${an}) is not asserted equal -- printed only`);
  }
});
