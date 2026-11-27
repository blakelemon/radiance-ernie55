'use strict';
// Multi-sequence isolation of the DSpark drafter (CPU reference kernels, a tiny synthetic ERNIE + DSpark checkpoint).
//
// With the default numerics (row identity: ERNIE_A16_MAX unset = 0) the CPU target is row independent: a request beside partners is bit-identical to the same request alone, so any dependence of a
// request on its partners at speculation 7 is the drafter, the scheduler or verify/rollback -- not target numerics. The checked quantities are each
// request's tokens, its top-3 logprobs AND its OWN drafted / accepted counters (timings.draft_n, timings.draft_n_accepted) against its alone run, across
// 2, 3, 4 and 8 sequences, staggered and ragged. The counters matter: a drafter that reads another sequence's rows still emits the same tokens
// (verification is lossless), so only the counters see it. Mutation proof (multiseq/fixes): with block_query's hidden_row taken from the first sequence's
// rows, tokens and logprobs stay equal and the counters differ, and this test fails.
//
// A positive control runs the same scenarios with the old small-step rule (ERNIE_A16_MAX=8): a request beside a partner is NOT bit-identical to itself
// alone there (row-count dependent numerics), so the comparison has teeth.
//
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55; the harness configures and builds it) and the sysfs mask:
//   nosys node --test tests/ernie55/dspark_multiseq.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const ck = require('./support/dspark_ckpt.cjs');
const {h} = ck;

const NT = Number(process.env.MS_NT || 48);
const SPEC = 7;
const PLEN = [12, 21, 2, 9, 33, 5, 17, 26];
const PR = PLEN.map((n, i) => ck.rnd(1000 + 37 * i, n));
const prompts = PR.map((p, i) => i === 0 ? [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8].map(x => x % ck.VOCAB)
  : i === 1 ? [17, 17, 17, 17, 2, 2, 2, 40, 41, 42, 17, 17, 60, 33, 21, 9, 9, 9, 50, 51, 52].map(x => x % ck.VOCAB) : p);

// scenario: list of {p: prompt index, n: max_tokens, d: start delay ms}
const SC = {
  '2-b80':     [{p: 0, n: NT, d: 0}, {p: 1, n: NT, d: 80}],
  '2-a-late':  [{p: 1, n: NT, d: 0}, {p: 0, n: NT, d: 150}],
  '2-b-short': [{p: 0, n: NT, d: 0}, {p: 1, n: 12, d: 0}],
  '3-mix':     [{p: 0, n: NT, d: 0}, {p: 1, n: 30, d: 0}, {p: 2, n: 20, d: 40}],
  '4-mix':     [{p: 0, n: NT, d: 0}, {p: 1, n: 40, d: 0}, {p: 2, n: 28, d: 60}, {p: 3, n: 16, d: 120}],
  '8-same':    PLEN.map((_, i) => ({p: i, n: NT, d: 0})),
  '8-ragged':  PLEN.map((_, i) => ({p: i, n: i === 0 ? NT : Math.max(4, NT - 6 * i), d: i % 3 * 50})),
};
// Same requests, other admission order / staggering: each must still equal its alone run.
SC['8-reordered'] = PLEN.map((_, i) => ({p: PLEN.length - 1 - i, n: PLEN.length - 1 - i === 0 ? NT : Math.max(4, NT - 6 * (PLEN.length - 1 - i)), d: (i * 37) % 5 * 30}));
SC['4-reordered'] = [{p: 3, n: 16, d: 0}, {p: 2, n: 28, d: 20}, {p: 1, n: 40, d: 90}, {p: 0, n: NT, d: 10}];
// Serialized control: one request at a time, so no step may carry two sequences.
const SERIAL = {'serial-3': [{p: 0, n: 24}, {p: 1, n: 24}, {p: 2, n: 24}]};
const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

let model;
function getModel() {
  if (!model) {
    const c = ck.buildCheckpoint('dspark-multiseq');
    const conv = ck.convertDspark(c.dir, SPEC, 'multiseq');
    assert.equal(conv.status, 0, 'convert must work: ' + (conv.stderr || '').slice(-600));
    model = conv.out;
  }
  return model;
}

// Runs every scenario in `names` at speculation 7 and compares each request with its own alone run (same server, same numerics mode).
async function run(a16, names) {
  const rad = getModel();
  const env = {RADIANCE_LOG_STEPS: 1, ...(a16 === undefined ? {} : {ERNIE_A16_MAX: a16})};
  const srv = await ck.serveX(rad, SPEC, env);
  assert.ok(!srv.failed, (srv.log || '').split('\n').slice(-30).join('\n'));
  const out = {scen: {}, diffs: [], steps: {}};
  // RADIANCE_LOG_STEPS: one engine log line per step, `step N <phase> n_seq=K n_tok=T ...`; the slice of the server log a scenario produced gives its n_seq per step
  const stepSeqs = from => [...srv.log.slice(from).matchAll(/step \d+ (?:prefill|decode|MIXED) n_seq=(\d+)/g)].map(m => Number(m[1]));
  try {
    const alone = {};
    for (const name of names.filter(n => SC[n])) for (const r of SC[name]) {
      const key = `${r.p}:${r.n}`;
      if (!alone[key]) alone[key] = await ck.genFull(srv, prompts[r.p], r.n);
    }
    for (const name of names.filter(n => SC[n])) {
      const mark = srv.log.length;
      const res = await Promise.all(SC[name].map(async r => { await new Promise(x => setTimeout(x, r.d)); return ck.genFull(srv, prompts[r.p], r.n); }));
      out.steps[name] = stepSeqs(mark);
      out.scen[name] = res.map((g, i) => {
        const z = alone[`${SC[name][i].p}:${SC[name][i].n}`];
        const row = {scenario: name, req: i, tokens: eq(g.ids, z.ids), logprobs: eq(g.top, z.top),
          counters: g.drafted === z.drafted && g.accepted === z.accepted, got: `${g.drafted}/${g.accepted}`, alone: `${z.drafted}/${z.accepted}`};
        if (!(row.tokens && row.logprobs && row.counters)) out.diffs.push(row);
        return row;
      });
    }
    for (const [name, reqs] of Object.entries(SERIAL)) if (names.includes(name)) {
      const mark = srv.log.length;
      const res = [];
      for (const r of reqs) res.push(await ck.genFull(srv, prompts[r.p], r.n));
      out.steps[name] = stepSeqs(mark);
      out.scen[name] = res.map((g, i) => ({scenario: name, req: i, got: `${g.drafted}/${g.accepted}`}));
    }
  } finally { srv.stop(); }
  return out;
}

test('with row-independent numerics every request equals its alone run in tokens, logprobs and its own drafted/accepted counters, 2 to 8 sequences', {timeout: 1800000}, async () => {
  const r = await run(undefined, [...Object.keys(SC), ...Object.keys(SERIAL)]);
  if (process.env.MS_DEBUG) for (const [k, v] of Object.entries(r.steps)) console.log('STEPS', k, v.join(''));
  const msg = r.diffs.map(d => `${d.scenario} req${d.req}: tokens ${d.tokens} logprobs ${d.logprobs} counters ${d.got} vs alone ${d.alone}`).join('\n');
  assert.equal(r.diffs.length, 0, 'a request depends on its partners:\n' + msg);
  // The scenarios really put sequences into shared steps: from the engine's own per-step n_seq (not from nonzero drafted counters, which a
  // serialized run also has). Every scenario with >= 2 requests had a step carrying >= 2 sequences; the serialized control never did.
  for (const name of Object.keys(SC)) {
    const st = r.steps[name], k = SC[name].length;
    assert.ok(st.length > 0, name + ': no step lines captured (RADIANCE_LOG_STEPS)');
    assert.ok(Math.max(...st) >= 2, `${name}: no step carried two sequences (n_seq per step: ${st.join(',')})`);
    assert.ok(Math.max(...st) <= k, `${name}: a step carried more sequences than requests (${Math.max(...st)} > ${k})`);
    // every scenario overlaps all its requests by design (delays <= 150 ms against runs of dozens of steps): some step carried all of them
    assert.equal(Math.max(...st), k, `${name}: all ${k} overlapping requests should have shared one step (n_seq: ${st.join(',')})`);
  }
  for (const name of Object.keys(SERIAL)) {
    const st = r.steps[name];
    assert.ok(st.length > 0, name + ': no step lines captured');
    assert.equal(Math.max(...st), 1, `${name}: serialized requests must never share a step (n_seq per step: ${st.join(',')})`);
  }
});

test('positive control: with the old small-step rule (ERNIE_A16_MAX=8) a request beside a partner differs from itself alone, so the comparison has teeth', {timeout: 1800000}, async () => {
  const r = await run('8', ['2-b80', '8-same']);
  assert.ok(r.diffs.length > 0, 'expected the default numerics to show the alone-vs-concurrent difference');
});
