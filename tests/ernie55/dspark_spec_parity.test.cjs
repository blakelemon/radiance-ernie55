'use strict';
// Speculation depth parity of the DSpark drafter on the integration branch (CPU reference kernels, a tiny synthetic ERNIE + DSpark checkpoint, the compressed group in the
// block layout the build serves by default): the same requests at --num-speculative-tokens 0, 2, 3 and 7 must give the same tokens, and, with the row-independence pin
// ERNIE_A16_MAX=0 (the CPU target is then row independent), the same top-3 logprobs bit for bit, at TP1 and TP2. Verification is lossless, so the tokens must match
// without the pin on most prompts (reported); only the counters (drafted / accepted) may differ, and spec 0 must draft nothing.
//   nosys node --test tests/ernie55/dspark_spec_parity.test.cjs   (host build in RAD_BUILD_DIR or <repo>/build-ernie55)
const test = require('node:test');
const assert = require('node:assert/strict');
const ck = require('./support/dspark_ckpt.cjs');

const SPECS = (process.env.SP_SPECS || '0,2,3,7').split(',').map(Number);
const NT = Number(process.env.SP_NT || 64);
const PROMPTS = [[1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8], [17, 17, 17, 17, 2, 2, 2, 40, 41, 42, 17, 17, 60, 33, 21, 9, 9, 9, 50, 51, 52], ck.rnd(1234, 33), ck.rnd(77, 5)]
  .map(p => p.map(x => x % ck.VOCAB));
const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

let model;
function getModel() {
  if (!model) {
    const c = ck.buildCheckpoint('dspark-spec-parity');
    const conv = ck.convertDspark(c.dir, Math.max(...SPECS), 'spec-parity');
    assert.equal(conv.status, 0, 'convert must work: ' + (conv.stderr || '').slice(-600));
    model = conv.out;
  }
  return model;
}

async function runAll(tp, env) {
  const rad = getModel();
  const res = {};
  for (const spec of SPECS) {
    const srv = await ck.serveX(rad, spec, env, {seqs: 1, tp});
    assert.ok(!srv.failed, (srv.log || '').split('\n').slice(-30).join('\n'));
    try {
      res[spec] = [];
      for (const p of PROMPTS) res[spec].push(await ck.genFull(srv, p, NT));   // one request at a time, alone in the engine
    } finally { srv.stop(); }
  }
  return res;
}

for (const tp of [1, 2]) {
  test(`TP${tp}: spec ${SPECS.join('/')} give identical tokens; with ERNIE_A16_MAX=0 identical top-3 logprobs too; spec 0 drafts nothing, the others draft`, {timeout: 2400000}, async () => {
    const pinned = await runAll(tp, {ERNIE_A16_MAX: '0'});
    const base = pinned[SPECS[0]];
    for (const spec of SPECS) {
      pinned[spec].forEach((g, i) => {
        assert.ok(eq(g.ids, base[i].ids), `TP${tp} pinned spec ${spec} prompt ${i}: tokens differ from spec ${SPECS[0]}`);
        assert.ok(eq(g.top, base[i].top), `TP${tp} pinned spec ${spec} prompt ${i}: top logprobs differ from spec ${SPECS[0]}`);
      });
    }
    assert.equal(base.reduce((a, g) => a + g.drafted, 0), 0, 'spec 0 must draft nothing');
    for (const spec of SPECS.slice(1)) assert.ok(pinned[spec].reduce((a, g) => a + g.drafted, 0) > 0, `spec ${spec} drafted nothing`);
    const free = await runAll(tp, {});
    // informational: at the shipped numerics (no pin) the target's row-count dependent bands may move a near tie; reported, not asserted
    for (const spec of SPECS) console.log(`TP${tp} default numerics spec ${spec}: ${free[spec].filter((g, i) => !eq(g.ids, free[SPECS[0]][i].ids)).length} of ${PROMPTS.length} prompts differ in tokens from spec ${SPECS[0]}`);
  });
}
