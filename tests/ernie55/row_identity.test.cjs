'use strict';
// Row identity end to end (the row-identity checks), CPU reference kernels, tiny synthetic ERNIE + DSpark checkpoint, NO env pins.
// At TP 1 and 2: every prompt's tokens AND top logprobs at speculation 2, 3 and 7 equal speculation 0 (lossless), and a request run beside
// three partners (staggered) equals itself alone at every speculation depth (batch independence), including the partners' prefill sharing steps.
// Needs node, a host build (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:  nosys node --test tests/ernie55/row_identity.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const ck = require('./support/dspark_ckpt.cjs');

const NT = Number(process.env.RI_NT || 40);
const TPS = (process.env.RI_TPS || '1,2').split(',').map(Number);
const SPECS = (process.env.RI_SPECS || '0,2,3,7').split(',').map(Number);
const prompts = [12, 21, 2, 33].map((n, i) => ck.rnd(500 + 41 * i, n));
prompts[1] = [17, 17, 17, 17, 2, 2, 2, 40, 41, 42, 17, 17, 60, 33, 21, 9, 9, 9, 50, 51, 52].map(x => x % ck.VOCAB);
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// RI_A16=8 (control): the old small-step activation rule, which must make this test fail (teeth).
const XENV = process.env.RI_A16 ? {ERNIE_A16_MAX: process.env.RI_A16} : {};
for (const k of ['ERNIE_A16_MAX', 'ERNIE_VHA_DEC_MAX', 'ERNIE_ROUTER_SPLIT']) delete process.env[k];

// Runs every (tp, spec): each prompt alone, then four staggered concurrent (steps up to 4 x (1 + spec) rows, past the old eight-row small-step bound
// at spec 3 and 7). Returns the differences: concurrent != alone at that spec, alone != alone at spec SPECS[0], and concurrent != alone at spec
// SPECS[0] (the lossless claim under load: a sequence's tokens and logprobs do not depend on the speculation depth or on its neighbours).
async function collect(conv, tps, specs, xenv, bad) {
  for (const tp of tps) {
    let ref = null;
    for (const spec of specs) {
      const srv = await ck.serveX(conv.out, spec, xenv, {tp});
      assert.ok(!srv.failed, `TP=${tp} spec=${spec}: ` + (srv.log || '').split('\n').slice(-30).join('\n'));
      try {
        const alone = [];
        for (const p of prompts) alone.push(await ck.genFull(srv, p, NT));
        const conc = await Promise.all(prompts.map(async (p, i) => { await new Promise(r => setTimeout(r, 40 * i)); return ck.genFull(srv, p, NT); }));
        let drafted = 0;
        for (let i = 0; i < prompts.length; i++) {
          drafted += alone[i].drafted || 0;
          if (!eq(conc[i].ids, alone[i].ids) || !eq(conc[i].top, alone[i].top)) bad.push(`TP=${tp} spec=${spec} prompt ${i}: concurrent != alone`);
          if (ref && (!eq(alone[i].ids, ref[i].ids) || !eq(alone[i].top, ref[i].top))) bad.push(`TP=${tp} spec=${spec} prompt ${i}: alone != spec ${specs[0]} alone`);
          if (ref && (!eq(conc[i].ids, ref[i].ids) || !eq(conc[i].top, ref[i].top))) bad.push(`TP=${tp} spec=${spec} prompt ${i}: concurrent != spec ${specs[0]} alone`);
        }
        console.log(`TP=${tp} spec=${spec}: drafted=${drafted} accepted=${alone.reduce((a, g) => a + (g.accepted || 0), 0)} tokens=${alone.reduce((a, g) => a + g.ids.length, 0)}`);
        if (spec > 0) assert.ok(drafted > 0, `TP=${tp} spec=${spec} drafted nothing`);
        if (!ref) ref = alone;
      } finally { srv.stop(); }
    }
  }
  return bad;
}

test('spec 0/2/3/7 identical tokens and logprobs, alone == concurrent == spec 0, TP 1 and 2, no pins', {timeout: 3600000}, async () => {
  const c = ck.buildCheckpoint('row-identity');
  const conv = ck.convertDspark(c.dir, 7, 'rowid');
  assert.equal(conv.status, 0, 'convert: ' + (conv.stderr || '').slice(-600));
  const bad = await collect(conv, TPS, SPECS, XENV, []);
  assert.equal(bad.length, 0, bad.join('\n'));
});

// Teeth for the lossless claim: with the OLD small-step rule (ERNIE_A16_MAX=8 ERNIE_VHA_DEC_MAX=8: steps of at most eight rows read the bf16
// activations, larger steps the fp8 ones) four concurrent sequences at spec 7 (steps of up to 32 rows) must NOT equal the same prompts alone at
// spec 0 (steps of one row). The new defaults make them equal (test above). If this test starts passing vacuously the claim above has no teeth.
test('control: the old bands make concurrent spec 7 differ from spec 0 alone', {timeout: 3600000}, async () => {
  const c = ck.buildCheckpoint('row-identity-ctl');
  const conv = ck.convertDspark(c.dir, 7, 'rowidctl');
  assert.equal(conv.status, 0, 'convert: ' + (conv.stderr || '').slice(-600));
  const bad = await collect(conv, [1], [0, 7], {ERNIE_A16_MAX: '8', ERNIE_VHA_DEC_MAX: '8'}, []);
  console.log('control differences:\n' + bad.join('\n'));
  assert.ok(bad.some(m => /spec=7 .*concurrent != spec 0 alone/.test(m)), 'the old bands should break spec 7 (concurrent) vs spec 0 (alone); got: ' + bad.join(' | '));
});
