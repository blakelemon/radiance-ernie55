'use strict';
// --kv-cache-dtype fp8 at architecture level, on a tiny synthetic ERNIE checkpoint on the CPU reference kernels. The operator tests (libref and the SIMT emulator) cannot see
// the wiring in the plugin's declare path (arch/ernie55/ernie55.cpp): the capability probe of the model entry's kv dtype, the u8 allocation of the MLA latent group, its binding
// to every MLA layer, the store op that issues the fp8_ds_mla row, and the reader's parameters. Covered here, for the default (bf16), an explicit bf16 and fp8, at TP1 and TP2:
//   * kv_latent's dtype-dependent page bytes (bf16 row 80 values x 2 B = 160 B a token and layer; fp8_ds_mla row 64 codes + 4 scale bytes + 16 rope x 2 B = 100 B), and that every
//     other group (indexer, HCA window, state, compressed) keeps its block, page bytes and kind;
//   * the program: in fp8 one ernie_mla_latent_store per MLA layer with the row geometry and the sparse attention reading kv_layout fp8_ds_mla with the same quant_block; in
//     bf16 neither parameter and no store op;
//   * one store + read step: a request completes, its logprobs are finite, and they differ from the bf16 cache's (the quantised cache was really read) by a small bounded amount.
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:
//   nosys node --test tests/ernie55/kv_fp8_latent_declare.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const kv = require('./support/kv_ckpt.cjs');

const MLA_LAYERS = 2, LAT = 64, ROPE = 16, QB = 64;            // the synthetic checkpoint: layers M H M H, latent 64, rope 16; quant_block = the largest of 128, 64, ... dividing 64
const BF16_ROW = 2 * (LAT + ROPE), FP8_ROW = LAT + 4 * (LAT / QB) + 2 * ROPE;
const OTHERS = ['kv_dsa_index', 'kv_hca_window', 'kv_hca_state', 'kv_hca_compressed'];

async function started(kvdt, tp = 1) {
  const srv = await kv.serve(kv.convertTarget(), {kvdt, tp});
  assert.ok(!srv.failed, `the engine must start (kv-cache-dtype ${kvdt || 'default'}, TP${tp}): ` + (srv.log || '').split('\n').slice(-12).join('\n'));
  return srv;
}

test('kv_latent is a u8 fp8_ds_mla group under fp8 and bf16 otherwise; every other group keeps its dtype and size', {timeout: 900000}, async () => {
  const seen = {};
  for (const [name, kvdt, tp] of [['default', '', 1], ['bf16', 'bf16', 1], ['fp8', 'fp8', 1], ['fp8 tp2', 'fp8', 2], ['default tp2', '', 2]]) {
    const srv = await started(kvdt, tp);
    try { seen[name] = kv.groups(srv.log); if (kvdt === 'fp8') seen[name].log = srv.log; } finally { srv.stop(); }
  }
  const bf = seen.default, fp = seen.fp8;
  assert.deepEqual(seen.bf16.kv_latent, bf.kv_latent, 'an explicit bf16 is the default');
  assert.equal(bf.kv_latent.pageBytes, MLA_LAYERS * kv.ATTN_BLOCK * BF16_ROW, 'bf16 latent page');
  const want = MLA_LAYERS * kv.ATTN_BLOCK * FP8_ROW;
  assert.ok(Math.abs(fp.kv_latent.pageBytes - want) <= 0.005 * want, `fp8 latent page ${fp.kv_latent.pageBytes}, want ${want} (the report prints two decimals)`);
  assert.equal(fp.kv_latent.block, bf.kv_latent.block);
  assert.equal(fp.kv_latent.kind, 'full');
  assert.match(fp.log, /--kv-cache-dtype fp8: the MLA latent cache \(kv_latent\) is stored as fp8_ds_mla rows \(100 bytes a token and layer/);
  for (const name of OTHERS) {
    assert.ok(bf[name] && fp[name], `${name} missing`);
    assert.equal(fp[name].pageBytes, bf[name].pageBytes, `${name} page bytes changed under fp8`);
    assert.equal(fp[name].block, bf[name].block, `${name} block changed under fp8`);
    assert.equal(fp[name].kind, bf[name].kind, `${name} kind changed under fp8`);
  }
  for (const [mode, g] of [['fp8 tp2', seen['fp8 tp2']], ['default tp2', seen['default tp2']]]) {
    const ref = mode.startsWith('fp8') ? fp : bf;
    for (const name of ['kv_latent', ...OTHERS]) assert.equal(g[name].pageBytes, ref[name].pageBytes, `${name} page bytes at TP2 (${mode})`);
  }
});

test('the program issues the store and reads the fp8_ds_mla row only under fp8', {timeout: 600000}, () => {
  const ops = kvdt => {
    const g = kv.declare(kv.convertTarget(), {kvdt});
    assert.equal(g.status, 0, g.text.slice(-800));
    const bs = kv.blocks(g.text);
    return {store: bs.filter(b => b[0].startsWith('ernie_mla_latent_store')).map(b => kv.params(b[0])),
            attn: bs.filter(b => b[0].startsWith('ernie_mla_sparse_attn')).map(b => kv.params(b[0]))};
  };
  const fp = ops('fp8'), bf = ops('');
  assert.equal(fp.store.length, MLA_LAYERS, 'one latent store per MLA layer under fp8');
  for (const p of fp.store) {
    assert.equal(p.latent_dim, String(LAT)); assert.equal(p.rope_dim, String(ROPE)); assert.equal(p.quant_block, String(QB)); assert.equal(p.block_size, String(kv.ATTN_BLOCK));
  }
  assert.equal(fp.attn.length, MLA_LAYERS);
  for (const p of fp.attn) { assert.equal(p.kv_layout, 'fp8_ds_mla'); assert.equal(p.quant_block, String(QB)); }
  assert.equal(bf.store.length, 0, 'no latent store op in bf16');
  assert.equal(bf.attn.length, MLA_LAYERS);
  for (const p of bf.attn) { assert.ok(!('kv_layout' in p) || p.kv_layout === 'bf16', 'the bf16 reader names no fp8 layout'); assert.ok(!('quant_block' in p)); }
});

test('one store and read step: fp8 serves finite logprobs that differ from the bf16 cache by a small bounded amount', {timeout: 900000}, async () => {
  for (const tp of [1, 2]) {
    const prompt = kv.rndPrompt(31, 13);
    const run = async kvdt => { const srv = await started(kvdt, tp); try { return await kv.gen(srv, prompt, 8); } finally { srv.stop(); } };
    const bf = await run(''), fp = await run('fp8');
    assert.equal(fp.ids.length, 8); assert.equal(bf.ids.length, 8);
    let maxd = 0, differs = 0, n = 0;
    for (let i = 0; i < 8; i++) {
      for (const [id, lp] of fp.top[i]) {
        assert.ok(Number.isFinite(lp), `step ${i}: non-finite logprob under fp8`);
        const other = bf.top[i].find(t => t[0] === id);
        if (other) { const d = Math.abs(lp - other[1]); maxd = Math.max(maxd, d); n++; if (d > 0) differs++; }
      }
    }
    assert.ok(n > 0, 'no shared top-5 token between the two caches');
    assert.ok(differs > 0, `TP${tp}: fp8 logprobs are bit-identical to bf16's: the quantised cache was not read`);
    assert.ok(maxd < 0.25, `TP${tp}: the lossy cache moved a logprob by ${maxd}`);
    assert.equal(fp.ids[0], bf.ids[0], `TP${tp}: the first generated token differs`);
  }
});
