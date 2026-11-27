'use strict';
// kv-fp8-latent v2: RADIANCE_ERNIE55_KV_LATENT_FMT at architecture level, on the tiny synthetic ERNIE checkpoint (latent 64, rope 16, 2 MLA layers) on the CPU reference kernels.
// The operator tests cannot see the selector's wiring in arch/ernie55/ernie55.cpp. Covered here:
//   * unset and ds_mla declare the same program as plain --kv-cache-dtype fp8 (the default graph is unchanged);
//   * i8gN declares kv_latent as the i8_ds_mla row (64 codes + 4 * 64 / N scale bytes + 32 rope bytes), one store per MLA layer naming kv_layout i8_ds_mla and quant_block N,
//     and the sparse attention reading kv_layout i8_ds_mla with the same quant_block; every other group keeps its page bytes;
//   * i10gN / i12gN (the 10- and 12-bit fallback rows) the same with kv_layout i10_ds_mla / i12_ds_mla and the packed low-bit bytes (LAT / 4, LAT / 2) in the row;
//   * a value that is not a known format, or whose group does not divide the latent width (i8g128 at latent 64), refuses to start;
//   * under bf16 the selector is ignored (no store op, no layout);
//   * one store + read step with i8g16: finite logprobs that differ from bf16's by a small bounded amount.
// Needs node, a host build (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:  nosys node --test tests/ernie55/kv_latent_fmt_declare.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const kv = require('./support/kv_ckpt.cjs');

const MLA_LAYERS = 2, LAT = 64, ROPE = 16;
const OTHERS = ['kv_dsa_index', 'kv_hca_window', 'kv_hca_state', 'kv_hca_compressed'];
const env = fmt => (fmt === undefined ? {} : {RADIANCE_ERNIE55_KV_LATENT_FMT: fmt});
const ops = (kvdt, fmt) => {
  const g = kv.declare(kv.convertTarget(), {kvdt, env: env(fmt)});
  if (g.status !== 0) return {status: g.status, text: g.text};
  const bs = kv.blocks(g.text);
  return {status: 0, text: g.text,
          store: bs.filter(b => b[0].startsWith('ernie_mla_latent_store')).map(b => kv.params(b[0])),
          attn: bs.filter(b => b[0].startsWith('ernie_mla_sparse_attn')).map(b => kv.params(b[0]))};
};

test('the selector: default and ds_mla are the fp8_ds_mla program; i8gN names i8_ds_mla in the store and the reader; bf16 ignores it', {timeout: 600000}, () => {
  const def = ops('fp8'), ds = ops('fp8', 'ds_mla');
  assert.equal(def.status, 0, def.text.slice(-800)); assert.equal(ds.status, 0, ds.text.slice(-800));
  assert.deepEqual(ds.store, def.store); assert.deepEqual(ds.attn, def.attn);
  for (const p of def.store) assert.ok(!('kv_layout' in p), 'the default store names no layout (graph unchanged)');
  for (const p of def.attn) assert.equal(p.kv_layout, 'fp8_ds_mla');
  for (const n of [64, 32, 16]) {
    const o = ops('fp8', `i8g${n}`);
    assert.equal(o.status, 0, o.text.slice(-800));
    assert.equal(o.store.length, MLA_LAYERS, `i8g${n}: one store per MLA layer`);
    for (const p of o.store) { assert.equal(p.kv_layout, 'i8_ds_mla'); assert.equal(p.quant_block, String(n)); assert.equal(p.latent_dim, String(LAT)); }
    assert.equal(o.attn.length, MLA_LAYERS);
    for (const p of o.attn) { assert.equal(p.kv_layout, 'i8_ds_mla'); assert.equal(p.quant_block, String(n)); }
  }
  for (const [bits, lay] of [[10, 'i10_ds_mla'], [12, 'i12_ds_mla']]) for (const n of [32, 16]) {
    const o = ops('fp8', `i${bits}g${n}`);
    assert.equal(o.status, 0, o.text.slice(-800));
    assert.equal(o.store.length, MLA_LAYERS, `i${bits}g${n}: one store per MLA layer`);
    for (const p of o.store) { assert.equal(p.kv_layout, lay); assert.equal(p.quant_block, String(n)); assert.equal(p.latent_dim, String(LAT)); }
    assert.equal(o.attn.length, MLA_LAYERS);
    for (const p of o.attn) { assert.equal(p.kv_layout, lay); assert.equal(p.quant_block, String(n)); }
  }
  const bf = ops('bf16', 'i8g16');
  assert.equal(bf.status, 0, bf.text.slice(-800));
  assert.equal(bf.store.length, 0, 'bf16 ignores the selector');
  for (const p of bf.attn) assert.ok(!('kv_layout' in p) || p.kv_layout === 'bf16');
});

test('a bad selector value refuses to start', {timeout: 600000}, () => {
  for (const bad of ['i8g128', 'i12g128', 'i8', 'fp8', 'i8g48', 'i11g32', 'i12', 'i12g32x']) {
    const o = ops('fp8', bad);
    assert.notEqual(o.status, 0, `RADIANCE_ERNIE55_KV_LATENT_FMT=${bad} must refuse`);
    assert.match(o.text, /RADIANCE_ERNIE55_KV_LATENT_FMT must be ds_mla, or i8gN, i10gN, i12gN with N one of 128, 64, 32, 16/);
  }
});

test('i8gN allocates the i8_ds_mla row and leaves every other group alone; one step serves bounded logprobs', {timeout: 900000}, async () => {
  const start = async (kvdt, fmt) => {
    const srv = await kv.serve(kv.convertTarget(), {kvdt, env: env(fmt)});
    assert.ok(!srv.failed, `the engine must start (${kvdt} ${fmt}): ` + (srv.log || '').split('\n').slice(-12).join('\n'));
    return srv;
  };
  let srv = await start('fp8', undefined); const fp = kv.groups(srv.log); srv.stop();
  for (const [fmt, lay, lo, n] of [['i8', 'i8_ds_mla', 0, 32], ['i8', 'i8_ds_mla', 0, 16], ['i10', 'i10_ds_mla', LAT / 4, 32], ['i12', 'i12_ds_mla', LAT / 2, 32], ['i12', 'i12_ds_mla', LAT / 2, 16]]) {
    srv = await start('fp8', `${fmt}g${n}`);
    const g = kv.groups(srv.log), log = srv.log; srv.stop();
    const row = LAT + lo + 4 * (LAT / n) + 2 * ROPE, want = MLA_LAYERS * kv.ATTN_BLOCK * row;
    assert.ok(Math.abs(g.kv_latent.pageBytes - want) <= 0.005 * want, `${fmt}g${n}: latent page ${g.kv_latent.pageBytes}, want ${want}`);
    assert.match(log, new RegExp(`stored as ${lay} rows \\(${row} bytes a token and layer`));
    for (const name of OTHERS) assert.equal(g[name].pageBytes, fp[name].pageBytes, `${name} page bytes under ${fmt}g${n}`);
  }
  const prompt = kv.rndPrompt(31, 13);
  const run = async (kvdt, fmt) => { const s = await start(kvdt, fmt); try { return await kv.gen(s, prompt, 8); } finally { s.stop(); } };
  const bf = await run('', undefined), i8 = await run('fp8', 'i8g16'), i12 = await run('fp8', 'i12g16');
  let maxd = 0, differs = 0, n = 0;
  for (let i = 0; i < 8; i++) for (const [id, lp] of i8.top[i]) {
    assert.ok(Number.isFinite(lp), `step ${i}: non-finite logprob under i8g16`);
    const other = bf.top[i].find(t => t[0] === id);
    if (other) { const d = Math.abs(lp - other[1]); maxd = Math.max(maxd, d); n++; if (d > 0) differs++; }
  }
  assert.ok(n > 0); assert.ok(differs > 0, 'i8g16 logprobs are bit-identical to bf16: the i8 cache was not read');
  assert.ok(maxd < 0.25, `i8g16 moved a logprob by ${maxd}`);
  assert.equal(i8.ids[0], bf.ids[0]);
  /* the 12-bit row is read too: finite, differs from bf16, bounded. (No ordering against i8: the bf16 run rounds its cache to 8 mantissa bits, so a row
   * finer than bf16 is not "closer" to it than an 8-bit row; the offline error table in NOTES.md carries the accuracy ordering.) */
  let d12 = 0, m12 = 0;
  for (let i = 0; i < 8; i++) for (const [id, lp] of i12.top[i]) {
    assert.ok(Number.isFinite(lp), `step ${i}: non-finite logprob under i12g16`);
    const other = bf.top[i].find(t => t[0] === id);
    if (other) { d12 += Math.abs(lp - other[1]); m12 = Math.max(m12, Math.abs(lp - other[1])); }
  }
  assert.ok(d12 > 0, 'i12g16 logprobs are bit-identical to bf16: the i12 cache was not read');
  assert.ok(m12 < 0.25, `i12g16 moved a logprob by ${m12}`);
  assert.equal(i12.ids[0], bf.ids[0]);
});
