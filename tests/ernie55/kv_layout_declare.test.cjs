'use strict';
// The HCA compressed KV group at architecture level, on a tiny synthetic ERNIE checkpoint on the CPU reference kernels: what the operator tests (libref / SIMT emulator) cannot
// see because it lives in the plugin's declare path (arch/ernie55/ernie55.cpp, ernie55_hca_cmp_layout):
//   * the group's block size and page bytes in each layout: `token` (the 3-4b form) is one slot per token in the model's attention block, `block` (the default where head_dim is a
//     multiple of 2 * period) is one row per `period`-token block, a page being exactly `period` rows of head_dim / period elements, i.e. one row;
//   * every other KV group is identical in both layouts (same block, same page bytes);
//   * the program carries what the layout needs: ernie_hca_compress out_layout, ernie_hca_union_attn cmp_layout and the group's block_size, token or block;
//   * the serve-time override RADIANCE_ERNIE55_HCA_CMP_LAYOUT and an unknown value refused.
// The last test runs requests through both layouts and compares tokens and top-5 logprobs bit for bit (a repeated prompt and a shared prefix with divergent suffixes through the prefix
// cache and its checkpoints, prompts and a chunk quantum that put chunk boundaries around the compression period, a run that recycles the 8-token window many times); the wider
// matrix (TP1 and TP2, attention blocks 8 and 2, period 128, against the pre-layout build) is the end-to-end evidence for the kv-cmpblock layout.
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:
//   nosys node --test tests/ernie55/kv_layout_declare.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const kv = require('./support/kv_ckpt.cjs');

const ROW_BYTES = 2 * kv.HGEO.D;                  // one compressed row, bf16
const LAYERS = 2;                                 // HCA layers of the checkpoint (M H M H)
const OTHERS = ['kv_latent', 'kv_dsa_index', 'kv_hca_window', 'kv_hca_state'];

async function alloc(keys, env = {}, hover = {}) {
  const srv = await kv.serve(kv.convertTarget(keys, hover), {env});
  assert.ok(!srv.failed, 'the engine must start: ' + (srv.log || '').split('\n').slice(-12).join('\n'));
  try { return kv.groups(srv.log); } finally { srv.stop(); }
}

test('token layout: one slot per token in the attention block; block layout: one row per period tokens', {timeout: 600000}, async () => {
  const tok = await alloc(['ernie55_hca_cmp_layout=token']);
  const blk = await alloc(['ernie55_hca_cmp_layout=block']);
  const auto = await alloc([]);
  for (const g of [tok, blk, auto]) assert.ok(g.kv_hca_compressed, 'no kv_hca_compressed group in the allocator report');
  assert.equal(tok.kv_hca_compressed.block, kv.ATTN_BLOCK, 'token layout: the group uses the attention block');
  assert.equal(tok.kv_hca_compressed.pageBytes, LAYERS * kv.ATTN_BLOCK * ROW_BYTES, 'token layout: a page is block rows of one token each, per layer');
  assert.equal(blk.kv_hca_compressed.block, kv.PERIOD, 'block layout: the group block is the period');
  assert.equal(blk.kv_hca_compressed.pageBytes, LAYERS * ROW_BYTES, 'block layout: a page is exactly one compressed row per layer');
  assert.equal(auto.kv_hca_compressed.block, kv.PERIOD, 'the default (auto) is the block layout where head_dim % (2 * period) == 0');
  assert.equal(auto.kv_hca_compressed.pageBytes, LAYERS * ROW_BYTES);
  for (const name of OTHERS) {
    for (const [what, g] of [['block', blk], ['auto', auto]]) {
      assert.ok(tok[name] && g[name], `${name} missing`);
      assert.equal(g[name].block, tok[name].block, `${name} block differs in ${what} layout`);
      assert.equal(g[name].pageBytes, tok[name].pageBytes, `${name} page bytes differ in ${what} layout`);
      assert.equal(g[name].kind, tok[name].kind, `${name} kind differs in ${what} layout`);
    }
  }
});

test('the program carries the layout: out_layout on the compressor, cmp_layout and block_size on the union attention', {timeout: 600000}, () => {
  for (const [keys, want] of [[['ernie55_hca_cmp_layout=token'], 'token'], [['ernie55_hca_cmp_layout=block'], 'block'], [[], 'block']]) {
    const g = kv.declare(kv.convertTarget(keys));
    assert.equal(g.status, 0, g.text.slice(-800));
    const bs = kv.blocks(g.text);
    const comp = bs.filter(b => b[0].startsWith('ernie_hca_compress'));
    const att = bs.filter(b => b[0].startsWith('ernie_hca_union_attn'));
    assert.ok(comp.length >= 1 && att.length >= 1, 'no compressor / union attention op in the program');
    for (const b of comp) {
      const p = kv.params(b[0]);
      if (want === 'block') assert.equal(p.out_layout, 'block', b[0]); else assert.ok(!p.out_layout || p.out_layout === 'token', b[0]);
    }
    for (const b of att) {
      const p = kv.params(b[0]);
      assert.equal(p.cmp_layout, want, b[0]);
      if (want === 'block') assert.equal(p.block_size, String(kv.PERIOD), b[0]);
    }
  }
});

test('the serve-time override wins over the container key; an unknown layout is refused by name', {timeout: 600000}, async () => {
  const blk = await alloc(['ernie55_hca_cmp_layout=block'], {RADIANCE_ERNIE55_HCA_CMP_LAYOUT: 'token'});
  assert.equal(blk.kv_hca_compressed.block, kv.ATTN_BLOCK, 'RADIANCE_ERNIE55_HCA_CMP_LAYOUT=token must give the token layout');
  const bad = kv.declare(kv.convertTarget(['ernie55_hca_cmp_layout=block']), {env: {RADIANCE_ERNIE55_HCA_CMP_LAYOUT: 'rows'}});
  assert.notEqual(bad.status, 0, 'an unknown layout must not start');
  assert.match(bad.text, /ernie55_hca_cmp_layout must be auto, block or token, got "rows"/);
});

// head_dim 124 at period 4: 124 % (2 * 4) != 0, so a block page (period rows of head_dim / period) cannot be one KV row of the declaration. auto must fall back to the token layout and
// an explicit block must be refused by name.
test('a head_dim that is not a multiple of 2 * period: auto falls back to the token layout, an explicit block is refused by name', {timeout: 600000}, async () => {
  const narrow = {D: 124};
  const auto = await alloc([], {}, narrow);
  const tok = await alloc(['ernie55_hca_cmp_layout=token'], {}, narrow);
  assert.equal(auto.kv_hca_compressed.block, kv.ATTN_BLOCK, 'auto must give the token layout where head_dim % (2 * period) != 0');
  // the report prints a page with two decimals (3.88 KiB), so compare to the report's resolution
  assert.ok(Math.abs(auto.kv_hca_compressed.pageBytes - LAYERS * kv.ATTN_BLOCK * 2 * 124) <= 0.005 * LAYERS * kv.ATTN_BLOCK * 2 * 124, 'page bytes ' + auto.kv_hca_compressed.pageBytes);
  assert.deepEqual(auto.kv_hca_compressed, tok.kv_hca_compressed);
  const g = kv.declare(kv.convertTarget([], narrow));
  assert.equal(g.status, 0, g.text.slice(-800));
  for (const b of kv.blocks(g.text).filter(b => b[0].startsWith('ernie_hca_union_attn'))) assert.equal(kv.params(b[0]).cmp_layout, 'token', b[0]);
  // the plugin's declaration runs at conversion too: an explicit block at an incompatible head_dim does not even convert
  const bad = kv.tryConvert(['ernie55_hca_cmp_layout=block'], narrow);
  assert.notEqual(bad.status, 0, 'block at an incompatible head_dim must not convert');
  assert.match(bad.stderr, /ernie55_hca_cmp_layout=block needs head_dim to be a multiple of 2 \* period/);
});

// Engine integration: the two layouts must serve the same tokens and logprobs. One sequence at a time (the target is not batch invariant), checkpoints on so a prefix hit of this
// windowed model resumes from one, a chunk quantum of 8 (two periods) and of 64.
async function runSet(layout, maxtok) {
  const srv = await kv.serve(kv.convertTarget([`ernie55_hca_cmp_layout=${layout}`]), {maxtok, ckpt: true});
  assert.ok(!srv.failed, 'the engine must start: ' + (srv.log || '').split('\n').slice(-12).join('\n'));
  try {
    const out = {};
    for (const len of [1, 3, 4, 5, 7, 8, 9, 12, 13, 17, 33]) out['len' + len] = await kv.gen(srv, kv.rndPrompt(100 + len, len), 12);
    out.long = await kv.gen(srv, kv.rndPrompt(55, 13), 120);                         // the window of 8 recycled ~15 times, ~30 periods crossed
    const X = kv.rndPrompt(77, 41), Y = X.slice(0, 24).concat(kv.rndPrompt(88, 15));   // 41 tokens: a hit of 40; Y shares 24 then diverges
    out.pfx_X = await kv.gen(srv, X, 16);
    out.pfx_Y = await kv.gen(srv, Y, 16);
    out.pfx_X_again = await kv.gen(srv, X, 16);
    out.counters = await kv.prefixCounters(srv);
    return out;
  } finally { srv.stop(); }
}
for (const maxtok of [64, 8]) {
  test(`token and block layouts serve identical tokens and logprobs (chunk quantum ${maxtok}), prefix reuse included`, {timeout: 900000}, async () => {
    const tok = await runSet('token', maxtok), blk = await runSet('block', maxtok);
    for (const name of Object.keys(tok)) assert.deepEqual(blk[name], tok[name], `${name} differs between the layouts`);
    assert.ok(tok.counters.length > 0, 'no prefix-cache counters in /metrics');
    assert.ok(tok.pfx_X_again.ids.length === 16 && tok.long.ids.length === 120, 'the runs generated what was asked');
    assert.deepEqual(tok.pfx_X_again.ids, tok.pfx_X.ids, 'a repeated prompt generates the same tokens');
  });
}
