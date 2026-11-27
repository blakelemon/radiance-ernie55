'use strict';
// The ERNIE DSpark drafter's declaration, on a tiny synthetic checkpoint with the drafter (MXFP4 routed experts, block-fp8 projections), on the CPU
// reference kernels. Covers what a kernel test cannot see because it lives in the plugin's declare path (arch/ernie55/ernie55.cpp):
//   * the drafter's expert tables are the trunk's moe_gemm_q schema (mxfp4a8, activation group 128, 2 stacked outputs for gate|up, 1 for down);
//   * the drafter's declared window is the window its attention is gated with (0 would declare the cache unbounded: after a prefix hit that left a
//     hole in it the scheduler's reach test never passes and the sequence never drafts again);
//   * a draft depth of 1 is refused at startup by name (the block is the anchor plus depth - 1 mask rows and the scheduler refuses a block below 2),
//     and every accepted depth 2..7 proposes tokens.
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:
//   nosys node --test tests/ernie55/dspark_declare.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ck = require('./support/dspark_ckpt.cjs');
const {h} = ck;

let model;
function getModel() {
  if (!model) {
    const c = ck.buildCheckpoint('dspark-declare');
    const conv = ck.convertDspark(c.dir, 7, 'declare');
    assert.equal(conv.status, 0, 'convert must work: ' + (conv.stderr || '').slice(-600));
    model = conv.out;
  }
  return model;
}

// `radiance --config` with debug-graph: the declared program, printed instead of served (host kernels, no card).
function debugGraph(spec) {
  const b = h.ensureBuilt();
  const y = path.join(h.tmpRoot, `dg${spec}.yaml`);
  fs.writeFileSync(y, ['server:', '  host: 127.0.0.1', '  port: 0', '  log: info', `  radiance-home: ${b.home}`, '  kernels: libref',
    '  debug-accept-reference-kernels: true', 'cards: all', 'models:', '  ern:', `    model: ${getModel()}`, '    tp: 1', '    debug-graph: true',
    '    max-num-seqs: 2', '    max-num-batched-tokens: 64', '    max-model-len: 256', `    num-speculative-tokens: ${spec}`, ''].join('\n'));
  const r = h.sh(b.radiance, ['--config', y], {env: {...process.env, ROCR_VISIBLE_DEVICES: '99', RADIANCE_HOST_DEVICES: '1', RADIANCE_HOST_VRAM_MIB: '512'}});
  return {status: r.status, text: r.stdout + r.stderr};
}

// The program's op blocks: a header line (`op k=v ...`) followed by indented lines, blocks separated by blank lines.
function blocks(text) { return text.split(/\n\s*\n/).map(b => b.split('\n')).filter(b => /^[a-z_0-9]+  /.test(b[0])); }
const sig = header => { const p = Object.fromEntries(header.split(/\s+/).slice(1).map(kv => kv.split('='))); return {op: header.split(/\s+/)[0], N: p.N, K: p.K, group: p.group, dtype: p.dtype, parts: p.parts, order: p.a_order}; };

test('the drafter experts are declared as the trunk declares them: mxfp4a8 at group 128, 2 stacked outputs for gate|up and 1 for down', () => {
  const g = debugGraph(7);
  assert.equal(g.status, 0, g.text.slice(-800));
  const mine = {gate_up: [], down: []};
  const trunk = {gate_up: [], down: []};
  for (const b of blocks(g.text)) {
    if (!b[0].startsWith('moe_gemm_q')) continue;
    const w = b.find(l => /^\s+weights\s/.test(l)) || '';
    const kind = /ffn_gate_up_exps/.test(w) ? 'gate_up' : /ffn_down_exps/.test(w) ? 'down' : null;
    if (!kind) continue;
    (/dspark\.blk\./.test(w) ? mine : trunk)[kind].push(sig(b[0]));
  }
  for (const kind of ['gate_up', 'down']) {
    assert.ok(mine[kind].length >= 1, `no drafter ${kind} expert op was declared`);
    assert.ok(trunk[kind].length >= 1, `no trunk ${kind} expert op was declared`);
    const want = kind === 'gate_up' ? {dtype: 'mxfp4a8', group: '128', parts: '2', order: 'token'} : {dtype: 'mxfp4a8', group: '128', parts: '1', order: 'sorted'};
    for (const s of mine[kind]) for (const k of Object.keys(want)) assert.equal(s[k], want[k], `drafter ${kind}: ${k} is ${s[k]}, want ${want[k]}`);
    for (const k of ['N', 'K', 'dtype', 'group', 'parts', 'order']) assert.equal(mine[kind][0][k], trunk[kind][0][k], `drafter ${kind} ${k} differs from the trunk's`);
  }
  assert.equal(g.text.split('\n').filter(l => /^moe_gemm\s/.test(l)).length, 0, 'a bf16 moe_gemm was declared: an expert table fell back from mxfp4');
});

test('the drafter window it declares is the window of its attention', () => {
  const g = debugGraph(7);
  assert.equal(g.status, 0, g.text.slice(-800));
  const decl = /drafter window (\d+)/.exec(g.text);
  assert.ok(decl, 'no drafter window line in the program');
  const att = blocks(g.text).find(b => b[0].startsWith('ernie_hca_dspark_attn'));
  assert.ok(att, 'no ernie_hca_dspark_attn op');
  const win = /window=(\d+)/.exec(att[0]);
  assert.ok(win, att[0]);
  assert.equal(decl[1], win[1], `declared drafter window ${decl[1]} but the attention is gated with ${win[1]}`);
  assert.notEqual(decl[1], '0');
});

test('a draft depth of 1 is refused at startup, naming the minimum depth 2', () => {
  const g = debugGraph(1);
  assert.notEqual(g.status, 0, 'depth 1 must not start');
  assert.match(g.text, /DSpark drafter needs a draft depth .* of at least 2, got 1/);
  assert.match(g.text, /use 0 to run without the drafter, or 2\.\.7/);
});

test('every accepted depth 2..7 proposes tokens', {timeout: 900000}, async () => {
  const rad = getModel();
  for (const depth of [2, 3, 7]) {
    const srv = await ck.serveX(rad, depth, {ERNIE_A16_MAX: '0'}, {seqs: 2, tokens: 64});
    assert.ok(!srv.failed, `depth ${depth} did not start: ` + (srv.log || '').split('\n').slice(-12).join('\n'));
    try {
      const r = await ck.genFull(srv, [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8].map(x => x % ck.VOCAB), 40);
      assert.equal(r.ids.length, 40);
      assert.ok(r.drafted > 0, `depth ${depth} drafted nothing`);
    } finally { srv.stop(); }
  }
});
