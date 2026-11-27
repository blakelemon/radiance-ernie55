'use strict';
// The collectives overlap (RADIANCE_ERNIE55_AR_OVERLAP=1): a prefill step of a tensor-parallel
// quantised forward run as two pieces with every all_reduce on the second lane must give the
// serial step's tokens and logprobs exactly, at TP 2, 4 and 8 on the host backend (libref kernels;
// the device kernels' row independence is the GPU gate's). The run must show the
// pieced path was taken (its log line), or the comparison proves nothing.
const test = require('node:test');
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const h = require('./support/synth/harness.cjs');
const fx = require('./support/synth/fixture34b.cjs');
// Every sharded dim divides by 4 and stays a multiple of 128 so ALL FP8 + MXFP4 weights shard cleanly at TP=1, 2, 4!
const GEO = {heads: 4, nope: 112, rope: 16, vdim: 128, qlora: 128, lat: 64, ih: 2, id: 64, topk: 4, window: 8, rank: 2, theta: 160000};
const HGEO = {heads: 4, D: 128, rope: 16, qlora: 128, groups: 4, rank: 128, vrank: 2, theta: 160000, period: 4, window: 8, yarn: {factor: 16, bf: 32, bs: 1, orig: 65536}};
const OVER = {
  vocab_size: 16,
  hidden_size: 128, intermediate_size: 512, moe_intermediate_size: 512, moe_latent_size: 128,
  n_routed_experts: 4, num_experts_per_tok: 2, n_shared_experts: 1, hc_mult: 2,
  quantization_config: {
    activation_scheme: "dynamic", fmt: "e4m3", quant_method: "fp8",
    expert_dtype: "mxfp4", scale_fmt: "ue8m0", weight_block_size: [128, 128]
  }
};

const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6];
function bf2f(u16) { const b = Buffer.alloc(4); b.writeUInt32LE((u16 << 16) >>> 0); return b.readFloatLE(0); }
function f2bf(x) { const b = Buffer.alloc(4); b.writeFloatLE(x); let u = b.readUInt32LE(0); u = (u + 0x7fff + ((u >>> 16) & 1)) >>> 16; return u & 0xffff; }

function scaleBuf(buf, factor) {
  const out = Buffer.alloc(buf.length);
  for (let i = 0; i < buf.length; i += 2) out.writeUInt16LE(f2bf(bf2f(buf.readUInt16LE(i)) * factor), i);
  return out;
}

function quantizeMx(buf, N, K) {
  const codes = Buffer.alloc(N * K / 2);
  const scales = Buffer.alloc(N * K / 32);
  for (let r = 0; r < N; r++) {
    for (let g = 0; g < K / 32; g++) {
      let mx = 0;
      const vals = new Float32Array(32);
      for (let i = 0; i < 32; i++) {
        const v = bf2f(buf.readUInt16LE((r * K + g * 32 + i) * 2));
        vals[i] = v;
        if (Math.abs(v) > mx) mx = Math.abs(v);
      }
      let exp = mx > 0 ? Math.ceil(Math.log2(mx / 6.0)) : -126;
      exp = Math.max(-126, Math.min(127, exp));
      scales[r * (K / 32) + g] = exp + 127;
      const sc = Math.pow(2, exp);
      for (let i = 0; i < 32; i++) {
        const q = vals[i] / sc;
        let best = 0, bd = Infinity;
        for (let c = 0; c < 16; c++) { const d = Math.abs(E2M1[c] - q); if (d < bd) { bd = d; best = c; } }
        const bi = r * (K / 2) + g * 16 + (i >> 1);
        if ((i & 1) === 0) codes[bi] = best; else codes[bi] |= (best << 4);
      }
    }
  }
  return {codes, scales};
}

function quantizeFp8(buf, N, K) {
  const tbl = new Float32Array(256);
  for (let u = 0; u < 256; u++) {
    const s = (u >> 7) ? -1 : 1, e = (u >> 3) & 0xf, m = u & 7;
    tbl[u] = (e === 15 && m === 7) ? NaN : (e === 0 ? s * (m / 8) * Math.pow(2, -6) : s * (1 + m / 8) * Math.pow(2, e - 7));
  }
  const codes = Buffer.alloc(N * K);
  const bn = N / 128, bk = K / 128;
  const scales = Buffer.alloc(bn * bk * 4);
  for (let br = 0; br < bn; br++) for (let bc = 0; bc < bk; bc++) {
    let mx = 0;
    for (let i = 0; i < 128; i++) for (let j = 0; j < 128; j++) {
      const v = Math.abs(bf2f(buf.readUInt16LE(((br * 128 + i) * K + bc * 128 + j) * 2)));
      if (v > mx) mx = v;
    }
    const sc = mx > 0 ? mx / 448.0 : 1.0;
    scales.writeFloatLE(sc, (br * bk + bc) * 4);
    for (let i = 0; i < 128; i++) for (let j = 0; j < 128; j++) {
      const r = br * 128 + i, c = bc * 128 + j;
      const v = bf2f(buf.readUInt16LE((r * K + c) * 2)) / sc;
      let best = 0, bd = Infinity;
      for (let u = 0; u < 256; u++) { if (!Number.isFinite(tbl[u])) continue; const d = Math.abs(tbl[u] - v); if (d < bd) { bd = d; best = u; } }
      codes[r * K + c] = best;
    }
  }
  return {codes, scales};
}

const isExpert = n => /\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$/.test(n);
const isFp8Trunk = (n, sh) => sh.length === 2 && sh[0] % 128 === 0 && sh[1] % 128 === 0 &&
  /(mlp\.(gate|up|down)_proj|shared_experts\.w[123]|fc[12]_latent_proj|self_attn\.(linear_q_down_proj|linear_q_up_proj|linear_kv_proj|q_a_proj|q_b_proj|gate_proj|o_proj|core_attention\.indexer\.wq_b))\.weight$/.test(n);

function serveTp(rad, tp, extraEnv, {seqs = 2, tokens = 32, len = 64} = {}) {
  const b = h.ensureBuilt();
  const dir = fs.mkdtempSync(path.join(h.tmpRoot, `ov-tp${tp}-`));
  const yaml = path.join(dir, 'serve.yaml');
  fs.writeFileSync(yaml, [
    'server:', '  host: 127.0.0.1', '  port: 0', '  log: info', `  radiance-home: ${b.home}`,
    '  kernels: libref', '  debug-accept-reference-kernels: true', 'cards: all',
    'models:', '  ern:', `    model: ${rad}`, `    tp: ${tp}`, `    max-num-seqs: ${seqs}`,
    `    max-num-batched-tokens: ${tokens}`, `    max-model-len: ${len}`, ...(process.env.OV_KVDT ? [`    kv-cache-dtype: ${process.env.OV_KVDT}`] : []), ''
  ].join('\n'));
  const env = {...process.env, ROCR_VISIBLE_DEVICES: '99', RADIANCE_HOST_DEVICES: String(tp),
               RADIANCE_HOST_VRAM_MIB: '512', OMP_NUM_THREADS: '2', OMP_WAIT_POLICY: 'passive', ...extraEnv};
  for (const k of Object.keys(env)) if (env[k] === undefined) delete env[k];
  const child = spawn(b.radiance, ['--config', yaml], {env, stdio: ['ignore', 'pipe', 'pipe']});
  let log = '';
  child.stdout.on('data', d => { log += d; });
  child.stderr.on('data', d => { log += d; });
  return new Promise(resolve => {
    const timer = setTimeout(() => { child.kill('SIGKILL'); resolve({failed: true, log: log + '\n[timeout]'}); }, 180000);
    const tick = setInterval(() => {
      const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(log);
      if (m && /model ern: ready/.test(log)) {
        clearInterval(tick); clearTimeout(timer);
        resolve({port: Number(m[1]), get log() { return log; }, stop: () => { child.kill('SIGKILL'); }});
      }
    }, 50);
    child.on('exit', () => { clearInterval(tick); clearTimeout(timer); resolve({failed: true, log}); });
  });
}

let modelPath;
function model() {
  if (modelPath) return modelPath;
  const dirQ = path.join(h.tmpRoot, 'ov-q-model');
  const cfgQ = fx.config(['M', 'H', 'M', 'H'], 1, OVER, GEO, HGEO);
  const realistic = ts => ts.map(t => (t.dtype === 'BF16' && t.shape.length === 2 && !t.name.includes('embed') && !t.name.includes('vha'))
    ? {...t, data: scaleBuf(t.data, 1.0 / Math.sqrt(t.shape[1]))} : t);
  fx.build(dirQ, cfgQ, {
    seed: 91, geo: GEO, h: HGEO,
    mutate: ts => realistic(ts).flatMap(t => {
      if (isExpert(t.name)) {
        const q = quantizeMx(t.data, t.shape[0], t.shape[1]);
        return [{...t, dtype: 'I8', shape: [t.shape[0], t.shape[1] / 2], data: q.codes},
                {name: t.name + '_scale_inv', dtype: 'F8_E8M0', shape: [t.shape[0], t.shape[1] / 32], data: q.scales}];
      }
      if (isFp8Trunk(t.name, t.shape)) {
        const q = quantizeFp8(t.data, t.shape[0], t.shape[1]);
        return [{...t, dtype: 'F8_E4M3', shape: t.shape, data: q.codes},
                {name: t.name + '_scale_inv', dtype: 'F32', shape: [t.shape[0] / 128, t.shape[1] / 128], data: q.scales}];
      }
      return [t];
    })
  });
  const conv = h.convert(dirQ, {keys: ['ernie55_quant_forward=1', 'ernie55_attention_block=8', 'ernie55_hca_period=4', ...(process.env.OV_TILE ? [`ernie55_idx_tile=${process.env.OV_TILE}`] : [])]});
  assert.equal(conv.status, 0, conv.stderr);
  modelPath = conv.out;
  return modelPath;
}

// TP8 needs every sharded dimension divisible by 8: the smaller bf16 geometry of the TP8 parity
// evidence (the TP8 parity end-to-end run).
let model8Path;
function model8() {
  if (model8Path) return model8Path;
  const dir8 = path.join(h.tmpRoot, 'ov-tp8-model');
  const geo8 = {heads: 8, nope: 16, rope: 8, vdim: 16, qlora: 32, lat: 32, ih: 2, id: 16, topk: 4, window: 8, rank: 2, theta: 160000};
  const hgeo8 = {heads: 8, D: 16, rope: 8, qlora: 32, groups: 8, rank: 8, vrank: 2, theta: 160000, period: 4, window: 8, yarn: {factor: 16, bf: 32, bs: 1, orig: 65536}};
  const over8 = {vocab_size: 16, hidden_size: 32, intermediate_size: 64, moe_intermediate_size: 64, moe_latent_size: 32,
                 n_routed_experts: 4, num_experts_per_tok: 2, n_shared_experts: 1, hc_mult: 2};
  const realistic = ts => ts.map(t => (t.dtype === 'BF16' && t.shape.length === 2 && !t.name.includes('embed') && !t.name.includes('vha'))
    ? {...t, data: scaleBuf(t.data, 1.0 / Math.sqrt(t.shape[1]))} : t);
  fx.build(dir8, fx.config(['M', 'H', 'M', 'H'], 1, over8, geo8, hgeo8), {seed: 108, geo: geo8, h: hgeo8, mutate: realistic});
  const conv = h.convert(dir8, {keys: ['ernie55_quant_forward=1', 'ernie55_attention_block=8', 'ernie55_hca_period=4']});
  assert.equal(conv.status, 0, conv.stderr);
  model8Path = conv.out;
  return model8Path;
}

// A one-chunk prompt (28 rows: pieces of 14), a two-chunk prompt (32 + 20 rows: the second chunk
// pieced at depth 32, crossing the HCA compression and window boundaries), and both at once (one
// prefill step of two sequences, when the server batches them so; reported, not asserted).
const P1 = [1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 0, 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
const P2 = [...P1, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5, 0, 2, 8, 8];
const ON = {RADIANCE_ERNIE55_AR_OVERLAP: '1', RADIANCE_ERNIE55_AR_OVERLAP_MIN: '16'};
// Integration round 5: OV_KVDT=fp8 serves with --kv-cache-dtype fp8 (the quantised latent store path of the pieces), OV_TILE=N converts with ernie55_idx_tile=N
// (the tiled indexer top-k path). Both need an on == off identity too (the serial step's own branches, mirrored in ov_segment).
// OV_CONTROL=1: both runs serial (the host backend's own run-to-run determinism, a control).
const CONTROL = process.env.OV_CONTROL === '1';
const OFF = {RADIANCE_ERNIE55_AR_OVERLAP: undefined, RADIANCE_ERNIE55_AR_OVERLAP_MIN: undefined};

async function runAll(srv) {
  const one = await h.complete(srv, P1, {max_tokens: 4, logprobs: 20});
  const two = await h.complete(srv, P2, {max_tokens: 4, logprobs: 20});
  const both = await Promise.all([h.complete(srv, P1, {max_tokens: 4, logprobs: 20}),
                                  h.complete(srv, P2, {max_tokens: 4, logprobs: 20})]);
  return [one, two, ...both].map(r => {
    assert.equal(r.status, 200, JSON.stringify(r.body));
    return r.body.choices[0].logprobs.content.map(x => ({id: x.id, lp: x.logprob,
      top: x.top_logprobs.map(t => [t.id, t.logprob]).sort((a, b) => a[0] - b[0])}));
  });
}

for (const tp of [2, 4, 8]) {
  test(`collectives overlap at TP${tp}: tokens and every logprob identical to the serial step`, {timeout: 900000}, async () => {
    const rad = tp === 8 ? model8() : model();
    const off = await serveTp(rad, tp, OFF);
    assert.ok(!off.failed, off.log.slice(-3000));
    let a, b, onLog;
    try { a = await runAll(off); } finally { off.stop(); }
    assert.doesNotMatch(off.log, /runs as two pieces/);
    const on = await serveTp(rad, tp, CONTROL ? OFF : ON);
    assert.ok(!on.failed, on.log.slice(-3000));
    try { b = await runAll(on); onLog = on.log; } finally { on.stop(); }
    if (CONTROL) { assert.deepEqual(b, a); return; }
    assert.match(onLog, /collectives overlap on/);
    assert.match(onLog, /prefill step runs as two pieces/, 'the pieced path must have run: ' + onLog.slice(-3000));
    assert.doesNotMatch(onLog, /overlap off for this program/);
    console.log(`TP${tp} tokens: ${a.map(r => r.map(x => x.id).join(',')).join(' | ')}`);
    // The prompts must reach the output, or identity says nothing about the prefill.
    assert.notDeepEqual(a[0][0].top, a[1][0].top, 'P1 and P2 give the same first logprobs: the fixture is degenerate');
    console.log(`TP${tp} per request (P1, P2, P1+P2 together): ${a.map((r, i) => JSON.stringify(r) === JSON.stringify(b[i]) ? 'same' : 'DIFF').join(' ')}`);
    // The two sequential requests must match exactly. The concurrent pair is reported, not
    // asserted: whether the server batches the two requests into one step depends on arrival
    // timing, and the serial target itself is batch-shape dependent on the host
    // (kv-context/batch-invariance-analysis.md), so its runs need not repeat even with the switch off.
    assert.deepEqual(b.slice(0, 2), a.slice(0, 2));
  });
}

test('collectives overlap at TP1: the switch turns itself off and changes nothing', {timeout: 600000}, async () => {
  const rad = model();
  const off = await serveTp(rad, 1, OFF);
  assert.ok(!off.failed, off.log.slice(-3000));
  let a, b, onLog;
  try { a = await runAll(off); } finally { off.stop(); }
  const on = await serveTp(rad, 1, ON);
  assert.ok(!on.failed, on.log.slice(-3000));
  try { b = await runAll(on); onLog = on.log; } finally { on.stop(); }
  assert.match(onLog, /collectives overlap requested but off/);
  assert.doesNotMatch(onLog, /runs as two pieces/);
  assert.deepEqual(b.slice(0, 2), a.slice(0, 2));
});
