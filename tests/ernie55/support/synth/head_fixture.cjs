'use strict';
// Tiny ERNIE 5.5 checkpoints for the 2-9 target-output program: the 2-8 routed fixture (every trunk tensor
// has seeded real values) plus real values for the head tensors the earlier fixtures left as sentinels:
// the learned collapse (mhc_contract.hc_head_{fn,base,scale}), the final norm, lm_head and the MultiMax
// ranges and coefficients. Shapes, dtypes and names are the release's (fixture25 spec: shard headers of
// baidu/ERNIE-5.5-Flash-Base 028544b2; hc_head_fn is stored [hc*H, hc] (in, out) F32, multimax_ranges and
// multimax_ts [4] BF16). Written down here, never read from the plugin.
const fs = require('node:fs');
const path = require('node:path');
const base = require('./fixture25.cjs');
const moe = require('./moe_fixture.cjs');
const {bf16} = require('./oracle.cjs');

// MultiMax values are BF16-exact (they are stored BF16): r = [lower linear, upper linear, lower quad, upper quad].
const MM = {r: [-0.75, 0.75, -1.75, 1.75], t: [0.625, -0.375, 0.4375, -0.5625]};
// Vocabulary 19: the sampler's top-20 log-probabilities then cover the whole vocabulary.
const config = (over = {}) => moe.config({vocab_size: 19, ...over});

function bf16bits(x) { const v = new DataView(new ArrayBuffer(4)); v.setFloat32(0, x); return v.getUint32(0) >>> 16; }
const bf16Bytes = vals => { const b = Buffer.alloc(2 * vals.length); vals.forEach((x, i) => b.writeUInt16LE(bf16bits(x), 2 * i)); return b; };
const f32Bytes = vals => { const b = Buffer.alloc(4 * vals.length); vals.forEach((x, i) => b.writeFloatLE(x, 4 * i)); return b; };

// Returns {dir, ck (trunk oracle model), head, tensors, byName}. opts.mm overrides the MultiMax {r, t};
// opts.mutate may change the written tensors afterwards.
function build(dir, cfg = config(), {seed = 1, mm = MM, mutate = x => x, headSeed = seed, lmScale = 1} = {}) {
  const H = cfg.hidden_size, hc = cfg.hc_mult, V = cfg.vocab_size, W = hc * H;
  const rnd = moe.lcg(headSeed * 104729 + 7);
  const uni = (lo, hi) => lo + (hi - lo) * rnd();
  const stored = {
    'model.mhc_contract.hc_head_fn': {dtype: 'F32', shape: [W, hc], v: Array.from({length: W * hc}, () => Math.fround(uni(-.5, .5)))},
    'model.mhc_contract.hc_head_base': {dtype: 'F32', shape: [hc], v: Array.from({length: hc}, () => Math.fround(uni(-1, 1)))},
    'model.mhc_contract.hc_head_scale': {dtype: 'F32', shape: [1], v: [Math.fround(uni(.6, 1.4))]},
    'model.norm.weight': {dtype: 'BF16', shape: [H], v: Array.from({length: H}, () => bf16(uni(.5, 1.5)))},
    'lm_head.weight': {dtype: 'BF16', shape: [V, H], v: Array.from({length: V * H}, () => bf16(lmScale * uni(-1, 1)))},
    'model.lm_head.multimax_ranges': {dtype: 'BF16', shape: [4], v: mm.r},
    'model.lm_head.multimax_ts': {dtype: 'BF16', shape: [4], v: mm.t},
  };
  const written = {};   // the head tensors' bytes as written
  const put = t => {
    if (!stored[t.name]) return t;
    const data = (stored[t.name].dtype === 'F32' ? f32Bytes : bf16Bytes)(stored[t.name].v);
    written[t.name] = data;
    return {...t, dtype: stored[t.name].dtype, shape: stored[t.name].shape, data};
  };
  const m = moe.build(dir, cfg, {seed, mutate: t => mutate(t.map(put))});
  const fnv = stored['model.mhc_contract.hc_head_fn'].v;
  const head = {
    hc, H, V,
    fn: Array.from({length: hc}, (_, j) => Array.from({length: W}, (_, k) => fnv[k * hc + j])),   // [hc, W] (out, in)
    fnStored: fnv, base: stored['model.mhc_contract.hc_head_base'].v, scale: stored['model.mhc_contract.hc_head_scale'].v[0],
    gain: stored['model.norm.weight'].v,
    lm: Array.from({length: V}, (_, v) => stored['lm_head.weight'].v.slice(v * H, (v + 1) * H)),
    r: mm.r.slice(), t: mm.t.slice(),
  };
  return {...m, head, written};
}

module.exports = {config, build, MM, base};
