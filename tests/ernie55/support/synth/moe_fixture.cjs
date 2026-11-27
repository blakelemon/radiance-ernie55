'use strict';
// Tiny ERNIE 5.5 checkpoints with REAL numbers for the 2-8 routed latent-MoE program. Built on the 2-7
// feed-forward fixture (every dense and shared projection, mHC mixer, gain and embedding row has its own
// seeded values), plus, for every routed layer, both router matrices, the correction bias and the learned
// per-expert scale (F32 values that are NOT exactly BF16, so a BF16 read of either is visible), the two
// latent projections, the latent norm and every expert, each expert at its own magnitude (so normalising
// each expert before the combine gives a different answer from normalising the combine). Names, dtypes
// and shapes are the release's (fixture.cjs: shard headers of baidu/ERNIE-5.5-Flash-Base 028544b2) and
// moe.py (b8414da): w1/w3 [F, Lt], w2 [Lt, F], fc1 [Lt, H], fc2 [H, Lt], gate.weight{,_1} [E, H].
// Written down here, never read from the plugin.
const fs = require('node:fs');
const path = require('node:path');
const base = require('./fixture25.cjs');
const routed = require('./fixture.cjs');
const {bf16} = require('./oracle.cjs');

function lcg(seed) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }

// Hidden 8, hc 3, dense width 12, shared width 5 * 2 = 10, latent 4, expert width 5, six routed experts of
// which three are chosen per token (E, k, Lt, F, H, the shared width and 2E all differ). The config's
// numbers are deliberately not the release's.
function config(over = {}) {
  return {...base.config({n_shared_experts: 2, moe_intermediate_size: 5, n_routed_experts: 6, num_experts_per_tok: 3}),
    num_attention_heads: 4, num_key_value_heads: 4, hc_mult: 3, rms_norm_eps: 0.01, hc_eps: 0.02, hc_sinkhorn_iters: 4,
    activation_situ_beta: 3.0, activation_situ_linear_beta: 5.0, norm_topk_prob: true, moe_split_feature_routing: true,
    routed_scaling_factor_learnable: true, n_group: 1, topk_group: 1, scoring_func: 'sigmoid', ...over};
}

function bf16bits(x) { const v = new DataView(new ArrayBuffer(4)); v.setFloat32(0, x); return v.getUint32(0) >>> 16; }
function bf16Bytes(values) { const b = Buffer.alloc(2 * values.length); values.forEach((x, i) => b.writeUInt16LE(bf16bits(x), 2 * i)); return b; }
function f32Bytes(values) { const b = Buffer.alloc(4 * values.length); values.forEach((x, i) => b.writeFloatLE(x, 4 * i)); return b; }

// Returns {dir, ck (the oracle's model), tensors, byName}. opts.mutate may change the written tensors.
function build(dir, cfg = config(), {seed = 1, mutate = x => x} = {}) {
  fs.mkdirSync(dir, {recursive: true});
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(cfg));
  fs.writeFileSync(path.join(dir, 'tokenizer.json'), base.tokenizerJson());
  const rnd = lcg(seed * 7919 + 13);
  const uni = (lo, hi) => lo + (hi - lo) * rnd();
  const H = cfg.hidden_size, hc = cfg.hc_mult, V = cfg.vocab_size, L = cfg.num_hidden_layers;
  const E = cfg.n_routed_experts, k = cfg.num_experts_per_tok, Lt = cfg.moe_latent_size, F = cfg.moe_intermediate_size;
  const mix = hc * (hc + 2), W = hc * H;
  const spec = routed.spec(cfg);
  const tensors = [], byName = {}, vals = {};
  let sentinel = 1;
  for (const e of spec) {
    if (e.kind === 'concat3') continue;
    const n = base.numel(e.shape);
    let data, v;
    const nm = e.src;
    const pick = (lo, hi, mul = 1) => { v = Array.from({length: n}, () => mul * uni(lo, hi)); };
    const ex = /experts\.(\d+)\.w(\d)\.weight$/.exec(nm);
    if (/embed_tokens\.weight$/.test(nm)) pick(-1, 1);
    else if (ex) pick(...({1: [-.8, .8], 3: [-1.4, 1.4], 2: [-.6, .6]})[ex[2]], 0.35 + 0.4 * Number(ex[1]));
    else if (/block_sparse_moe\.gate\.weight(_1)?$/.test(nm)) pick(-.9, .9);
    else if (/e_score_correction_bias$/.test(nm)) { pick(-.35, .35); v = v.map(Math.fround); }
    else if (/routed_scaling_factor_param$/.test(nm)) { pick(.6, 1.8); v = v.map(Math.fround); }
    else if (/fc1_latent_proj\.weight$/.test(nm)) pick(-.8, .8);
    else if (/fc2_latent_proj\.weight$/.test(nm)) pick(-.6, .6);
    else if (/latent_norm\.weight$/.test(nm)) pick(.5, 1.5);
    else if (/(gate_proj|\.w1)\.weight$/.test(nm) && /mlp|shared_experts/.test(nm)) pick(-.7, .7);
    else if (/(up_proj|\.w3)\.weight$/.test(nm) && /mlp|shared_experts/.test(nm)) pick(-1.6, 1.6);
    else if (/(down_proj|\.w2)\.weight$/.test(nm) && /mlp|shared_experts/.test(nm)) pick(-.5, .5);
    else if (/(post_attention_layernorm|input_layernorm)\.weight$/.test(nm)) pick(.5, 1.5);
    else if (/mapping_proj\.weight$/.test(nm)) pick(-.35, .35);
    else if (/hyper_connection\.bias$/.test(nm)) pick(-1.2, 1.2);
    else if (/alpha_pre$/.test(nm)) v = [uni(.5, 1.3)];
    else if (/alpha_post$/.test(nm)) v = [uni(-1.2, -.3)];
    else if (/alpha_res$/.test(nm)) v = [uni(.4, 1.2)];
    if (v) {
      if (e.dtype === 'BF16') { v = v.map(bf16); data = bf16Bytes(v); } else data = f32Bytes(v);
    } else data = base.bytesOf(e.dtype, sentinel++, n);
    const t = {name: nm, dtype: e.dtype, shape: e.shape, data};
    tensors.push(t); byName[nm] = t; vals[nm] = v;
  }
  const final = mutate(tensors);
  base.writeSafetensors(path.join(dir, 'model.safetensors'), final);

  const rows = (name, r, c) => { const v = vals[name]; return Array.from({length: r}, (_, i) => v.slice(i * c, (i + 1) * c)); };
  const layers = [];
  for (let l = 0; l < L; ++l) {
    const p = `model.layers.${l}`;
    const dense = l < cfg.first_k_dense_replace;
    const I = dense ? cfg.intermediate_size : cfg.moe_intermediate_size * cfg.n_shared_experts;
    const s = dense ? `${p}.mlp` : `${p}.block_sparse_moe.shared_experts`;
    const names = dense ? ['gate_proj', 'up_proj', 'down_proj'] : ['w1', 'w3', 'w2'];
    const st = rows(`${p}.mlp_hyper_connection.mapping_proj.weight`, W, mix);
    const layer = {
      kind: dense ? 'dense' : 'routed', I,
      fn: Array.from({length: mix}, (_, j) => st.map(r => r[j])),
      scale: ['pre', 'post', 'res'].map(a => vals[`${p}.mlp_hyper_connection.alpha_${a}`][0]),
      base: vals[`${p}.mlp_hyper_connection.bias`],
      gain: vals[`${p}.post_attention_layernorm.weight`],
      gate: rows(`${s}.${names[0]}.weight`, I, H), up: rows(`${s}.${names[1]}.weight`, I, H),
      down: rows(`${s}.${names[2]}.weight`, H, I),
    };
    if (!dense) {
      const m = `${p}.block_sparse_moe`;
      layer.moe = {E, k, renorm: !!cfg.norm_topk_prob,
        r0: rows(`${m}.gate.weight`, E, H), r1: rows(`${m}.gate.weight_1`, E, H),
        bias: vals[`${m}.e_score_correction_bias`], scale: vals[`${m}.gate.routed_scaling_factor_param`],
        fc1: rows(`${m}.fc1_latent_proj.weight`, Lt, H), fc2: rows(`${m}.fc2_latent_proj.weight`, H, Lt),
        lnorm: vals[`${m}.latent_norm.weight`],
        exps: Array.from({length: E}, (_, e) => ({w1: rows(`${m}.experts.${e}.w1.weight`, F, Lt),
          w3: rows(`${m}.experts.${e}.w3.weight`, F, Lt), w2: rows(`${m}.experts.${e}.w2.weight`, Lt, F)}))};
    }
    layers.push(layer);
  }
  const ck = {H, hc, V, iters: cfg.hc_sinkhorn_iters, rmsEps: cfg.rms_norm_eps, hcEps: cfg.hc_eps,
    beta: cfg.activation_situ_beta, lbeta: cfg.activation_situ_linear_beta, latent: Lt,
    emb: rows('model.embed_tokens.weight', V, H), layers};
  return {dir, ck, tensors: final, byName};
}

module.exports = {config, build, lcg};
