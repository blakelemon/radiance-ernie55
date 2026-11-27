'use strict';
// Tiny ERNIE 5.5 checkpoints with real seeded numbers for the 3-4b HCA attention program: the 2-9 head fixture plus, per layer,
// the attention tensors of the layer's family. MLA-DSA layers carry the tensors of the 3-2b fixture; HCA layers carry the release's
// HCA tensors (names and layouts from the pinned checkpoint index and shard headers of baidu/ERNIE-5.5-Flash-Base 028544b2, layer 1:
// linear_q_down_proj.weight [1024,4096], q_layernorm.weight [1024], linear_q_up_proj.weight [32768,1024], linear_kv_proj.weight [512,4096],
// kv_layernorm.weight [512], gate_proj.weight [8192,1024], linear_o_group_proj [8192,4096] (no ".weight"), o_proj.weight [4096,8192],
// vha_postmix_U/V [64,16], core_attention.attn_sink F32 [64], compressor.linear_wkv.weight / linear_wgate.weight [512,4096],
// compressor.ape F32 [128,512], compressor.norm.weight [512]; all BF16 once dequantised except the F32 ones). Written down here from
// the pinned sources, never read from the plugin. The geometry is deliberately not the release's.
const head = require('./head_fixture.cjs');
const moe = require('./moe_fixture.cjs');
const base = require('./fixture25.cjs');
const f32 = require('./fixture32b.cjs');
const I = require('./idx_oracle.cjs');
const {bf16} = require('./oracle.cjs');

const GEO = f32.GEO;                       // MLA-DSA geometry of 3-2b (window 4, top-k 3)
// HCA: heads 4, head width 16 (rope 4), q rank 8, 2 groups of 2 heads, o rank 3 (wo_a output 6), VHA rank 2.
const HGEO = {heads: 4, D: 16, rope: 4, qlora: 8, groups: 2, rank: 3, vrank: 2, theta: 160000, period: 128, window: 128, yarn: {factor: 16, bf: 32, bs: 1, orig: 65536}};

function config(kinds, first, over = {}, geo = GEO, h = HGEO) {
  const layers = kinds.length;
  return head.config({num_hidden_layers: layers, first_k_dense_replace: first,
    compress_ratios: [...kinds.map(k => (k === 'H' ? 128 : -2)), -2],
    hybrid_mla_num_attention_heads: geo.heads, hybrid_mla_q_lora_rank: geo.qlora, hybrid_mla_kv_lora_rank: geo.lat,
    hybrid_mla_qk_nope_head_dim: geo.nope, hybrid_mla_qk_rope_head_dim: geo.rope, hybrid_mla_v_head_dim: geo.vdim,
    index_n_heads: geo.ih, index_head_dim: geo.id, index_topk: geo.topk, sliding_window: h.window, rope_theta: geo.theta,
    head_dim: h.D, num_attention_heads: h.heads, qk_rope_head_dim: h.rope, q_lora_rank: h.qlora, o_groups: h.groups, o_lora_rank: h.rank,
    compress_rope_theta: h.theta,
    rope_scaling: {type: 'yarn', factor: h.yarn.factor, beta_fast: h.yarn.bf, beta_slow: h.yarn.bs, original_max_position_embeddings: h.yarn.orig},
    ...over});
}
const bf16bits = x => { const v = new DataView(new ArrayBuffer(4)); v.setFloat32(0, x); return v.getUint32(0) >>> 16; };
const bf16Bytes = vals => { const b = Buffer.alloc(2 * vals.length); vals.forEach((x, i) => b.writeUInt16LE(bf16bits(x), 2 * i)); return b; };
const f32Bytes = vals => { const b = Buffer.alloc(4 * vals.length); vals.forEach((x, i) => b.writeFloatLE(x, 4 * i)); return b; };
const rowsOf = (v, r, c) => Array.from({length: r}, (_, i) => v.slice(i * c, (i + 1) * c));

// opts: {seed, geo, h, mutate(tensors), mul: {name-regex: multiplier}, drop: set of suffixes not written}
function build(dir, cfg, {seed = 1, geo = GEO, h = HGEO, mutate = x => x, mul = {}, fp8 = []} = {}) {
  const rnd = moe.lcg(seed * 6151 + 17), uni = (lo, hi) => lo + (hi - lo) * rnd();
  const H = cfg.hidden_size, hc = cfg.hc_mult, mixn = hc * (hc + 2), Wd = hc * H;
  const kinds = cfg.compress_ratios.slice(0, cfg.num_hidden_layers).map(r => (r === 128 ? 'H' : 'M'));
  const qw = geo.nope + geo.rope, cw = geo.lat + geo.rope;
  const coff = h.period === 4 ? 2 : 1, W = coff * h.D, GR = h.groups * h.rank, per = (h.heads / h.groups) * h.D;
  const mspec = [
    ['q_a_proj.weight', [geo.qlora, H], -.7, .7], ['q_a_layernorm.weight', [geo.qlora], .5, 1.5],
    ['q_b_proj.weight', [geo.heads * qw, geo.qlora], -.8, .8], ['kv_a_proj_with_mqa.weight', [cw, H], -.8, .8],
    ['kv_a_layernorm.weight', [geo.lat], .5, 1.5], ['kv_b_proj.weight', [geo.heads * (geo.nope + geo.vdim), geo.lat], -.8, .8],
    ['gate_proj.weight', [geo.heads * geo.vdim, geo.qlora], -1, 1], ['o_proj.weight', [H, geo.heads * geo.vdim], -.6, .6],
    ['vha_postmix_U', [geo.heads, geo.rank], -.9, .9], ['vha_postmix_V', [geo.heads, geo.rank], -.9, .9],
    ['core_attention.softmax_offset', [geo.heads], -1, 1],
    ['core_attention.indexer.wq_b.weight', [geo.ih * geo.id, geo.qlora], -1, 1], ['core_attention.indexer.wk.weight', [geo.id, H], -1, 1],
    ['core_attention.indexer.weights_proj.weight', [geo.ih, H], -1, 1], ['core_attention.indexer.k_norm.weight', [geo.id], .5, 1.5],
    ['core_attention.indexer.k_norm.bias', [geo.id], -.3, .3]];
  const hspec = [   // [suffix, shape, lo, hi, dtype]
    ['linear_q_down_proj.weight', [h.qlora, H], -.8, .8], ['q_layernorm.weight', [h.qlora], .5, 1.5],
    ['linear_q_up_proj.weight', [h.heads * h.D, h.qlora], -1.2, 1.2], ['linear_kv_proj.weight', [h.D, H], -1, 1],
    ['kv_layernorm.weight', [h.D], .5, 1.5], ['gate_proj.weight', [GR, h.qlora], -1, 1],
    ['linear_o_group_proj', [GR, per], -.6, .6], ['o_proj.weight', [H, GR], -.8, .8],
    ['vha_postmix_U', [h.heads, h.vrank], -.9, .9], ['vha_postmix_V', [h.heads, h.vrank], -.9, .9],
    ['core_attention.attn_sink', [h.heads], -1, 1, 'F32'],
    ['core_attention.compressor.linear_wkv.weight', [W, H], -1, 1], ['core_attention.compressor.linear_wgate.weight', [W, H], -1.5, 1.5],
    ['core_attention.compressor.ape', [h.period, W], -1, 1, 'F32'], ['core_attention.compressor.norm.weight', [h.D], .5, 1.5]];
  const vals = {};
  const addAtt = tensors => {
    const out = tensors.slice();
    for (let l = 0; l < cfg.num_hidden_layers; l++) {
      for (const [suf, shape, lo, hi, dt] of kinds[l] === 'H' ? hspec : mspec) {
        const name = `model.layers.${l}.self_attn.${suf}`;
        const n = shape.reduce((a, b) => a * b, 1);
        const m = Object.entries(mul).find(([re]) => new RegExp(re).test(name));
        const raw = Array.from({length: n}, () => (m ? m[1] : 1) * uni(lo, hi));
        if (dt === 'F32') { vals[name] = raw.map(Math.fround); out.push({name, dtype: 'F32', shape, data: f32Bytes(vals[name])}); continue; }
        if (fp8.includes(suf)) {
          // block-fp8 as the release stores it: E4M3 codes and one fp32 scale_inv a 128x128 block; the value is code * scale
          const [Rr, Cc] = shape, br = Rr / 128, bc = Cc / 128;
          const scales = Array.from({length: br * bc}, (_, k) => 2 ** -(1 + (k % 3)));
          const codes = Buffer.alloc(n), dq = new Array(n);
          for (let r = 0; r < Rr; r++) for (let c = 0; c < Cc; c++) {
            const sc = scales[Math.floor(r / 128) * bc + Math.floor(c / 128)], code = I.e4enc(raw[r * Cc + c] / sc);
            codes[r * Cc + c] = code; dq[r * Cc + c] = bf16(I.e4dec(code) * sc);
          }
          vals[name] = dq;
          out.push({name, dtype: 'F8_E4M3', shape, data: codes});
          const sb = Buffer.alloc(4 * scales.length); scales.forEach((x, k) => sb.writeFloatLE(x, 4 * k));
          out.push({name: /\.weight$/.test(name) ? name.replace(/\.weight$/, '.weight_scale_inv') : name + '_scale_inv', dtype: 'F32', shape: [br, bc], data: sb});
          continue;
        }
        vals[name] = raw.map(bf16);
        out.push({name, dtype: 'BF16', shape, data: bf16Bytes(vals[name])});
      }
    }
    for (let l = 0; l < cfg.num_hidden_layers; l++) {
      const p = `model.layers.${l}`, get = nm => f32.decode(out.find(t => t.name === nm));
      const st = rowsOf(get(`${p}.self_attention_hyper_connection.mapping_proj.weight`), Wd, mixn);
      vals[`${p}.attn_hc`] = {
        fn: Array.from({length: mixn}, (_, j) => st.map(r => r[j])),
        scale: ['pre', 'post', 'res'].map(a => get(`${p}.self_attention_hyper_connection.alpha_${a}`)[0]),
        base: get(`${p}.self_attention_hyper_connection.bias`), gain: get(`${p}.input_layernorm.weight`)};
    }
    return mutate(out);
  };
  const m = head.build(dir, cfg, {seed, mutate: addAtt});
  const layers = [];
  for (let l = 0; l < cfg.num_hidden_layers; l++) {
    const v = suf => vals[`model.layers.${l}.self_attn.${suf}`];
    if (kinds[l] === 'H') {
      layers.push({kind: 'hca', hc: vals[`model.layers.${l}.attn_hc`],
        qa: rowsOf(v('linear_q_down_proj.weight'), h.qlora, H), qaN: v('q_layernorm.weight'), qb: rowsOf(v('linear_q_up_proj.weight'), h.heads * h.D, h.qlora),
        kv: rowsOf(v('linear_kv_proj.weight'), h.D, H), kvN: v('kv_layernorm.weight'), g: rowsOf(v('gate_proj.weight'), GR, h.qlora),
        og: rowsOf(v('linear_o_group_proj'), GR, per), o: rowsOf(v('o_proj.weight'), H, GR),
        U: rowsOf(v('vha_postmix_U'), h.heads, h.vrank), V: rowsOf(v('vha_postmix_V'), h.heads, h.vrank), sink: v('core_attention.attn_sink'),
        ckv: rowsOf(v('core_attention.compressor.linear_wkv.weight'), W, H), cg: rowsOf(v('core_attention.compressor.linear_wgate.weight'), W, H),
        ape: rowsOf(v('core_attention.compressor.ape'), h.period, W), cn: v('core_attention.compressor.norm.weight')});
    } else {
      layers.push({kind: 'mla', hc: vals[`model.layers.${l}.attn_hc`],
        qa: rowsOf(v('q_a_proj.weight'), geo.qlora, H), qaN: v('q_a_layernorm.weight'), qb: rowsOf(v('q_b_proj.weight'), geo.heads * qw, geo.qlora),
        kva: rowsOf(v('kv_a_proj_with_mqa.weight'), cw, H), kvN: v('kv_a_layernorm.weight'),
        kvb: rowsOf(v('kv_b_proj.weight'), geo.heads * (geo.nope + geo.vdim), geo.lat),
        g: rowsOf(v('gate_proj.weight'), geo.heads * geo.vdim, geo.qlora), o: rowsOf(v('o_proj.weight'), H, geo.heads * geo.vdim),
        U: rowsOf(v('vha_postmix_U'), geo.heads, geo.rank), V: rowsOf(v('vha_postmix_V'), geo.heads, geo.rank), sink: v('core_attention.softmax_offset'),
        iq: rowsOf(v('core_attention.indexer.wq_b.weight'), geo.ih * geo.id, geo.qlora), ik: rowsOf(v('core_attention.indexer.wk.weight'), geo.id, H),
        iw: rowsOf(v('core_attention.indexer.weights_proj.weight'), geo.ih, H), kg: v('core_attention.indexer.k_norm.weight'), kb: v('core_attention.indexer.k_norm.bias')});
    }
  }
  return {...m, att: {layers}, geo, h, kinds};
}
module.exports = {GEO, HGEO, config, build, head, moe, base, decode: f32.decode};
