'use strict';
// Tiny all-MLA-DSA ERNIE 5.5 checkpoints with real seeded numbers for the 3-2b attention program: the 2-9 head fixture (trunk,
// ffn, routed layers, head) plus, for every layer, the attention tensors of the release (names and layouts from the pinned
// checkpoint index and shard headers of baidu/ERNIE-5.5-Flash-Base 028544b2, layer 7: q_a_proj [1024,4096], q_a_layernorm
// [1024], q_b_proj [16384,1024], kv_a_proj_with_mqa [576,4096], kv_a_layernorm [512], kv_b_proj [28672,512], gate_proj
// [16384,1024], o_proj [4096,16384], vha_postmix_U/V [64,16], core_attention.softmax_offset [64], indexer.wq_b [8192,1024],
// wk [128,4096], weights_proj [64,4096], k_norm weight and bias [128], all BF16 once dequantised). Written down here from the
// pinned sources, never read from the plugin. The geometry is deliberately not the release's.
const head = require('./head_fixture.cjs');
const moe = require('./moe_fixture.cjs');
const base = require('./fixture25.cjs');
const {bf16} = require('./oracle.cjs');
const I = require('./idx_oracle.cjs');

// heads 2, nope 4, rope 4, v 4, latent 6 (row 10), q rank 6, indexer 2 x 8 (rope 4), top-k 3, window 4, VHA rank 2.
const GEO = {heads: 2, qlora: 6, lat: 6, nope: 4, rope: 4, vdim: 4, ih: 4, id: 8, topk: 3, window: 4, theta: 10000, rank: 2};
function config(layers, first, over = {}, geo = GEO) {
  return head.config({num_hidden_layers: layers, first_k_dense_replace: first, compress_ratios: Array.from({length: layers + 1}, () => -2),
    hybrid_mla_num_attention_heads: geo.heads, hybrid_mla_q_lora_rank: geo.qlora, hybrid_mla_kv_lora_rank: geo.lat,
    hybrid_mla_qk_nope_head_dim: geo.nope, hybrid_mla_qk_rope_head_dim: geo.rope, hybrid_mla_v_head_dim: geo.vdim,
    index_n_heads: geo.ih, index_head_dim: geo.id, index_topk: geo.topk, sliding_window: geo.window, rope_theta: geo.theta, ...over});
}
function bf16bits(x) { const v = new DataView(new ArrayBuffer(4)); v.setFloat32(0, x); return v.getUint32(0) >>> 16; }
const bf16Bytes = vals => { const b = Buffer.alloc(2 * vals.length); vals.forEach((x, i) => b.writeUInt16LE(bf16bits(x), 2 * i)); return b; };
const decode = t => {
  const out = [];
  if (t.dtype === 'F32') for (let i = 0; i < t.data.length; i += 4) out.push(t.data.readFloatLE(i));
  else for (let i = 0; i < t.data.length; i += 2) { const v = new DataView(new ArrayBuffer(4)); v.setUint32(0, t.data.readUInt16LE(i) << 16); out.push(v.getFloat32(0)); }
  return out;
};
const rowsOf = (v, r, c) => Array.from({length: r}, (_, i) => v.slice(i * c, (i + 1) * c));

// opts: {seed, geo, mutate(tensors), scale: {name-regex: multiplier}, fp8: set of projection suffixes stored as block-fp8}
function build(dir, cfg, {seed = 1, geo = GEO, mutate = x => x, mul = {}, fp8 = []} = {}) {
  const rnd = moe.lcg(seed * 6151 + 17), uni = (lo, hi) => lo + (hi - lo) * rnd();
  const H = cfg.hidden_size, hc = cfg.hc_mult, mixn = hc * (hc + 2), W = hc * H;
  const qw = geo.nope + geo.rope, cw = geo.lat + geo.rope;
  const spec = [   // [suffix, shape, lo, hi]
    ['q_a_proj.weight', [geo.qlora, H], -.7, .7], ['q_a_layernorm.weight', [geo.qlora], .5, 1.5],
    ['q_b_proj.weight', [geo.heads * qw, geo.qlora], -.8, .8], ['kv_a_proj_with_mqa.weight', [cw, H], -.8, .8],
    ['kv_a_layernorm.weight', [geo.lat], .5, 1.5], ['kv_b_proj.weight', [geo.heads * (geo.nope + geo.vdim), geo.lat], -.8, .8],
    ['gate_proj.weight', [geo.heads * geo.vdim, geo.qlora], -1, 1], ['o_proj.weight', [H, geo.heads * geo.vdim], -.6, .6],
    ['vha_postmix_U', [geo.heads, geo.rank], -.9, .9], ['vha_postmix_V', [geo.heads, geo.rank], -.9, .9],
    ['core_attention.softmax_offset', [geo.heads], -1, 1],
    ['core_attention.indexer.wq_b.weight', [geo.ih * geo.id, geo.qlora], -1, 1], ['core_attention.indexer.wk.weight', [geo.id, H], -1, 1],
    ['core_attention.indexer.weights_proj.weight', [geo.ih, H], -1, 1], ['core_attention.indexer.k_norm.weight', [geo.id], .5, 1.5],
    ['core_attention.indexer.k_norm.bias', [geo.id], -.3, .3]];
  const vals = {};
  const addAtt = tensors => {
    const out = tensors.slice();
    for (let l = 0; l < cfg.num_hidden_layers; l++) {
      for (const [suf, shape, lo, hi] of spec) {
        const name = `model.layers.${l}.self_attn.${suf}`;
        const n = shape.reduce((a, b) => a * b, 1);
        const m = Object.entries(mul).find(([re]) => new RegExp(re).test(name));
        const raw = Array.from({length: n}, () => (m ? m[1] : 1) * uni(lo, hi));
        if (fp8.includes(suf)) {
          // block-fp8 as the release stores it: E4M3 codes and one fp32 scale_inv a 128x128 block; the value is code * scale
          const [R, C] = shape, br = R / 128, bc = C / 128;
          const scales = Array.from({length: br * bc}, (_, k) => 2 ** -(1 + (k % 3)));
          const codes = Buffer.alloc(n), dq = new Array(n);
          for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
            const sc = scales[Math.floor(r / 128) * bc + Math.floor(c / 128)], code = I.e4enc(raw[r * C + c] / sc);
            codes[r * C + c] = code; dq[r * C + c] = bf16(I.e4dec(code) * sc);
          }
          vals[name] = dq;
          out.push({name, dtype: 'F8_E4M3', shape, data: codes});
          const sb = Buffer.alloc(4 * scales.length); scales.forEach((x, k) => sb.writeFloatLE(x, 4 * k));
          out.push({name: name.replace(/\.weight$/, '.weight_scale_inv'), dtype: 'F32', shape: [br, bc], data: sb});
          continue;
        }
        vals[name] = raw.map(bf16);
        out.push({name, dtype: 'BF16', shape, data: bf16Bytes(vals[name])});
      }
    }
    // the attn-branch mHC and input norm were seeded by the 2-8 fixture; read them back from the tensors it wrote
    for (let l = 0; l < cfg.num_hidden_layers; l++) {
      const p = `model.layers.${l}`, get = nm => decode(out.find(t => t.name === nm));
      const st = rowsOf(get(`${p}.self_attention_hyper_connection.mapping_proj.weight`), W, mixn);
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
    layers.push({
      hc: vals[`model.layers.${l}.attn_hc`],
      qa: rowsOf(v('q_a_proj.weight'), geo.qlora, H), qaN: v('q_a_layernorm.weight'), qb: rowsOf(v('q_b_proj.weight'), geo.heads * qw, geo.qlora),
      kva: rowsOf(v('kv_a_proj_with_mqa.weight'), cw, H), kvN: v('kv_a_layernorm.weight'),
      kvb: rowsOf(v('kv_b_proj.weight'), geo.heads * (geo.nope + geo.vdim), geo.lat),
      g: rowsOf(v('gate_proj.weight'), geo.heads * geo.vdim, geo.qlora), o: rowsOf(v('o_proj.weight'), H, geo.heads * geo.vdim),
      U: rowsOf(v('vha_postmix_U'), geo.heads, geo.rank), V: rowsOf(v('vha_postmix_V'), geo.heads, geo.rank), sink: v('core_attention.softmax_offset'),
      iq: rowsOf(v('core_attention.indexer.wq_b.weight'), geo.ih * geo.id, geo.qlora), ik: rowsOf(v('core_attention.indexer.wk.weight'), geo.id, H),
      iw: rowsOf(v('core_attention.indexer.weights_proj.weight'), geo.ih, H), kg: v('core_attention.indexer.k_norm.weight'), kb: v('core_attention.indexer.k_norm.bias'),
    });
  }
  return {...m, att: {layers}, geo};
}
module.exports = {GEO, config, build, decode, head, moe, base};
