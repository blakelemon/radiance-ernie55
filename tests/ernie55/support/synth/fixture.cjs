// Independent oracle for 2-6: a tiny ERNIE 5.5 checkpoint whose routed layers carry both router
// matrices, the correction bias, the learned expert scale, the latent projections and norm and
// every routed expert, each tensor with its own sentinel, laid out across several shards in
// deliberately scrambled orders. The names, dtypes and shapes below are written down from the
// pinned release's index and shard headers (baidu/ERNIE-5.5-Flash-Base 028544b2: layer 1 gate
// [512, 4096] BF16, e_score_correction_bias and routed_scaling_factor_param [512] F32,
// fc1_latent_proj [2048, 4096], fc2_latent_proj [4096, 2048], latent_norm [2048] BF16, experts
// w1/w3 [2048, 2048], w2 [2048, 2048] with moe_latent_size = moe_intermediate_size = 2048) and the
// Baidu model (vLLM-ERNIE b8414da nvidia/moe.py: experts run at hidden=moe_latent_size,
// intermediate=moe_intermediate_size, so w1/w3 are [F, Lt] and w2 is [Lt, F]; gate.weight and
// gate.weight_1 are the two [E, H] router halves). Never read from the plugin.
const fs = require('node:fs');
const path = require('node:path');
const base = require('./fixture25.cjs');
const {numel, bytesOf, writeSafetensors, tokenizerJson} = base;

// Five routed experts (not the four of 2-5) so no extent coincides with the latent width of 4.
const config = (over = {}) => base.config({n_routed_experts: 5, ...over});

// Routed layers are first_k_dense_replace .. num_hidden_layers-1; the draft layer (index
// num_hidden_layers) is a decoy that no declaration reads.
function routedSpec(cfg = config()) {
  const H = cfg.hidden_size, E = cfg.n_routed_experts, Lt = cfg.moe_latent_size, F = cfg.moe_intermediate_size;
  const t = [];
  const add = (decl, src, dtype, shape, extra = {}) => t.push({decl, src, dtype, shape, dshape: shape, kind: 'copy', ...extra});
  for (let l = cfg.first_k_dense_replace; l < cfg.num_hidden_layers; ++l) {
    const m = `model.layers.${l}.block_sparse_moe`;
    add(`blk.${l}.ffn_gate_inp.weight`, `${m}.gate.weight`, 'BF16', [E, H], {layer: l, role: 'router0'});
    add(`blk.${l}.ffn_gate_inp_1.weight`, `${m}.gate.weight_1`, 'BF16', [E, H], {layer: l, role: 'router1'});
    add(`blk.${l}.exp_probs_b.bias`, `${m}.e_score_correction_bias`, 'F32', [E], {layer: l, role: 'bias'});
    add(`blk.${l}.ffn_gate_inp.scale`, `${m}.gate.routed_scaling_factor_param`, 'F32', [E], {layer: l, role: 'scale'});
    add(`blk.${l}.ffn_latent_in.weight`, `${m}.fc1_latent_proj.weight`, 'BF16', [Lt, H], {layer: l, role: 'latent_in'});
    add(`blk.${l}.ffn_latent_out.weight`, `${m}.fc2_latent_proj.weight`, 'BF16', [H, Lt], {layer: l, role: 'latent_out'});
    add(`blk.${l}.ffn_latent_norm.weight`, `${m}.latent_norm.weight`, 'BF16', [Lt], {layer: l, role: 'latent_norm'});
    for (let e = 0; e < E; ++e) {
      add(`blk.${l}.ffn_gate_exps.${e}.weight`, `${m}.experts.${e}.w1.weight`, 'BF16', [F, Lt], {layer: l, expert: e, role: 'w1'});
      add(`blk.${l}.ffn_up_exps.${e}.weight`, `${m}.experts.${e}.w3.weight`, 'BF16', [F, Lt], {layer: l, expert: e, role: 'w3'});
      add(`blk.${l}.ffn_down_exps.${e}.weight`, `${m}.experts.${e}.w2.weight`, 'BF16', [Lt, F], {layer: l, expert: e, role: 'w2'});
    }
  }
  return t;
}

// Everything the weights-only + routed conversion must hold.
const spec = (cfg = config()) => [...base.spec(cfg), ...routedSpec(cfg)];

// The draft layer's routed tensors, with shapes that are wrong on purpose: nothing may read them.
function draftDecoys(cfg) {
  const l = cfg.num_hidden_layers, m = `model.layers.${l}.block_sparse_moe`, E = cfg.n_routed_experts;
  const odd = (name, dtype, shape) => ({name, dtype, shape});
  const out = [odd(`${m}.gate.weight`, 'BF16', [E + 1, 3]), odd(`${m}.e_score_correction_bias`, 'BF16', [E + 2]),
    odd(`${m}.fc1_latent_proj.weight`, 'F32', [3, 3]), odd(`${m}.latent_norm.weight`, 'F32', [5])];
  for (let e = 0; e < E; ++e) out.push(odd(`${m}.experts.${e}.w1.weight`, 'F32', [2, 3]));
  return out.map((x, i) => ({...x, data: bytesOf(x.dtype === 'F32' ? 'F32' : 'BF16', 900 + i, numel(x.shape))}));
}

// Deterministic pseudo-random for the scrambled layouts.
function hash(s) { let a = 7; for (const ch of s) a = (a * 131 + ch.charCodeAt(0)) % 1000003; return a; }

const SHARDS = 4;
// Which shard (1..SHARDS) holds a tensor. Shard files are named model-0000K-of-00004.safetensors,
// so name order is shard order; every layout puts experts where enumeration order and logical
// order disagree.
function shardOf(layout, name, E) {
  const m = /experts\.(\d+)\.w(\d)/.exec(name), l = /layers\.(\d+)\./.exec(name);
  if (layout === 'ascending') return m ? 1 + Math.floor(Number(m[1]) * SHARDS / E) : 1;
  if (layout === 'descending') return m ? SHARDS - Math.floor(Number(m[1]) * SHARDS / E) : SHARDS;
  if (layout === 'interleaved') return m ? 1 + (Number(m[1]) + Number(m[2]) + (l ? Number(l[1]) : 0)) % SHARDS : 1 + hash(name) % SHARDS;
  return 1 + hash(name) % SHARDS;   // scrambled
}

// Build a checkpoint directory. opts: {layout: 'single' | 'ascending' | 'descending' | 'interleaved' |
// 'scrambled', index: 'sorted' | 'reversed' (order of weight_map keys), mutate, decoys}.
// Returns {tensors (as written, after mutate), id (sentinel id by name), dir}.
function checkpoint(dir, cfg = config(), opts = {}) {
  const {layout = 'single', index = 'sorted', mutate = x => x, decoys = true} = opts;
  fs.mkdirSync(dir, {recursive: true});
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(cfg));
  fs.writeFileSync(path.join(dir, 'tokenizer.json'), tokenizerJson());
  const id = {}; let next = 1; const tensors = [];
  for (const e of spec(cfg)) {
    if (e.kind === 'concat3') continue;
    id[e.src] = next++;
    tensors.push({name: e.src, dtype: e.dtype, shape: e.shape, data: bytesOf(e.dtype, id[e.src], numel(e.shape))});
  }
  if (decoys) tensors.push(...draftDecoys(cfg));
  const final = mutate(tensors);
  if (layout === 'single') { writeSafetensors(path.join(dir, 'model.safetensors'), final); return {dir, tensors: final, id}; }
  const E = cfg.n_routed_experts, byShard = {}, map = {};
  for (const t of final) {
    // A scale plane travels with its codes.
    const home = shardOf(layout, t.name.replace(/_scale_inv$/, ''), E);
    (byShard[home] = byShard[home] || []).push(t);
  }
  for (let s = 1; s <= SHARDS; ++s) {
    const file = `model-${String(s).padStart(5, '0')}-of-${String(SHARDS).padStart(5, '0')}.safetensors`;
    const list = (byShard[s] || []).slice().sort((a, b) => hash(a.name + 'z') - hash(b.name + 'z'));
    if (!list.length) list.push({name: `pad.${s}`, dtype: 'F32', shape: [1], data: Buffer.alloc(4)});
    writeSafetensors(path.join(dir, file), list);
    for (const t of list) if (!t.name.startsWith('pad.')) map[t.name] = file;
  }
  const names = Object.keys(map).sort();
  if (index === 'reversed') names.reverse();
  const weight_map = {}; for (const n of names) weight_map[n] = map[n];
  fs.writeFileSync(path.join(dir, 'model.safetensors.index.json'), JSON.stringify({metadata: {}, weight_map}));
  return {dir, tensors: final, id};
}

module.exports = {config, spec, routedSpec, numel, checkpoint, bytesOf, SHARDS, base};
