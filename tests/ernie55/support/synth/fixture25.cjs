// Verbatim copy of the sealed 2-5 oracle (dense/shared/head tensors) except its export line, which also
// exports writeSafetensors and tokenizerJson; the 2-6 fixture extends it.
// Independent oracle for 2-5: a tiny ERNIE 5.5 checkpoint with a distinct sentinel in every tensor,
// and the list of weights a converted container must hold, written down here from the pinned
// release's tensor names, shard-header shapes and the Baidu model (never read from the plugin).
const fs = require('node:fs');
const path = require('node:path');

const DIM = {hidden: 8, hc: 2, vocab: 259, dense: 12, moe: 6, shared: 1, layers: 3, draft: 1, dense_layers: 1};
const MIX = (2 + DIM.hc) * DIM.hc;       // 8 coefficients per mHC read
const WIDTH = DIM.hc * DIM.hidden;       // 16
const SH = DIM.moe * DIM.shared;         // shared FFN width

function config(over = {}) {
  return {
    model_type: 'ernie5_5_moe', architectures: ['Ernie5_5_MoeForCausalLM'],
    num_hidden_layers: DIM.layers, num_nextn_predict_layers: DIM.draft,
    compress_ratios: [128, -2, 128, 128], first_k_dense_replace: DIM.dense_layers,
    hidden_size: DIM.hidden, head_dim: 4, hc_mult: DIM.hc, vocab_size: DIM.vocab,
    intermediate_size: DIM.dense, moe_intermediate_size: DIM.moe, n_shared_experts: DIM.shared,
    n_routed_experts: 4, num_experts_per_tok: 2, moe_latent_size: 4,
    hidden_act: 'situ', tie_word_embeddings: false, ...over,
  };
}

// The required tensors: checkpoint name, stored dtype, stored shape, and what the container must
// hold under which declared name.  kind: copy | transpose | concat3.
function spec(cfg = config()) {
  const H = cfg.hidden_size, V = cfg.vocab_size, hc = cfg.hc_mult, L = cfg.num_hidden_layers;
  const mix = (2 + hc) * hc, width = hc * H;
  const t = [];
  const add = (decl, src, dtype, shape, kind = 'copy', dshape = shape) => t.push({decl, src, dtype, shape, kind, dshape});
  add('token_embd.weight', 'model.embed_tokens.weight', 'BF16', [V, H]);
  add('head_hc.fn', 'model.mhc_contract.hc_head_fn', 'F32', [width, hc], 'transpose', [hc, width]);
  add('head_hc.base', 'model.mhc_contract.hc_head_base', 'F32', [hc]);
  add('head_hc.scale', 'model.mhc_contract.hc_head_scale', 'F32', [1]);
  add('output_norm.weight', 'model.norm.weight', 'BF16', [H]);
  add('output.weight', 'lm_head.weight', 'BF16', [V, H]);
  add('multimax.ranges', 'model.lm_head.multimax_ranges', 'BF16', [4]);
  add('multimax.coefficients', 'model.lm_head.multimax_ts', 'BF16', [4]);
  for (let l = 0; l < L; ++l) {
    const p = `model.layers.${l}`;
    add(`blk.${l}.attn_norm.weight`, `${p}.input_layernorm.weight`, 'BF16', [H]);
    add(`blk.${l}.ffn_norm.weight`, `${p}.post_attention_layernorm.weight`, 'BF16', [H]);
    for (const [d, c] of [['attn_hc', 'self_attention_hyper_connection'], ['ffn_hc', 'mlp_hyper_connection']]) {
      add(`blk.${l}.${d}.fn`, `${p}.${c}.mapping_proj.weight`, 'F32', [width, mix], 'transpose', [mix, width]);
      add(`blk.${l}.${d}.base`, `${p}.${c}.bias`, 'F32', [mix]);
      t.push({decl: `blk.${l}.${d}.scale`, kind: 'concat3', dtype: 'F32', dshape: [3],
        srcs: ['pre', 'post', 'res'].map(a => `${p}.${c}.alpha_${a}`)});
      for (const a of ['pre', 'post', 'res']) t.push({src: `${p}.${c}.alpha_${a}`, dtype: 'F32', shape: [1], kind: 'input-only'});
    }
    if (l < cfg.first_k_dense_replace) {
      const I = cfg.intermediate_size;
      add(`blk.${l}.ffn_gate.weight`, `${p}.mlp.gate_proj.weight`, 'BF16', [I, H]);
      add(`blk.${l}.ffn_up.weight`, `${p}.mlp.up_proj.weight`, 'BF16', [I, H]);
      add(`blk.${l}.ffn_down.weight`, `${p}.mlp.down_proj.weight`, 'BF16', [H, I]);
    } else {
      const I = cfg.moe_intermediate_size * cfg.n_shared_experts, s = `${p}.block_sparse_moe.shared_experts`;
      add(`blk.${l}.ffn_gate_shexp.weight`, `${s}.w1.weight`, 'BF16', [I, H]);
      add(`blk.${l}.ffn_up_shexp.weight`, `${s}.w3.weight`, 'BF16', [I, H]);
      add(`blk.${l}.ffn_down_shexp.weight`, `${s}.w2.weight`, 'BF16', [H, I]);
    }
  }
  return t;
}

const numel = shape => shape.reduce((a, b) => a * b, 1);

// A distinct, finite value for element i of tensor number id.
function bf16(id, i) { return 0x3c00 + ((id * 977 + i * 131) % 0x300); }
function f32(id, i) { return id + (i + 1) / 64; }

function bytesOf(dtype, id, n) {
  if (dtype === 'BF16') { const b = Buffer.alloc(2 * n); for (let i = 0; i < n; ++i) b.writeUInt16LE(bf16(id, i), 2 * i); return b; }
  const b = Buffer.alloc(4 * n); for (let i = 0; i < n; ++i) b.writeFloatLE(f32(id, i), 4 * i); return b;
}

// tensors: [{name, dtype, shape, data}] (safetensors dtype names), written as one shard.
function writeSafetensors(file, tensors) {
  const header = {}; let off = 0;
  for (const t of tensors) { header[t.name] = {dtype: t.dtype, shape: t.shape, data_offsets: [off, off + t.data.length]}; off += t.data.length; }
  const h = Buffer.from(JSON.stringify(header));
  const pad = Buffer.alloc((8 - h.length % 8) % 8, 0x20);
  const len = Buffer.alloc(8); len.writeBigUInt64LE(BigInt(h.length + pad.length));
  fs.writeFileSync(file, Buffer.concat([len, h, pad, ...tensors.map(t => t.data)]));
}

// A byte-level BPE tokenizer.json with no merges: 256 byte tokens, then three specials (the
// converter needs a tokenizer beside a safetensors checkpoint; its content is not under test).
function tokenizerJson() {
  const keep = []; for (let b = 33; b <= 126; ++b) keep.push(b); for (let b = 161; b <= 172; ++b) keep.push(b); for (let b = 174; b <= 255; ++b) keep.push(b);
  const sym = {}; let n = 0;
  for (let b = 0; b < 256; ++b) sym[b] = String.fromCodePoint(keep.includes(b) ? b : 256 + n++);
  const vocab = {}; for (let b = 0; b < 256; ++b) vocab[sym[b]] = b;
  const sp = ['<|endoftext|>', '<|im_start|>', '<|im_end|>'];
  sp.forEach((c, i) => { vocab[c] = 256 + i; });
  const bl = {type: 'ByteLevel', add_prefix_space: false, trim_offsets: false, use_regex: true};
  return JSON.stringify({version: '1.0', added_tokens: sp.map((c, i) => ({id: 256 + i, content: c, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true})),
    normalizer: null, pre_tokenizer: bl, post_processor: null, decoder: bl,
    model: {type: 'BPE', dropout: null, unk_token: null, byte_fallback: false, vocab, merges: []}});
}

// Build a checkpoint directory; every stored tensor gets its own id, so its values are its own.
// mutate(tensors) may drop, rename, reshape or retype tensors. Returns {tensors, id} where id maps
// a checkpoint name to its sentinel id.
function checkpoint(dir, cfg = config(), mutate = x => x) {
  fs.mkdirSync(dir, {recursive: true});
  fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(cfg));
  fs.writeFileSync(path.join(dir, 'tokenizer.json'), tokenizerJson());
  const id = {}; let next = 1; const tensors = [];
  for (const e of spec(cfg)) {
    if (e.kind === 'concat3') continue;
    id[e.src] = next++;
    tensors.push({name: e.src, dtype: e.dtype, shape: e.shape, data: bytesOf(e.dtype, id[e.src], numel(e.shape))});
  }
  const final = mutate(tensors);
  writeSafetensors(path.join(dir, 'model.safetensors'), final);
  return {tensors: final, id};
}

module.exports = {DIM, config, spec, numel, checkpoint, bytesOf, writeSafetensors, tokenizerJson};
