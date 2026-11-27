"use strict";
// A tiny synthetic ERNIE 5.5 checkpoint WITH the DSpark drafter (4 target layers, 3 draft layers, MXFP4 routed experts, block-fp8 projections) and the
// helpers to convert and serve it on the CPU reference kernels. Self-contained: the fixtures come from ./synth (vendored from the ERNIE lane).
// Run under the sysfs mask, with a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55).
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const h = require("./synth/harness.cjs");
const fx = require("./synth/fixture34b.cjs");
const moe = fx.moe;

const SEED = 91;
const VOCAB = 64;
const TAPS = [1, 2, 3];
const MAIN = 4, NDRAFT = 3, BLOCK = 7, RANK = 32, NOISE = 5;
const MAXLEN = 1024;

// Every sharded dim divides by 4 and stays a multiple of 128 so ALL FP8 + MXFP4 weights shard cleanly at TP=1, 2, 4.
const GEO = {heads: 4, nope: 112, rope: 16, vdim: 128, qlora: 128, lat: 64, ih: 2, id: 64, topk: 4, window: 8, rank: 2, theta: 160000};
const HGEO = {heads: 4, D: 128, rope: 16, qlora: 128, groups: 4, rank: 128, vrank: 2, theta: 160000, period: 4, window: 8, yarn: {factor: 16, bf: 32, bs: 1, orig: 65536}};
const OVER = {
  vocab_size: VOCAB,
  hidden_size: 128, intermediate_size: 512, moe_intermediate_size: 512, moe_latent_size: 128,
  n_routed_experts: 4, num_experts_per_tok: 2, n_shared_experts: 1, hc_mult: 2,
  quantization_config: {
    activation_scheme: "dynamic", fmt: "e4m3", quant_method: "fp8",
    expert_dtype: "mxfp4", scale_fmt: "ue8m0", weight_block_size: [128, 128]
  }
};

const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6];
function bf2f(u16) { const b = Buffer.alloc(4); b.writeUInt32LE((u16 << 16) >>> 0); return b.readFloatLE(0); }
// bf16 encode of a float32 bit pattern: round to nearest even; any NaN (whatever its payload) becomes a quiet NaN (0x7fc0 | sign), never rounds to inf.
function f2bfBits(u) {
  u >>>= 0;
  if ((u & 0x7fffffff) > 0x7f800000) return ((u >>> 16) & 0x8000) | 0x7fc0;
  return ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) & 0xffff;
}
function f2bf(x) { const b = Buffer.alloc(4); b.writeFloatLE(x); return f2bfBits(b.readUInt32LE(0)); }
// Asserts the encoder against half-words worked out by hand from the float32 bit patterns (not by calling f2bf): top 16 bits, rounded up iff the
// low 16 bits are above 0x8000, or equal to it with an odd top half.
function selfCheckBf16() {
  const dv = new DataView(new ArrayBuffer(4));
  const bitsOf = x => { dv.setFloat32(0, x); return dv.getUint32(0); };
  const cases = [ // value, float32 bits (by hand), expected bf16
    [1, 0x3f800000, 0x3f80], [-1, 0xbf800000, 0xbf80], [0, 0, 0], [-0, 0x80000000, 0x8000], [0.5, 0x3f000000, 0x3f00],
    [1.00390625, 0x3f808000, 0x3f80],  // tie, top half even -> stays
    [1.01171875, 0x3f818000, 0x3f82],  // tie, top half odd -> rounds up
    [3.14159, 0x40490fd0, 0x4049],     // below the tie -> down
    [65504, 0x477fe000, 0x4780],       // above the tie -> up (bf16 65536)
    [3.4e38, 0x7f7fc99e, 0x7f80],      // rounds up to inf
    [Infinity, 0x7f800000, 0x7f80], [-Infinity, 0xff800000, 0xff80], [NaN, 0x7fc00000, 0x7fc0],
  ];
  for (const [x, bits, want] of cases) {
    assert.equal(bitsOf(x), bits, `fixture: float32 bits of ${x}`);
    const got = f2bf(x);
    assert.equal(got, want, `f2bf(${x}) = 0x${got.toString(16)}, expected 0x${want.toString(16)}`);
  }
  // NaN payloads, both signs, must stay NaN (quiet), not round to inf
  for (const [bits, want] of [[0x7f800001, 0x7fc0], [0x7fbfffff, 0x7fc0], [0xff800001, 0xffc0], [0x7fffffff, 0x7fc0], [0xffffffff, 0xffc0]])
    assert.equal(f2bfBits(bits), want, `f2bfBits(0x${bits.toString(16)}) must be a quiet NaN`);
  // decode: every half-word's float round trips through the encoder (NaN canonicalised)
  for (let u = 0; u < 0x10000; u++) {
    const f = bf2f(u);
    if (Number.isNaN(f)) { assert.ok((u & 0x7fff) > 0x7f80); assert.equal(f2bf(f) & 0x7fff, 0x7fc0); continue; }
    assert.equal(f2bf(f), u, `bf2f/f2bf round trip of 0x${u.toString(16)}`);
  }
  assert.equal(bf2f(0x3f80), 1); assert.equal(bf2f(0xc000), -2); assert.equal(bf2f(0x7f80), Infinity);
}
selfCheckBf16();
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

// ---- drafter tensors (names, dtypes, shapes from declare_dspark_weights in arch/ernie55/ernie55.cpp) ----
function bf16Buf(vals) { const b = Buffer.alloc(2 * vals.length); vals.forEach((x, i) => b.writeUInt16LE(f2bf(x), 2 * i)); return b; }
function f32Buf(vals) { const b = Buffer.alloc(4 * vals.length); vals.forEach((x, i) => b.writeFloatLE(x, 4 * i)); return b; }
function dsparkTensors(seed) {
  const rnd = moe.lcg(seed * 7727 + 3), uni = (lo, hi) => lo + (hi - lo) * rnd();
  const H = OVER.hidden_size, hc = OVER.hc_mult, W = hc * H, V = VOCAB, nt = TAPS.length;
  const bf = (name, shape, lo, hi, mul = 1) => ({name, dtype: "BF16", shape, data: bf16Buf(Array.from({length: shape.reduce((a, b) => a * b, 1)}, () => mul * uni(lo, hi)))});
  const f32 = (name, shape, lo, hi) => ({name, dtype: "F32", shape, data: f32Buf(Array.from({length: shape.reduce((a, b) => a * b, 1)}, () => Math.fround(uni(lo, hi))))});
  return [
    bf("model.mtp.feature_projection.weight", [H, nt * H], -1, 1, 1 / Math.sqrt(nt * H)),
    bf("model.mtp.feature_norm.weight", [H], 0.5, 1.5),
    f32("model.mtp.mhc_contract.hc_head_fn", [W, hc], -0.5, 0.5),      // stored [hc*H, hc] (in, out), transposed at convert
    f32("model.mtp.mhc_contract.hc_head_base", [hc], -1, 1),
    f32("model.mtp.mhc_contract.hc_head_scale", [1], 0.6, 1.4),
    bf("model.mtp.norm.weight", [H], 0.5, 1.5),
    bf("model.mtp.markov_head.markov_w1.weight", [RANK, V], -1, 1, 0.5),   // stored [rank, V], transposed at convert
    bf("model.mtp.markov_head.markov_w2.weight", [V, RANK], -1, 1, 0.5),
  ];
}

function renameDraft(t) {
  const m = /^model\.layers\.(\d+)\.(.*)$/.exec(t.name);
  if (!m || Number(m[1]) < MAIN) return t;
  return {...t, name: `model.mtp.${Number(m[1]) - MAIN}.${m[2]}`};
}

const DSPARK_RULES = [
  "dspark.blk.*.attn_hc.fn=cast:dtype=f32", "dspark.blk.*.ffn_hc.fn=cast:dtype=f32",
  "dspark.head_hc.fn=cast:dtype=f32", "dspark.blk.*.attn_o_group.weight=cast:dtype=bf16",
  "dspark.markov_w1=cast:dtype=bf16",
];

function buildCheckpoint(name) {
  const dir = path.join(h.tmpRoot, name);
  const kinds = ["M", "H", "M", "H", "H", "H", "H"];   // 4 target layers + 3 draft layers (HCA, no compressor)
  const cfgTotal = fx.config(kinds, 1, OVER, GEO, HGEO);
  const realistic = ts => ts.map(t => (t.dtype === "BF16" && t.shape.length === 2 && !t.name.includes("embed") && !t.name.includes("vha"))
    ? {...t, data: scaleBuf(t.data, 1.0 / Math.sqrt(t.shape[1]))} : t);
  const stats = {nMx: 0, nFp8: 0};
  fx.build(dir, cfgTotal, {
    seed: SEED, geo: GEO, h: HGEO,
    mutate: ts0 => {
      const ts = ts0.map(renameDraft).filter(t => !(t.name.startsWith("model.mtp.") && t.name.includes(".core_attention.compressor.")));
      return [...realistic(ts), ...dsparkTensors(SEED)].flatMap(t => {
        if (isExpert(t.name)) {
          const q = quantizeMx(t.data, t.shape[0], t.shape[1]);
          stats.nMx++;
          return [
            {...t, dtype: "I8", shape: [t.shape[0], t.shape[1] / 2], data: q.codes},
            {name: t.name + "_scale_inv", dtype: "F8_E8M0", shape: [t.shape[0], t.shape[1] / 32], data: q.scales}
          ];
        }
        if (isFp8Trunk(t.name, t.shape)) {
          const q = quantizeFp8(t.data, t.shape[0], t.shape[1]);
          stats.nFp8++;
          return [
            {...t, dtype: "F8_E4M3", shape: t.shape, data: q.codes},
            {name: t.name + "_scale_inv", dtype: "F32", shape: [t.shape[0] / 128, t.shape[1] / 128], data: q.scales}
          ];
        }
        return [t];
      });
    }
  });
  // The real config: 4 target layers + 3 draft (num_nextn_predict_layers) layers.
  const cfgPath = path.join(dir, "config.json");
  const cfg = JSON.parse(fs.readFileSync(cfgPath, "utf8"));
  cfg.num_hidden_layers = MAIN;
  cfg.num_nextn_predict_layers = NDRAFT;
  cfg.compress_ratios = [-2, 128, -2, 128, 128, 128, 128];
  cfg.dspark_block_size = BLOCK;
  cfg.dspark_markov_rank = RANK;
  cfg.dspark_noise_token_id = NOISE;
  cfg.dspark_target_layer_ids = TAPS;
  fs.writeFileSync(cfgPath, JSON.stringify(cfg));
  return {dir, stats, cfg};
}

function convertDspark(dir, maxSpec, tag, more = []) {
  const keys = ["ernie55_quant_forward=1", "ernie55_dspark=1", "ernie55_attention_block=8", "ernie55_hca_period=4"];
  const c = h.convert(dir, {keys, rules: [...h.RULES, ...DSPARK_RULES], extra: ["--max-spec", String(maxSpec), ...more]});
  console.log(`[convert ${tag}] max-spec=${maxSpec} status=${c.status} produced=${c.produced}`);
  if (c.status !== 0) {
    const errs = (c.stderr + c.stdout).split("\n");
    console.log(`[convert ${tag}] FAILED; ${errs.length} output lines; first 60:`);
    console.log(errs.slice(0, 60).join("\n"));
  }
  return c;
}



// `prefix` serves with the prefix cache and window checkpoints instead of --no-prefix-cache (the
// slots and interval kv_ckpt.cjs's own checkpointed server uses). Default false: every existing
// caller's yaml is unchanged. `ckptInterval`/`ckptSlots` override the checkpoint grid (default 8/8,
// the resolved-chunk-is-8 geometry); a larger interval than --max-num-batched-tokens puts a window
// tail snapshot strictly inside a budget chunk. The window switch is removed from the spawn env when
// the caller does not pass it, so an ambient RADIANCE_ERNIE55_DSPARK_KV_WINDOW cannot leak into an
// "unset" run; an OFF arm passes it as '0' explicitly.
function serveX(rad, nspec, xenv, {seqs = 8, tokens = 128, len = MAXLEN, tp = 1, extraYaml = [], prefix = false, ckptInterval = 8, ckptSlots = 8} = {}) {
  const b = h.ensureBuilt();
  const dir = fs.mkdtempSync(path.join(h.tmpRoot, `srv-s${nspec}-`));
  const yaml = path.join(dir, "serve.yaml");
  fs.writeFileSync(yaml, [
    "server:", "  host: 127.0.0.1", "  port: 0", `  log: ${"info"}`, `  radiance-home: ${b.home}`,
    "  kernels: libref", "  debug-accept-reference-kernels: true", "cards: all",
    "models:", "  ern:", `    model: ${rad}`, `    tp: ${tp}`, `    max-num-seqs: ${seqs}`,
    `    max-num-batched-tokens: ${tokens}`, `    max-model-len: ${len}`,
    `    num-speculative-tokens: ${nspec}`, ...extraYaml,
    ...(prefix ? [`    checkpoint-slots: ${ckptSlots}`, `    checkpoint-interval: ${ckptInterval}`] : ["    no-prefix-cache: true"]), ""
  ].join("\n"));
  const env = {...process.env, RADIANCE_DFLASH_TRACE: "1", ROCR_VISIBLE_DEVICES: "99", RADIANCE_HOST_DEVICES: String(tp), RADIANCE_HOST_VRAM_MIB: "512", OMP_NUM_THREADS: "2", OMP_WAIT_POLICY: "passive", ...xenv};
  if (!("RADIANCE_ERNIE55_DSPARK_KV_WINDOW" in xenv)) delete env.RADIANCE_ERNIE55_DSPARK_KV_WINDOW;
  const child = spawn(b.radiance, ["--config", yaml], {
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  child.stdout.on("data", d => { log += d; });
  child.stderr.on("data", d => { log += d; });
  return new Promise(resolve => {
    const timer = setTimeout(() => { child.kill("SIGKILL"); resolve({failed: true, log: log + "\n[timeout]"}); }, 180000);
    const tick = setInterval(() => {
      const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(log);
      if (m && /model ern: ready/.test(log)) {
        clearInterval(tick); clearTimeout(timer);
        resolve({port: Number(m[1]), get log() { return log; }, stop: () => { child.kill("SIGKILL"); }});
      }
    }, 50);
    child.on("exit", () => { clearInterval(tick); clearTimeout(timer); resolve({failed: true, log}); });
  });
}


// The declared graph only, from a debugGraph() result (stdout+stderr merged). Two identical runs differ in
// two ways that are not the program: (1) engine log lines (a single capital letter then a space: `I `, `W `,
// `E `, `D `) float to different positions, and plugin notes (`radiance: ...`, printed on stderr) are
// concatenated after stdout, so slicing from the banner alone does not drop them; (2) the `---- unresolved:`
// table is emitted in a run-dependent order. This keeps the `==== declared graph:` section (or the whole text
// if that banner is absent), drops the log and `radiance: ` lines, and for the `---- unresolved:` section keeps
// its header but SORTS the table lines (the section runs to the next `---- `/`==== ` header or the end), so two
// runs of the same program compare equal. Assertions that read a log line or the unsorted graph body stay on the
// raw text and are not passed through here.
function graphText(text) {
  const all = text.split('\n');
  const start = all.findIndex(l => l.startsWith('==== declared graph:'));
  const lines = (start >= 0 ? all.slice(start) : all)
    .filter(l => !/^[IWED] /.test(l) && !/^radiance: /.test(l));
  const ui = lines.findIndex(l => l.startsWith('---- unresolved:'));
  if (ui < 0) return lines.join('\n');
  let end = lines.length;
  for (let i = ui + 1; i < lines.length; i++) {
    if (/^(---- |==== )/.test(lines[i])) { end = i; break; }
  }
  const table = lines.slice(ui + 1, end).sort();
  return [...lines.slice(0, ui), lines[ui], ...table, ...lines.slice(end)].join('\n');
}

const rnd = (seed, len) => { let s = seed >>> 0; const o = []; for (let i = 0; i < len; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; o.push((s >>> 8) % VOCAB); } return o; };
async function genFull(srv, prompt, n) {
  const r = await h.complete(srv, prompt, {max_tokens: n, logprobs: 3});
  if (r.status !== 200) throw new Error("status " + r.status + " " + JSON.stringify(r.body).slice(0, 300));
  const c = r.body.choices[0].logprobs.content;
  const T = r.body.timings || {};
  return {ids: c.map(x => x.id), lp: c.map(x => x.logprob),
    top: c.map(x => (x.top_logprobs || []).map(t => t.id + "@" + t.logprob).join(",")),
    drafted: T.draft_n || 0, accepted: T.draft_n_accepted || 0};
}
module.exports = {selfCheckBf16, f2bf, bf2f, h, VOCAB, BLOCK, MAIN, NDRAFT, TAPS, GEO, HGEO, MAXLEN, DSPARK_RULES, buildCheckpoint, convertDspark, serveX, genFull, rnd, graphText};
