"use strict";
// A tiny synthetic ERNIE 5.5 checkpoint (4 layers M H M H, MXFP4 routed experts, block-fp8 projections) plus helpers to convert it and to start or declare it on the CPU
// reference kernels, for tests of what lives in the plugin's declare path (arch/ernie55/ernie55.cpp): the KV groups' block sizes and page bytes, the dtype of each group,
// the ops and parameters the program carries. Self-contained: the fixtures come from ./synth (vendored from the ERNIE lane, as for the DSpark tests).
// Run under the sysfs mask, with a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55).
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const h = require("./synth/harness.cjs");
const fx = require("./synth/fixture34b.cjs");

const SEED = 7, VOCAB = 64;
const GEO = {heads: 4, nope: 112, rope: 16, vdim: 128, qlora: 128, lat: 64, ih: 2, id: 64, topk: 4, window: 8, rank: 2, theta: 160000};
const HGEO = {heads: 4, D: 128, rope: 16, qlora: 128, groups: 4, rank: 128, vrank: 2, theta: 160000, period: 4, window: 8, yarn: {factor: 16, bf: 32, bs: 1, orig: 65536}};
const OVER = {
  vocab_size: VOCAB, hidden_size: 128, intermediate_size: 512, moe_intermediate_size: 512, moe_latent_size: 128,
  n_routed_experts: 4, num_experts_per_tok: 2, n_shared_experts: 1, hc_mult: 2,
  quantization_config: {activation_scheme: "dynamic", fmt: "e4m3", quant_method: "fp8", expert_dtype: "mxfp4", scale_fmt: "ue8m0", weight_block_size: [128, 128]}
};
// what the latent row is at this geometry: latent_dim = lat (64), rope = rope (16): a bf16 row is (lat + rope) * 2 bytes, an fp8_ds_mla row lat + 4 * (lat / 128 rounded up) + 2 * rope
const ATTN_BLOCK = 8, PERIOD = HGEO.period;

const E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6];
function bf2f(u16) { const b = Buffer.alloc(4); b.writeUInt32LE((u16 << 16) >>> 0); return b.readFloatLE(0); }
function f2bf(x) { const b = Buffer.alloc(4); b.writeFloatLE(x); let u = b.readUInt32LE(0); u = (u + 0x7fff + ((u >>> 16) & 1)) >>> 16; return u & 0xffff; }
function scaleBuf(buf, factor) { const out = Buffer.alloc(buf.length); for (let i = 0; i < buf.length; i += 2) out.writeUInt16LE(f2bf(bf2f(buf.readUInt16LE(i)) * factor), i); return out; }
function quantizeMx(buf, N, K) {
  const codes = Buffer.alloc(N * K / 2), scales = Buffer.alloc(N * K / 32);
  for (let r = 0; r < N; r++) for (let g = 0; g < K / 32; g++) {
    let mx = 0; const vals = new Float32Array(32);
    for (let i = 0; i < 32; i++) { const v = bf2f(buf.readUInt16LE((r * K + g * 32 + i) * 2)); vals[i] = v; if (Math.abs(v) > mx) mx = Math.abs(v); }
    let exp = mx > 0 ? Math.ceil(Math.log2(mx / 6.0)) : -126; exp = Math.max(-126, Math.min(127, exp));
    scales[r * (K / 32) + g] = exp + 127; const sc = Math.pow(2, exp);
    for (let i = 0; i < 32; i++) {
      const q = vals[i] / sc; let best = 0, bd = Infinity;
      for (let c = 0; c < 16; c++) { const d = Math.abs(E2M1[c] - q); if (d < bd) { bd = d; best = c; } }
      const bi = r * (K / 2) + g * 16 + (i >> 1); if ((i & 1) === 0) codes[bi] = best; else codes[bi] |= (best << 4);
    }
  }
  return {codes, scales};
}
function quantizeFp8(buf, N, K) {
  const tbl = new Float32Array(256);
  for (let u = 0; u < 256; u++) { const s = (u >> 7) ? -1 : 1, e = (u >> 3) & 0xf, m = u & 7; tbl[u] = (e === 15 && m === 7) ? NaN : (e === 0 ? s * (m / 8) * Math.pow(2, -6) : s * (1 + m / 8) * Math.pow(2, e - 7)); }
  const codes = Buffer.alloc(N * K), bn = N / 128, bk = K / 128, scales = Buffer.alloc(bn * bk * 4);
  for (let br = 0; br < bn; br++) for (let bc = 0; bc < bk; bc++) {
    let mx = 0;
    for (let i = 0; i < 128; i++) for (let j = 0; j < 128; j++) { const v = Math.abs(bf2f(buf.readUInt16LE(((br * 128 + i) * K + bc * 128 + j) * 2))); if (v > mx) mx = v; }
    const sc = mx > 0 ? mx / 448.0 : 1.0; scales.writeFloatLE(sc, (br * bk + bc) * 4);
    for (let i = 0; i < 128; i++) for (let j = 0; j < 128; j++) {
      const r = br * 128 + i, c = bc * 128 + j, v = bf2f(buf.readUInt16LE((r * K + c) * 2)) / sc;
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

// The checkpoint directory (built once per process and HCA geometry; `hover` overrides fields of HGEO, e.g. {D: 124}).
const targets = new Map();
function buildTarget(hover = {}) {
  const hg = {...HGEO, ...hover};
  const id = JSON.stringify(hover);
  if (targets.has(id)) return targets.get(id);
  const dir = path.join(h.tmpRoot, "kv-model-" + targets.size);
  const cfg = fx.config("MHMH".split(""), 1, OVER, GEO, hg);
  const realistic = ts => ts.map(t => (t.dtype === "BF16" && t.shape.length === 2 && !t.name.includes("embed") && !t.name.includes("vha"))
    ? {...t, data: scaleBuf(t.data, 1.0 / Math.sqrt(t.shape[1]))} : t);
  fx.build(dir, cfg, {seed: SEED, geo: GEO, h: hg, mutate: ts => realistic(ts).flatMap(t => {
    if (isExpert(t.name)) { const q = quantizeMx(t.data, t.shape[0], t.shape[1]); return [{...t, dtype: "I8", shape: [t.shape[0], t.shape[1] / 2], data: q.codes}, {name: t.name + "_scale_inv", dtype: "F8_E8M0", shape: [t.shape[0], t.shape[1] / 32], data: q.scales}]; }
    if (isFp8Trunk(t.name, t.shape)) { const q = quantizeFp8(t.data, t.shape[0], t.shape[1]); return [{...t, dtype: "F8_E4M3", shape: t.shape, data: q.codes}, {name: t.name + "_scale_inv", dtype: "F32", shape: [t.shape[0] / 128, t.shape[1] / 128], data: q.scales}]; }
    return [t];
  })});
  targets.set(id, dir);
  return dir;
}

// The container for one set of conversion keys (cached by key list).
const converted = new Map();
function convertTarget(extraKeys = [], hover = {}) {
  const keys = ["ernie55_quant_forward=1", `ernie55_attention_block=${ATTN_BLOCK}`, `ernie55_hca_period=${PERIOD}`, ...extraKeys];
  const id = keys.join(",") + JSON.stringify(hover);
  if (!converted.has(id)) {
    const conv = h.convert(buildTarget(hover), {keys});
    assert.equal(conv.status, 0, "convert must work: " + (conv.stderr || "").slice(-800));
    converted.set(id, conv.out);
  }
  return converted.get(id);
}

function yaml(rad, tp, kvdt, debugGraph, maxtok = 64, ckpt = false) {
  const b = h.ensureBuilt();
  const dir = fs.mkdtempSync(path.join(h.tmpRoot, "srv-"));
  const file = path.join(dir, "serve.yaml");
  fs.writeFileSync(file, ["server:", "  host: 127.0.0.1", "  port: 0", "  log: info", `  radiance-home: ${b.home}`, "  kernels: libref", "  debug-accept-reference-kernels: true", "cards: all",
    "models:", "  ern:", `    model: ${rad}`, `    tp: ${tp}`, ...(debugGraph ? ["    debug-graph: true"] : []), "    max-num-seqs: 4", `    max-num-batched-tokens: ${maxtok}`, "    max-model-len: 1024",
    ...(ckpt ? ["    checkpoint-slots: 8", "    checkpoint-interval: 8"] : []), ...(kvdt ? [`    kv-cache-dtype: ${kvdt}`] : []), ""].join("\n"));
  return {b, file};
}
const envFor = (tp, extra) => ({...process.env, ROCR_VISIBLE_DEVICES: "99", RADIANCE_HOST_DEVICES: String(tp), RADIANCE_HOST_VRAM_MIB: "512", OMP_NUM_THREADS: "2", OMP_WAIT_POLICY: "passive", ...extra});

// `radiance --config` with debug-graph: the declared program and the KV allocation, printed instead of served. {status, text}.
function declare(rad, {tp = 1, kvdt = "", env = {}} = {}) {
  const {b, file} = yaml(rad, tp, kvdt, true);
  const r = h.sh(b.radiance, ["--config", file], {env: envFor(tp, env)});
  return {status: r.status, text: r.stdout + r.stderr};
}

// Serve one container; resolves {port, log, stop} or {failed, log}.
function serve(rad, {tp = 1, kvdt = "", env = {}, maxtok = 64, ckpt = false} = {}) {
  const {b, file} = yaml(rad, tp, kvdt, false, maxtok, ckpt);
  const child = spawn(b.radiance, ["--config", file], {env: envFor(tp, env), stdio: ["ignore", "pipe", "pipe"]});
  let log = ""; child.stdout.on("data", d => { log += d; }); child.stderr.on("data", d => { log += d; });
  return new Promise(resolve => {
    const timer = setTimeout(() => { child.kill("SIGKILL"); resolve({failed: true, log: log + "\n[timeout]"}); }, 180000);
    const tick = setInterval(() => { const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(log); if (m && /model ern: ready/.test(log)) { clearInterval(tick); clearTimeout(timer); resolve({port: Number(m[1]), get log() { return log; }, stop: () => child.kill("SIGKILL")}); } }, 50);
    child.on("exit", () => { clearInterval(tick); clearTimeout(timer); resolve({failed: true, log}); });
  });
}

// Single-sequence greedy generation with the top-5 logprobs of every step, and the prefix-cache counters.
async function gen(srv, prompt, n) {
  const r = await h.complete(srv, prompt, {max_tokens: n, logprobs: 5});
  assert.equal(r.status, 200, JSON.stringify(r.body).slice(0, 400));
  const c = r.body.choices[0].logprobs.content;
  return {ids: c.map(x => x.id), top: c.map(x => x.top_logprobs.map(t => [t.id, t.logprob]))};
}
async function prefixCounters(srv) {
  const r = await fetch(`http://127.0.0.1:${srv.port}/metrics`);
  return (await r.text()).split("\n").filter(l => /prefix/i.test(l) && !l.startsWith("#") && /(queries|hit|tokens)/i.test(l));
}
const rndPrompt = (seed, len) => { let s = seed; const o = []; for (let i = 0; i < len; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; o.push((s >>> 8) % VOCAB); } return o; };

// The allocator report's group lines: name -> {kind, layers, block (tokens), pageBytes}.
const UNIT = {B: 1, KiB: 1024, MiB: 1024 * 1024};
function groups(text) {
  const out = {};
  for (const l of text.split("\n")) {
    const m = /^\s+(kv_\w+)\s+(full|window)\s+(\d+) layers\s+block\s+(\d+) tok\s+page\s+([\d.]+) (B|KiB|MiB)\s+(\d+) blocks/.exec(l);
    if (m) out[m[1]] = {kind: m[2], layers: Number(m[3]), block: Number(m[4]), pageBytes: Math.round(Number(m[5]) * UNIT[m[6]]), blocks: Number(m[7])};
  }
  return out;
}
// The program's op blocks: a header line (`op k=v ...`) followed by indented lines, blocks separated by blank lines.
const blocks = text => text.split(/\n\s*\n/).map(b => b.split("\n")).filter(b => /^[a-z_0-9]+  /.test(b[0]));
const params = header => Object.fromEntries(header.split(/\s+/).slice(1).filter(x => x.includes("=")).map(kv => { const i = kv.indexOf("="); return [kv.slice(0, i), kv.slice(i + 1)]; }));

// The raw conversion result for a key set that may be refused: {status, stderr}.
function tryConvert(extraKeys = [], hover = {}) {
  const keys = ["ernie55_quant_forward=1", `ernie55_attention_block=${ATTN_BLOCK}`, `ernie55_hca_period=${PERIOD}`, ...extraKeys];
  const conv = h.convert(buildTarget(hover), {keys});
  return {status: conv.status, stderr: conv.stderr || ""};
}

module.exports = {h, tryConvert, gen, prefixCounters, rndPrompt, GEO, HGEO, ATTN_BLOCK, PERIOD, VOCAB, buildTarget, convertTarget, declare, serve, groups, blocks, params};
