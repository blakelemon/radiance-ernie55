"use strict";
// (ernie/dsa-score-tiling: KVC_TILE sets ernie55_idx_tile, KVC_MAXLEN max-model-len, KVC_CONVERT_EXTRA extra rad-convert args, KVC_EXPECT_TILE=1 asserts the log reports a tiled score plane, KVC_DECLARE_ONLY=1 stops after the arena lines)
// kv-cmpblock end-to-end check on the synthetic ERNIE target (CPU libref): single-sequence greedy tokens AND logprobs, dumped to JSON,
// for one build and one compressed-group layout. kvcmp-compare.cjs then holds two dumps to bit identity.
//
// Run under the sysfs mask:  ERNIE_PORT_ROOT=<port root> RAD_BUILD_DIR=<repo with build-ernie55> nosys node kvcmp-e2e.cjs
// Env: KVC_OUT (json, required), KVC_TP (1), KVC_BLOCK (ernie55_attention_block, 8), KVC_MAXTOK (max-num-batched-tokens, 64),
//      KVC_LAYOUT ("" = the build's default | token | block: sets ernie55_hca_cmp_layout), KVC_SEED (7), KVC_LONG (400),
//      KVC_CKSLOTS (8), KVC_CKINT (8): checkpoint slots and interval, without them every prefix hit of this windowed model is cut
//      KVC_EXPECT_BLOCK ("1" = assert the allocator log shows the compressed group at block == period 4 with one row a page; the red gate)
// Every run is a single sequence alone in the engine (the target is not batch invariant, see batch-invariance-analysis.md).
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const lane = process.env.ERNIE_PORT_ROOT || path.join(__dirname, "support", "synth");
const h = require(path.join(__dirname, "support", "synth", "harness.cjs"));
const fx = require(path.join(__dirname, "support", "synth", "fixture34b.cjs"));
const E = k => process.env[k];
const TP = Number(E("KVC_TP") || 1), BLOCK = Number(E("KVC_BLOCK") || 8), MAXTOK = Number(E("KVC_MAXTOK") || 64), SEED = Number(E("KVC_SEED") || 7);
const CKS = Number(E("KVC_CKSLOTS") || 8), CKI = Number(E("KVC_CKINT") || 8);   // checkpoint slots: a prefix hit on a windowed model resumes from one
const NLONG = Number(E("KVC_LONG") || 400), VOCAB = 64, MAXLEN = Number(E("KVC_MAXLEN") || 1024), LAYOUT = E("KVC_LAYOUT") || "";

// KVC_PERIOD / KVC_D / KVC_WIN (4 / 128 / 8) reshape the HCA layers for the allocator report at the production period (KVC_STARTUP_ONLY=1); the identity runs use the defaults
const PERIOD = Number(E("KVC_PERIOD") || 4), HD = Number(E("KVC_D") || 128), WIN = Number(E("KVC_WIN") || 8);
const GEO = {heads: 4, nope: 112, rope: 16, vdim: 128, qlora: 128, lat: 64, ih: 2, id: 64, topk: 4, window: WIN, rank: 2, theta: 160000};
const HGEO = {heads: 4, D: HD, rope: 16, qlora: 128, groups: 4, rank: 128, vrank: 2, theta: 160000, period: PERIOD, window: WIN, yarn: {factor: 16, bf: 32, bs: 1, orig: 65536}};
const OVER = {
  vocab_size: VOCAB, hidden_size: 128, intermediate_size: 512, moe_intermediate_size: 512, moe_latent_size: 128,
  n_routed_experts: 4, num_experts_per_tok: 2, n_shared_experts: 1, hc_mult: 2,
  quantization_config: {activation_scheme: "dynamic", fmt: "e4m3", quant_method: "fp8", expert_dtype: "mxfp4", scale_fmt: "ue8m0", weight_block_size: [128, 128]}
};
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

function buildTarget() {
  const dir = path.join(h.tmpRoot, "kvc-model");
  const cfg = fx.config("MHMH".split(""), 1, OVER, GEO, HGEO);
  const realistic = ts => ts.map(t => (t.dtype === "BF16" && t.shape.length === 2 && !t.name.includes("embed") && !t.name.includes("vha"))
    ? {...t, data: scaleBuf(t.data, 1.0 / Math.sqrt(t.shape[1]))} : t);
  fx.build(dir, cfg, {seed: SEED, geo: GEO, h: HGEO, mutate: ts => realistic(ts).flatMap(t => {
    if (isExpert(t.name)) { const q = quantizeMx(t.data, t.shape[0], t.shape[1]); return [{...t, dtype: "I8", shape: [t.shape[0], t.shape[1] / 2], data: q.codes}, {name: t.name + "_scale_inv", dtype: "F8_E8M0", shape: [t.shape[0], t.shape[1] / 32], data: q.scales}]; }
    if (isFp8Trunk(t.name, t.shape)) { const q = quantizeFp8(t.data, t.shape[0], t.shape[1]); return [{...t, dtype: "F8_E4M3", shape: t.shape, data: q.codes}, {name: t.name + "_scale_inv", dtype: "F32", shape: [t.shape[0] / 128, t.shape[1] / 128], data: q.scales}]; }
    return [t];
  })});
  return dir;
}

function serve(rad, tp) {
  const b = h.ensureBuilt();
  const dir = fs.mkdtempSync(path.join(h.tmpRoot, "srv-"));
  const yaml = path.join(dir, "serve.yaml");
  fs.writeFileSync(yaml, ["server:", "  host: 127.0.0.1", "  port: 0", "  log: info", `  radiance-home: ${b.home}`, "  kernels: libref", "  debug-accept-reference-kernels: true", "cards: all",
    "models:", "  ern:", `    model: ${rad}`, `    tp: ${tp}`, `    max-num-seqs: 4`, `    max-num-batched-tokens: ${MAXTOK}`, `    max-model-len: ${MAXLEN}`,
    ...(CKS > 0 ? [`    checkpoint-slots: ${CKS}`, `    checkpoint-interval: ${CKI}`] : []), ""].join("\n"));
  const child = spawn(b.radiance, ["--config", yaml], {env: {...process.env, ROCR_VISIBLE_DEVICES: "99", RADIANCE_HOST_DEVICES: String(tp), RADIANCE_HOST_VRAM_MIB: "512", OMP_NUM_THREADS: "2", OMP_WAIT_POLICY: "passive"}, stdio: ["ignore", "pipe", "pipe"]});
  let log = ""; child.stdout.on("data", d => { log += d; }); child.stderr.on("data", d => { log += d; });
  return new Promise(resolve => {
    const timer = setTimeout(() => { child.kill("SIGKILL"); resolve({failed: true, log: log + "\n[timeout]"}); }, 180000);
    const tick = setInterval(() => { const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(log); if (m && /model ern: ready/.test(log)) { clearInterval(tick); clearTimeout(timer); resolve({port: Number(m[1]), get log() { return log; }, stop: () => child.kill("SIGKILL")}); } }, 50);
    child.on("exit", () => { clearInterval(tick); clearTimeout(timer); resolve({failed: true, log}); });
  });
}
async function gen(srv, prompt, n) {
  const r = await h.complete(srv, prompt, {max_tokens: n, logprobs: 5});
  assert.equal(r.status, 200, JSON.stringify(r.body).slice(0, 400));
  const c = r.body.choices[0].logprobs.content;
  return {ids: c.map(x => x.id), top: c.map(x => x.top_logprobs.map(t => [t.id, t.logprob]))};
}
async function metrics(srv) {
  const r = await fetch(`http://127.0.0.1:${srv.port}/metrics`);
  return (await r.text()).split("\n").filter(l => /prefix/i.test(l) && !l.startsWith("#") && /(queries|hit|tokens)/i.test(l));
}
const rndPrompt = (seed, len) => { let s = seed; const o = []; for (let i = 0; i < len; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; o.push((s >>> 8) % VOCAB); } return o; };

(async () => {
  const dir = buildTarget();
  const keys = ["ernie55_quant_forward=1", `ernie55_attention_block=${BLOCK}`, `ernie55_hca_period=${PERIOD}`];
  if (LAYOUT) keys.push(`ernie55_hca_cmp_layout=${LAYOUT}`);
  if (E("KVC_TILE")) keys.push(`ernie55_idx_tile=${E("KVC_TILE")}`);
  const conv = h.convert(dir, {keys, extra: (E("KVC_CONVERT_EXTRA") || "").split(" ").filter(Boolean)});
  assert.equal(conv.status, 0, conv.stderr.slice(0, 1500));
  const srv = await serve(conv.out, TP);
  assert.ok(!srv.failed, srv.log.split("\n").filter(l => /^E |^W |invalid|batched|chunk|checkpoint|quantum|interval/i.test(l)).slice(0, 30).join("\n"));
  const out = {cfg: {TP, BLOCK, MAXTOK, MAXLEN, SEED, LAYOUT, NLONG, TILE: E("KVC_TILE") || "", repo: h.repo}, runs: {}, prefix: {}, alloc: []};
  try {
    out.alloc = srv.log.split("\n").filter(l => /kv_hca|kv_latent|kv_dsa|KV pool|kv: |per token|tokens\/|page |arena|score plane|max-model-len|vram/i.test(l)).slice(0, 80);
    if (E("KVC_EXPECT_TILE") === "1") assert.ok(out.alloc.some(l => /indexer score plane.*tiled/i.test(l)), "the serve log must report a tiled indexer score plane: " + out.alloc.join(" | "));
    if (E("KVC_DECLARE_ONLY") === "1") { out.errors = []; fs.writeFileSync(E("KVC_OUT"), JSON.stringify(out, null, 1)); console.log(out.alloc.join("\n")); srv.stop(); return; }
    if (E("KVC_EXPECT_BLOCK") === "1") {
      const line = out.alloc.find(l => /kv_hca_compressed/.test(l) && /block/.test(l));
      assert.ok(line && new RegExp(`block\\s+${PERIOD}\\s+tok`).test(line) && /page\s+0?\.\d+\s+KiB|page\s+[\d.]+\s+(KiB|B)/.test(line), "the compressed group must be allocated at block 4 (= period): " + (line || "no kv_hca_compressed line in the log"));
    }
    if (E("KVC_STARTUP_ONLY") === "1") { out.errors = []; fs.writeFileSync(E("KVC_OUT"), JSON.stringify(out, null, 1)); console.log(out.alloc.join("\n")); srv.stop(); return; }
    const run = async (name, prompt, n) => { out.runs[name] = await gen(srv, prompt, n); out.runs[name].len = prompt.length; };
    // period-4 analogues of the production boundaries 126-129 / 255-257: prompts around a multiple of the period, short and long
    for (const len of (E("KVC_LENS") || "1,3,4,5,7,8,9,12,13,15,16,17,25,33").split(",").map(Number)) await run(`len${len}`, rndPrompt(100 + len, len), 24);
    // a long single sequence: recycles the window (sliding window 8) many times and crosses ~100 periods
    await run("long", rndPrompt(55, 13), NLONG);
    // prefix reuse: X (41 tokens: the cap leaves a hit of 40, a multiple of every block size here), a prompt sharing X[:24] then diverging, X again
    // (KVC_XLEN / KVC_SHARE: the production-period run uses 385 / 256, whole multiples of 128 that leave a hit of 384)
    const XL = Number(E("KVC_XLEN") || 41), SH = Number(E("KVC_SHARE") || 24);
    const X = rndPrompt(77, XL), Y = X.slice(0, SH).concat(rndPrompt(88, 15));
    out.prefix.before = await metrics(srv);
    await run("pfx_X", X, 30);
    await run(`pfx_Y_shares_${SH}`, Y, 30);
    await run("pfx_X_again", X, 30);
    out.prefix.after = await metrics(srv);
    out.errors = srv.log.split("\n").filter(l => /assert|fatal|error/i.test(l) && !/control vectors/.test(l)).slice(0, 5);
  } finally { srv.stop(); }
  fs.writeFileSync(E("KVC_OUT"), JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out.cfg));
  console.log("runs:", Object.keys(out.runs).length, "alloc lines:", out.alloc.length, "prefix:", JSON.stringify(out.prefix));
  for (const l of out.alloc) console.log("  " + l);
})().catch(e => { console.error(e); process.exit(2); });
