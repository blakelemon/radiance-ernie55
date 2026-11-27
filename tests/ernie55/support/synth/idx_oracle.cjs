// Independent double-precision oracles for the ERNIE MLA-DSA indexer score path (story 3-2i).
// Nothing here shares code with libref: the Hadamard rotation is the explicit +-1 matrix (not a
// butterfly), the E4M3 codec is built from the value table, the select is a sort. Rounding to bf16
// happens exactly where the pinned model rounds (docs/ernie55/attention-reference.md m09/m10).
const {bf, rng} = require('./probe.cjs');

function gauss(seed) {
  const r = rng(seed);
  return () => { const u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
}
const randBf = (n, seed, sc = 1) => { const g = gauss(seed); return Array.from({length: n}, () => bf(sc * g())); };

// ---- E4M3 (finite, no inf, max 448) -------------------------------------------------------
const E4 = [];
for (let c = 0; c < 127; c++) { const e = c >> 3, m = c & 7; E4.push(e === 0 ? (m / 8) * 2 ** -6 : (1 + m / 8) * 2 ** (e - 7)); }
function e4enc(x) {
  const sign = x < 0 || Object.is(x, -0) ? 128 : 0, a = Math.min(Math.abs(x), 448);
  if (a >= E4[126]) return sign | 126;
  let lo = 0, hi = 126;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (E4[mid] <= a) lo = mid; else hi = mid; }
  const dl = a - E4[lo], dh = E4[lo + 1] - a;
  const c = dl < dh ? lo : dh < dl ? lo + 1 : (lo % 2 === 0 ? lo : lo + 1);
  return sign | c;
}
const e4dec = b => { const v = E4[b & 127]; return b & 128 ? -v : v; };

// ---- Hadamard: the explicit orthonormal matrix H[i][j] = (-1)^popcount(i & j) / sqrt(n) ------
const hadCache = new Map();
function hadMatrix(n) {
  if (!hadCache.has(n)) {
    const H = new Float64Array(n * n), par = x => { let c = 0; while (x) { c ^= x & 1; x >>= 1; } return c; };
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) H[i * n + j] = (par(i & j) ? -1 : 1) / Math.sqrt(n);
    hadCache.set(n, H);
  }
  return hadCache.get(n);
}
function hadamard(v, normalised = true) {
  const n = v.length, H = hadMatrix(n), o = new Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += H[i * n + j] * v[j]; o[i] = normalised ? s : s * Math.sqrt(n); }
  return o;
}

// ---- rope: `neox` pairs (i, i + R/2); `gptj` pairs (2i, 2i + 1). Absolute position `pos`. -----
// `table` is the dtype of the cos/sin table: bf16 (the model dtype, default) or f32.
function ropeRot(v, pos, theta, table = 'bf16', style = 'neox') {
  const R = v.length, o = v.slice();
  for (let i = 0; i < R / 2; i++) {
    const ang = pos * Math.pow(theta, -2 * i / R), c0 = Math.cos(ang), s0 = Math.sin(ang);
    const c = table === 'f32' ? Math.fround(c0) : bf(c0), s = table === 'f32' ? Math.fround(s0) : bf(s0);
    const a = style === 'neox' ? i : 2 * i, b = style === 'neox' ? i + R / 2 : 2 * i + 1;
    o[a] = v[a] * c - v[b] * s;
    o[b] = v[b] * c + v[a] * s;
  }
  return o.map(bf);
}

// ---- power-of-two (UE8M0) E4M3 quantiser over one group --------------------------------------
function quantGroup(v, eps = 1e-10, pow2 = true) {
  const am = Math.max(eps, ...v.map(Math.abs)), t = am / 448;
  let scale;
  if (pow2) {
    let e = Math.ceil(Math.log2(t));
    while (2 ** e < t) e++;
    while (2 ** (e - 1) >= t) e--;
    scale = 2 ** e;
  } else scale = t;
  return {codes: v.map(x => e4enc(Math.max(-448, Math.min(448, x / scale)))), scale};
}
const f32bytes = x => Array.from(new Uint8Array(new Float32Array([x]).buffer));
const bytesToF32 = b => new Float32Array(new Uint8Array(b).buffer)[0];

// ---- the indexer pipeline ---------------------------------------------------------------------
// pre-quantisation vector of one head (query) or the key row after the norm: split rope-first,
// rotate the rope part at the absolute position, concatenate, Hadamard, bf16.
function indexerVector(x, pos, o) {
  const D = x.length, R = o.R, order = o.order || 'rope-first';
  const rp = order === 'rope-first' ? x.slice(0, R) : x.slice(D - R), nope = order === 'rope-first' ? x.slice(R) : x.slice(0, D - R);
  const rot = o.rope === false ? rp : ropeRot(rp, pos, o.theta, o.table || 'bf16', o.style || 'neox');
  const v = order === 'rope-first' ? [...rot, ...nope] : [...nope, ...rot];
  return o.hadamard === false ? v : hadamard(v, o.hnorm !== false).map(bf);
}
function layerNorm(x, g, b, eps, useBias = true) {
  const n = x.length, mean = x.reduce((a, c) => a + c, 0) / n, v = x.reduce((a, c) => a + (c - mean) ** 2, 0) / n, r = 1 / Math.sqrt(v + eps);
  return x.map((c, i) => bf((c - mean) * r * g[i] + (useBias ? b[i] : 0)));
}
// q rows: [M][H*D] -> codes [M][H*D], scales [M][H]
function oracleQuery(q, pos, H, D, o) {
  return q.map((row, m) => {
    const codes = [], scales = [];
    for (let h = 0; h < H; h++) {
      const g = quantGroup(indexerVector(row.slice(h * D, (h + 1) * D), pos[m], o), o.eps, o.pow2);
      codes.push(...g.codes); scales.push(g.scale);
    }
    return {codes, scales};
  });
}
// one key row (raw projection of one token) -> {codes, scale}
function oracleKey(k, g, b, pos, o) {
  const x = layerNorm(k, g, b, o.lnEps === undefined ? 1e-6 : o.lnEps, o.bias !== false);
  const pre = o.lnAfterRope ? indexerVector(k, pos, o) : indexerVector(x, pos, o);
  return quantGroup(o.lnAfterRope ? layerNorm(pre, g, b, 1e-6).map(bf) : pre, o.eps, o.pow2);
}
const rowBytes = ({codes, scale}) => [...codes, ...f32bytes(scale)];
const deq = ({codes, scale}) => codes.map(c => e4dec(c) * scale);

// logits of one query row against keys 0..end-1 (keys: [{codes, scale}] by LOGICAL token, null = unusable)
function oracleLogits({qc, qs, w, keys, end, L, H, D, window = 128, mut = {}}) {
  const ws = Math.pow(D, -0.5) * Math.pow(H, -0.5) * (mut.scale || 1);
  const out = new Array(L).fill(-Infinity);
  const qd = []; for (let h = 0; h < H; h++) qd.push(qc.slice(h * D, (h + 1) * D).map(e4dec));
  for (let j = 0; j < end && j < L; j++) {
    const key = keys[j]; if (!key) continue;
    const kd = key.codes.map(e4dec);
    let s = 0;
    for (let h = 0; h < H; h++) {
      let dot = 0; for (let d = 0; d < D; d++) dot += qd[h][d] * kd[d];
      let t = dot * key.scale;
      if (mut.bf16dot) t = bf(dot) * key.scale;
      if (mut.norelu !== true) t = Math.max(t, 0);
      s += t * (bf(w[h]) * qs[h] * ws);
    }
    out[j] = s;
  }
  return out;
}
// the selection row of ernie_dsa_select
function oracleSelect(scores, p, topk, window, width = topk + window) {
  const seq = p + 1, end = Math.max(seq - window, 0);
  let cand = []; for (let i = 0; i < end; i++) if (scores[i] > -Infinity) cand.push(i);
  if (cand.length > topk) cand = cand.sort((a, b) => scores[b] - scores[a] || a - b).slice(0, topk);
  const row = [...cand]; for (let i = seq - Math.min(seq, window); i < seq; i++) row.push(i);
  while (row.length < width) row.push(-1);
  return row;
}
module.exports = {gauss, randBf, e4enc, e4dec, hadamard, ropeRot, quantGroup, f32bytes, bytesToF32, indexerVector, layerNorm,
  oracleQuery, oracleKey, rowBytes, deq, oracleLogits, oracleSelect};
