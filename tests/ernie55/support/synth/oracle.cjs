'use strict';
// Independent double-precision oracle for the 2-7 feed-forward program, extended for 2-8 with the routed
// latent-MoE branch. It is written from the pinned Baidu model (vLLM-ERNIE b8414da, nvidia/model.py +
// nvidia/moe.py + mhc.py) and the 1-1 SiTU definition, and shares no code with libref or the plugin. The
// only rounding is the explicit BF16 steps, each of which records how far its input sat from a BF16
// rounding midpoint, so a fixture can be chosen whose every rounding decision is immune to
// f32-versus-double accumulation.
//
//   embedding row (bf16) -> hc identical streams (first layer's broadcast)
//   per layer:  x, post, comb = mhc_pre(h) with the fused weighted RMSNorm (ffn_norm)
//               dense/shared: gate = bf16(Wg x), up = bf16(Wu x), a = bf16(situ(gate, up)), y = bf16(Wd a)
//               routed layer (2-8, nvidia/moe.py Ernie5_5MoE.forward):
//                 logits = [Wr0 x | Wr1 x] (f32, never narrowed)       scores = sigmoid(l[:E]) + sigmoid(l[E:])
//                 ids = top-k of scores + e_score_correction_bias (ties to the lower id)
//                 w = scores[ids]; w /= max(sum(w), 1e-20) (norm_topk_prob); w *= scale[ids]   (f32, per expert id)
//                 xl = bf16(fc1 x)                                       latent width, fed to the experts
//                 per expert: g = bf16(W1 xl), u = bf16(W3 xl), a = bf16(situ(g, u)), ye = bf16(W2 a)
//                 comb = bf16(sum_j w_j ye_j)                            experts COMBINED first
//                 nl = bf16(rmsnorm(comb) * latent_norm)                 latent norm AFTER the combine
//                 ro = bf16(fc2 nl)                                      routed output transform
//                 y = bf16(ro + shared(x))                               shared expert reads the ORIGINAL x
//               h' = bf16(post[d] * y + sum_s comb[s][d] * h[s])
//   pooled = bf16(mean over a sequence's tokens of the final streams)
function bf16(value) {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value);
  const bits = view.getUint32(0);
  view.setUint32(0, ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16);
  return view.getFloat32(0);
}
// Relative distance of a value from the nearest BF16 rounding midpoint.
function margin(value) {
  if (!Number.isFinite(value) || value === 0) return Infinity;
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, Math.abs(value));
  const bits = view.getUint32(0) & 0xffff0000;
  view.setUint32(0, bits);
  const lo = view.getFloat32(0);
  view.setUint32(0, bits + 0x10000);
  const hi = view.getFloat32(0);
  return Math.abs(Math.abs(value) - (lo + hi) / 2) / Math.abs(value);
}
const sum = v => v.reduce((a, b) => a + b, 0);
const sigmoid = x => 1 / (1 + Math.exp(-x));

// The SiTU gated product of 1-1: beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta).
function situ(g, u, beta, lbeta, variant) {
  if (variant === 'silu') return g * sigmoid(g) * u;
  const gate = variant === 'noGateClip' ? g * sigmoid(g) : beta * Math.tanh(g / beta) * sigmoid(g);
  const up = variant === 'noUpClip' ? u : lbeta * Math.tanh(u / lbeta);
  return gate * up;
}

// Variants that are each one plausible wrong equation; a fixture must tell every one apart.
const VARIANTS = ['swapGateUp', 'silu', 'noUpClip', 'noGateClip', 'swapBetas', 'noInterRounding',
  'unnormed', 'dropResidual', 'sharedOnLatent', 'postMultOne', 'epsSwapped', 'transposedComb'];
// 2-8: wrong routed-block equations (a fixture with routed layers must tell every one apart).
const MOE_VARIANTS = ['perExpertNorm', 'noLatentNorm', 'biasWeights', 'noScale', 'scaleBeforeRenorm',
  'noRenorm', 'onlyRouter0', 'noBiasSelect', 'softmaxRouter', 'expertIdShift', 'noShared',
  'swapExpertGateUp', 'siluExpert', 'uniformWeights', 'bf16Logits', 'bf16Aux'];

const dot = (w, v) => sum(v.map((a, k) => a * w[k]));

// The routed branch of one token: x is the normed full-width input (bf16 values). Returns
// {y: routed output after the transform (bf16), ids, w, comb, nl, xl, experts: [{g,u,a,ye}], gap}.
function routed(M, x, ck, variant, r) {
  const E = M.E, k = M.k;
  const rnd = v => (variant === 'bf16Logits' ? bf16(v) : v);
  const l0 = M.r0.map(w => rnd(dot(w, x)));
  const l1 = M.r1.map(w => rnd(dot(w, x)));
  const bias = variant === 'bf16Aux' ? M.bias.map(bf16) : M.bias;
  const scaleP = variant === 'bf16Aux' ? M.scale.map(bf16) : M.scale;
  let score;
  if (variant === 'softmaxRouter') {
    const mx = Math.max(...l0); const ex = l0.map(v => Math.exp(v - mx)); const s = sum(ex); score = ex.map(v => v / s);
  } else if (variant === 'onlyRouter0') score = l0.map(sigmoid);
  else score = l0.map((v, e) => sigmoid(v) + sigmoid(l1[e]));
  const sel = score.map((s, e) => s + (variant === 'noBiasSelect' ? 0 : bias[e]));
  const order = sel.map((s, e) => e).sort((a, b) => sel[b] - sel[a] || a - b);
  const ids = order.slice(0, k);
  const gap = sel[order[k - 1]] - sel[order[k]];
  let raw = ids.map(e => (variant === 'biasWeights' ? sel[e] : score[e]));
  const scl = e => (variant === 'noScale' ? 1 : scaleP[e]);
  let w;
  if (variant === 'uniformWeights') w = ids.map(e => scl(e) / k);
  else if (variant === 'scaleBeforeRenorm') {
    const t = raw.map((v, j) => v * scl(ids[j])); const s = Math.max(sum(t), 1e-20);
    w = t.map(v => (M.renorm ? v / s : v));
  } else {
    const s = Math.max(sum(raw), 1e-20);
    w = raw.map((v, j) => (variant === 'noRenorm' || !M.renorm ? v : v / s) * scl(ids[j]));
  }
  const xl = M.fc1.map(wr => r(dot(wr, x)));
  const experts = ids.map(e0 => {
    const e = variant === 'expertIdShift' ? (e0 + 1) % E : e0;
    const X = M.exps[e];
    const gw = variant === 'swapExpertGateUp' ? X.w3 : X.w1, uw = variant === 'swapExpertGateUp' ? X.w1 : X.w3;
    const gg = gw.map(wr => r(dot(wr, xl))), uu = uw.map(wr => r(dot(wr, xl)));
    const a = gg.map((v, i) => r(situ(v, uu[i], ck.beta, ck.lbeta, variant === 'siluExpert' ? 'silu' : 'exact')));
    const ye = X.w2.map(wr => r(dot(wr, a)));
    return {g: gg, u: uu, a, ye};
  });
  const nrm = v => { const q = 1 / Math.sqrt(sum(v.map(t => t * t)) / v.length + ck.rmsEps); return v.map((t, d) => r(t * q * M.lnorm[d])); };
  let comb, nl;
  if (variant === 'perExpertNorm') {
    const normed = experts.map(e => nrm(e.ye));
    comb = Array.from({length: xl.length}, (_, d) => r(sum(normed.map((n, j) => w[j] * n[d]))));
    nl = comb;
  } else {
    comb = Array.from({length: xl.length}, (_, d) => r(sum(experts.map((e, j) => w[j] * e.ye[d]))));
    nl = variant === 'noLatentNorm' ? comb : nrm(comb);
  }
  const ro = M.fc2.map(wr => r(dot(wr, nl)));
  return {ro, ids, w, comb, nl, xl, experts, gap, sel, score};
}

// ck: {H, hc, V, iters, rmsEps, hcEps, beta, lbeta, firstDense, layers: [{kind, I, fn[mix][W],
// scale[3], base[mix], gain[H], gate[I][H], up[I][H], down[H][I], moe? }], emb[V][H]}. A `routed` layer's
// moe is {E, k, renorm, r0[E][H], r1[E][H], bias[E], scale[E], fc1[Lt][H], fc2[H][Lt], lnorm[Lt],
// exps[E]: {w1[F][Lt], w3[F][Lt], w2[Lt][F]}}.
// tokens: array of token ids (rows are independent). Returns the final streams per token and the
// smallest rounding margin seen.
function run(ck, tokens, variant = 'exact') {
  const {H, hc} = ck;
  const mix = hc * (hc + 2);
  let minMargin = Infinity, minGap = Infinity;
  const trace = [];
  const r = v => {
    if (variant === 'noInterRounding') return v;
    minMargin = Math.min(minMargin, margin(v));
    return bf16(v);
  };
  // Roundings that exist in every variant (embedding table, stream, collapse) use this one.
  const rr = v => { minMargin = Math.min(minMargin, margin(v)); return bf16(v); };
  const beta = variant === 'swapBetas' ? ck.lbeta : ck.beta;
  const lbeta = variant === 'swapBetas' ? ck.beta : ck.lbeta;
  const postMult = variant === 'postMultOne' ? 1 : 2;
  const rows = tokens.map(t => {
    let h = [];
    for (let s = 0; s < hc; s++) h.push(...ck.emb[t]);        // hc_enter
    const layerTrace = [];
    for (const L of ck.layers) {
      const W = hc * H;
      const eps = variant === 'epsSwapped' ? ck.hcEps : ck.rmsEps;
      const preEps = variant === 'epsSwapped' ? ck.rmsEps : ck.hcEps;
      const denom = Math.sqrt(sum(h.map(v => v * v)) / W) + eps;   // projection eps sits outside the root
      const z = L.fn.map((w, j) => sum(w.map((a, k) => a * h[k])) / denom *
        L.scale[Math.min(2, Math.floor(j / hc))] + L.base[j]);
      const pre = z.slice(0, hc).map(v => sigmoid(v) + preEps);
      const post = z.slice(hc, 2 * hc).map(v => sigmoid(v) * postMult);
      const c = Array.from({length: hc}, (_, j) => {
        const lg = z.slice(2 * hc + j * hc, 2 * hc + (j + 1) * hc);
        const e = lg.map(v => Math.exp(v - Math.max(...lg)));
        return e.map(v => v / sum(e) + preEps);
      });
      for (let it = 0; it < ck.iters; it++) {
        if (it > 0) for (let j = 0; j < hc; j++) { const d = sum(c[j]) + preEps; c[j] = c[j].map(v => v / d); }
        for (let k = 0; k < hc; k++) { const d = sum(c.map(q => q[k])) + preEps; for (let j = 0; j < hc; j++) c[j][k] /= d; }
      }
      const col = Array.from({length: H}, (_, i) => sum(pre.map((g, s) => g * h[s * H + i])));
      const rms = Math.sqrt(sum(col.map(v => v * v)) / H + ck.rmsEps);
      const x = col.map((v, i) => {
        const cr = rr(v);
        if (variant === 'unnormed') return cr;
        return rr(cr * L.gain[i] / rms);
      });
      const isRouted = L.kind === 'routed';
      const input = variant === 'sharedOnLatent' && L.kind !== 'dense' ? x.slice(0, ck.latent) : x;
      const gW = variant === 'swapGateUp' ? L.up : L.gate, uW = variant === 'swapGateUp' ? L.gate : L.up;
      const gate = L.gate.map((_, i) => r(dot(gW[i], input)));
      const up = L.up.map((_, i) => r(dot(uW[i], input)));
      const act = gate.map((g, i) => r(situ(g, up[i], beta, lbeta, variant)));
      let y = L.down.map(w => r(dot(w, act)));
      let rt = null;
      if (isRouted) {
        const sharedY = variant === 'noShared' ? null : y;
        rt = routed(L.moe, x, ck, variant, r);
        minGap = Math.min(minGap, rt.gap);
        // The add of two BF16 values is exact in f32, so a tie rounds the same in the engine: no margin is taken.
        y = rt.ro.map((v, d) => (sharedY ? (variant === 'noInterRounding' ? v + sharedY[d] : bf16(v + sharedY[d])) : v));
        rt.shared = sharedY;
      }
      const next = [];
      for (let d = 0; d < hc; d++) for (let i = 0; i < H; i++) {
        let v = post[d] * y[i];
        if (variant !== 'dropResidual') for (let s = 0; s < hc; s++) v += (variant === 'transposedComb' ? c[d][s] : c[s][d]) * h[s * H + i];
        next.push(rr(v));
      }
      layerTrace.push({x, post, comb: c.flat(), gate, up, act, y, h: next, routed: rt});
      h = next;
    }
    trace.push(layerTrace);
    return h;
  });
  const pooled = Array.from({length: hc * H}, (_, i) => rr(sum(rows.map(h => h[i])) / rows.length));
  return {pooled, rows, trace, minMargin, minGap};
}

// What the server returns for a pooled vector: L2-normalised.
function normalise(v) { const n = Math.sqrt(sum(v.map(a => a * a))); return v.map(a => a / n); }

module.exports = {bf16, margin, run, normalise, situ, VARIANTS, MOE_VARIANTS, sum, sigmoid, routed};  // 3-2b: routed exported
