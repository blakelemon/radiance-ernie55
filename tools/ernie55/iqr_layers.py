#!/usr/bin/env python3
"""Streaming conversion and measurement of real ERNIE 5.5 Flash layers (story 5-0-iqr-v4-target). CPU only.

Source = the block-FP8 checkpoint, read tensor by tensor (header + that tensor's byte range; fp8_stream.py). Destination = the recipe of data/recipes/ernie55-iqr-v4.recipe, applied by libquant's own
quantisers through tools/ernie55/iqr_probe (quantise as rad-convert does, decode with the core's decoder). Reference = the FP8 dequantisation in f32. Nothing is calibrated: no ERNIE Hessians
exist, so GPTQ-class rules run with the identity Hessian (the recipe's calib option is dropped), an UPPER BOUND on the error of a calibrated conversion.

  iqr_layers.py route    --work W            # fixed tokens, the real routers, the sampled experts
  iqr_layers.py quantise --work W --jobs N   # every tensor of layers 0, 1, 7 through the recipe
  iqr_layers.py blocks   --work W            # block outputs, reference vs converted, on the fixed inputs
  iqr_layers.py lens     --work W            # logit-lens KL / top-1 through the real lm_head (also quantises the head, chunk by chunk)
  iqr_layers.py assemble --work W --out docs/ernie55/iqr-v4-layers.json

Memory: tensors are read one at a time, the head in 8192-row chunks; a watcher sums the RSS of this process and its children and stops the run past 15 GiB.
"""
import argparse, concurrent.futures as cf, json, math, os, re, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fp8_stream as fs          # noqa: E402
import iqr_precision as ip       # noqa: E402

CK = Path(os.environ.get("ERNIE_CKPT", Path.home() / "models/ernie55-flash-fp8"))
PROBE = Path(os.environ.get("IQR_PROBE", Path.home() / "scratch/driver6/iqr_probe"))
HOME = Path(os.environ.get("RAD_HOME", Path.home() / "src/ernie/build-ernie55/radiance_home"))
NOSYS = Path(os.environ.get("NOSYS", Path.home() / "scratch/qfn21-e5/nosys"))
RECIPE = Path(os.environ.get("IQR_RECIPE", HERE.parent.parent / "data/recipes/ernie55-iqr-v4.recipe"))
LAYERS = (0, 1, 7)
T = 96
HID = 4096
EPS = 1e-6
BETA, LB = 4.0, 25.0
K_SUB = 3
N_EXPERTS_SAMPLED = 8
RSS_LIMIT_GIB = 15.0
GIB = 2 ** 30

# checkpoint suffix -> logical suffix of the recipe
LOGICAL = {
    "self_attn.linear_q_down_proj.weight": "attn_q_a", "self_attn.linear_q_up_proj.weight": "attn_q_b", "self_attn.linear_kv_proj.weight": "attn_kv_a_mqa", "self_attn.linear_o_group_proj": "attn_o_group",
    "self_attn.q_a_proj.weight": "attn_q_a", "self_attn.q_b_proj.weight": "attn_q_b", "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa", "self_attn.kv_b_proj.weight": "attn_kv_b",
    "self_attn.gate_proj.weight": "attn_gate", "self_attn.o_proj.weight": "attn_output", "self_attn.core_attention.indexer.wq_b.weight": "attn_idx_q_b",
    "mlp.gate_proj.weight": "ffn_gate", "mlp.up_proj.weight": "ffn_up", "mlp.down_proj.weight": "ffn_down",
    "block_sparse_moe.fc1_latent_proj.weight": "ffn_latent_in", "block_sparse_moe.fc2_latent_proj.weight": "ffn_latent_out",
    "block_sparse_moe.shared_experts.w1.weight": "ffn_gate_shexp", "block_sparse_moe.shared_experts.w3.weight": "ffn_up_shexp", "block_sparse_moe.shared_experts.w2.weight": "ffn_down_shexp",
}
EXPERT_LOGICAL = {"w1": "ffn_gate_exps", "w3": "ffn_up_exps", "w2": "ffn_down_exps"}


class Fail(Exception):
    pass


# ------------------------------------------------------------------ memory watcher
class Rss(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0.0
        self.stop = False
        self.tripped = False

    @staticmethod
    def tree_rss_kib(root):
        kids = {}
        rss = {}
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                st = open(f"/proc/{p}/stat").read()
                ppid = int(st[st.rindex(")") + 2:].split()[1])
                kids.setdefault(ppid, []).append(int(p))
                for ln in open(f"/proc/{p}/status"):
                    if ln.startswith("VmRSS:"):
                        rss[int(p)] = int(ln.split()[1])
            except (OSError, ValueError, IndexError):
                pass
        tot, stack = 0, [root]
        while stack:
            q = stack.pop()
            tot += rss.get(q, 0)
            stack.extend(kids.get(q, []))
        return tot

    def run(self):
        while not self.stop:
            g = self.tree_rss_kib(os.getpid()) / 2 ** 20
            self.peak = max(self.peak, g)
            if g > RSS_LIMIT_GIB:
                self.tripped = True
                os.kill(os.getpid(), 15)
            time.sleep(1.0)


def memguard():
    try:
        h = json.load(open("/var/lib/memguard/host.json"))
        return h
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ numpy helpers
def rms(x):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)).astype(np.float32)


def sig(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def situ(g, u):
    return (BETA * np.tanh(g / BETA) * sig(g) * (LB * np.tanh(u / LB))).astype(np.float32)


def bf16_round(a):
    u = a.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def kl_rows(lr, lq):
    """KL(softmax(lr) || softmax(lq)) per row, f64 accumulation."""
    a = lr.astype(np.float64)
    b = lq.astype(np.float64)
    a -= a.max(-1, keepdims=True)
    b -= b.max(-1, keepdims=True)
    la = a - np.log(np.exp(a).sum(-1, keepdims=True))
    lb = b - np.log(np.exp(b).sum(-1, keepdims=True))
    return (np.exp(la) * (la - lb)).sum(-1)


def relerr(a, b):
    return float(np.linalg.norm((a - b).astype(np.float64)) / max(np.linalg.norm(a.astype(np.float64)), 1e-30))


def cosine(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


# ------------------------------------------------------------------ the checkpoint
def ckpt():
    return fs.Checkpoint(CK)


def read_f32(ck, name):
    dt, shape, buf = ck.raw(name)
    n = int(np.prod(shape))
    return fs.widen(dt, buf, n).reshape(shape)


def config():
    return json.loads((CK / "config.json").read_text())


def layer_kind(L):
    r = config()["compress_ratios"][L]
    return "hca" if r == 128 else "mla"


def tokens():
    x, out = 12345, []
    while len(out) < T:
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(int(x % 200000) + 100)
    return out


def embed_rows(ck, ids):
    rows = []
    for t in ids:
        _, _, b = ck.raw("model.embed_tokens.weight", t, 1)
        rows.append(fs.widen("BF16", b, HID))
    return np.stack(rows).astype(np.float32)


# ------------------------------------------------------------------ recipe
def recipe_rules():
    return ip.parse_recipe(RECIPE)


def opts_of(rule):
    o = dict(rule["opts"])
    o.pop("calib", None)        # no ERNIE Hessians exist: the identity Hessian
    return o


def logical_of(L, suffix):
    if suffix in LOGICAL:
        return f"blk.{L}.{LOGICAL[suffix]}.weight"
    m = re.match(r"block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$", suffix)
    if m:
        return f"blk.{L}.{EXPERT_LOGICAL[m.group(2)]}.{m.group(1)}.weight"
    return None


# ------------------------------------------------------------------ stage: route
def stage_route(work):
    ck = ckpt()
    ids = tokens()
    emb = embed_rows(ck, ids)
    x0 = rms(emb)
    out = {"tokens": ids, "layers": {}}
    for L in (1, 7):
        p = f"model.layers.{L}"
        gain = read_f32(ck, f"{p}.post_attention_layernorm.weight")
        xin = x0 * gain
        r0 = xin @ read_f32(ck, f"{p}.block_sparse_moe.gate.weight").T
        r1 = xin @ read_f32(ck, f"{p}.block_sparse_moe.gate.weight_1").T
        bias = read_f32(ck, f"{p}.block_sparse_moe.e_score_correction_bias")
        score = sig(r0) + sig(r1)
        top = np.argsort(-(score + bias), axis=1, kind="stable")[:, :10]
        cnt = np.bincount(top.ravel(), minlength=512)
        S = [int(e) for e in np.argsort(-cnt, kind="stable")[:N_EXPERTS_SAMPLED]]
        out["layers"][str(L)] = {"experts": sorted(S), "top10_hits_in_sample": float(np.isin(top, S).mean()), "distinct_experts_routed": int((cnt > 0).sum())}
    ck.close()
    Path(work).mkdir(parents=True, exist_ok=True)
    (Path(work) / "route.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["layers"]))


# ------------------------------------------------------------------ stage: quantise
def task_list(work):
    route = json.loads((Path(work) / "route.json").read_text())
    rules = recipe_rules()
    cfg = config()
    tasks = []
    for L in LAYERS:
        p = f"model.layers.{L}"
        suffixes = []
        kind = layer_kind(L)
        if kind == "hca":
            suffixes += ["self_attn.linear_q_down_proj.weight", "self_attn.linear_q_up_proj.weight", "self_attn.linear_kv_proj.weight", "self_attn.gate_proj.weight", "self_attn.o_proj.weight", "self_attn.linear_o_group_proj"]
        else:
            suffixes += ["self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight", "self_attn.gate_proj.weight", "self_attn.o_proj.weight", "self_attn.core_attention.indexer.wq_b.weight"]
        if L < cfg["first_k_dense_replace"]:
            suffixes += ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
        else:
            suffixes += ["block_sparse_moe.fc1_latent_proj.weight", "block_sparse_moe.fc2_latent_proj.weight", "block_sparse_moe.shared_experts.w1.weight", "block_sparse_moe.shared_experts.w3.weight", "block_sparse_moe.shared_experts.w2.weight"]
            for e in route["layers"][str(L)]["experts"]:
                suffixes += [f"block_sparse_moe.experts.{e}.{w}.weight" for w in ("w1", "w3", "w2")]
        for s in suffixes:
            lg = logical_of(L, s)
            r = ip.match(rules, lg)
            if r is None:
                raise Fail(f"{lg}: no recipe rule")
            tasks.append(dict(layer=L, name=f"{p}.{s}", logical=lg, rule=r["line"].strip(), quantiser=r["quantiser"], opts=opts_of(r), precision=ip.rule_precision(r)))
        if L >= cfg["first_k_dense_replace"]:      # the router alternative: IQ8R (the rule of the attention output), measured only
            r = ip.match(rules, f"blk.{L}.attn_output.weight")
            for g in ("gate.weight", "gate.weight_1"):
                tasks.append(dict(layer=L, name=f"{p}.block_sparse_moe.{g}", logical=f"blk.{L}.ffn_gate_inp.{g}", rule="(alternative) " + r["line"].strip(), quantiser=r["quantiser"], opts=opts_of(r), precision="router_iq8r_alt", alt=True))
    return tasks


def collect(work):
    """The per-tensor records of the CURRENT task set (the quantise stage may run in several processes; a record of another identity, a stale file in the work directory or a tensor that is
    no longer sampled is never read): every task must have a record bound to its identity."""
    ck = ckpt()
    out = []
    try:
        for t in task_list(work):
            f = Path(work) / f"L{t['layer']}" / (safe(t["name"]) + ".json")
            if not f.exists():
                raise Fail(f"no record for {t['name']}: run the quantise stage")
            shard, info, _ = ck.locate(t["name"])
            r = json.loads(f.read_text())
            if r.get("identity") != identity(t, shard, info, ck):
                raise Fail(f"{t['name']}: the record is bound to another recipe rule, options or checkpoint bytes: rerun quantise")
            out.append(r)
    finally:
        ck.close()
    return sorted(out, key=lambda r: r["name"])


def direct_inputs_fp():
    """The checkpoint bytes the block and lens stages read directly (not through a converted tensor): the sampled embedding rows, every norm gain, router weight, bias, scale, latent norm and the
    indexer's kept tensors of the three layers, the final norm; and the size and mtime of the shard that holds lm_head."""
    import hashlib, re as _re
    ck = ckpt()
    h = hashlib.sha256()
    try:
        ids = tokens()
        xs = 777
        for _ in range(2304):
            xs = (xs * 1664525 + 1013904223) & 0xFFFFFFFF
            ids.append(int(xs % 200000) + 100)
        for t in ids:
            h.update(ck.raw("model.embed_tokens.weight", t, 1)[2])
        rx = _re.compile(r"layernorm|\.norm\.|k_norm|gate\.weight(_1)?$|e_score|routed_scaling|indexer\.(wk|weights_proj)|latent_norm|q_layernorm|kv_layernorm")
        for L in LAYERS:
            for name in sorted(n for n in ck.map if n.startswith(f"model.layers.{L}.") and rx.search(n) and not n.endswith("_scale_inv")):
                h.update(name.encode())
                h.update(ck.raw(name)[2])
        h.update(ck.raw("model.norm.weight")[2])
        shard = ck.map["lm_head.weight"]
        st = os.stat(CK / shard)
        h.update(f"{shard}:{st.st_size}:{st.st_mtime_ns}".encode())
    finally:
        ck.close()
    return h.hexdigest()[:24]


def inputs_stamp(work):
    """What the block and lens artifacts depend on: every current task's identity (rule, options, checkpoint bytes), the routing sample, the head's rule."""
    import hashlib
    ck = ckpt()
    ids = []
    try:
        for t in task_list(work):
            shard, info, _ = ck.locate(t["name"])
            ids.append(identity(t, shard, info, ck))
    finally:
        ck.close()
    route = json.loads((Path(work) / "route.json").read_text())
    head = ip.match(recipe_rules(), "output.weight")
    return hashlib.sha256(json.dumps([sorted(ids), route, head["line"], direct_inputs_fp(), "blocks-lens/4"], sort_keys=True).encode()).hexdigest()[:24]


def safe(n):
    return n.replace("/", "_")


_FP = {}


def payload_fp(ck, name):
    """sha256 of a tensor's bytes and, for block FP8, of its scale plane: a refreshed checkpoint with the same header but other payload changes the identity."""
    import hashlib
    if name not in _FP:
        h = hashlib.sha256()
        _, _, b = ck.raw(name)
        h.update(b)
        try:
            _, _, sb = ck.raw(name + "_scale_inv")
            h.update(sb)
        except fs.Fail:
            pass
        _FP[name] = h.hexdigest()[:24]
    return _FP[name]


def identity(t, shard, info, ck):
    """What a cached per-tensor record is bound to: the tensor's place and bytes in the checkpoint (a fingerprint of the payload and the scale plane), the recipe rule that converted it, its options and the quantiser."""
    import hashlib
    return hashlib.sha256(json.dumps([t["name"], t["quantiser"], sorted(t["opts"].items()), t["rule"], shard, info["data_offsets"], info["dtype"], info["shape"], payload_fp(ck, t["name"]), "iqr_layers/3"], sort_keys=True).encode()).hexdigest()[:24]


def run_task(work, t, omp):
    wdir = Path(work) / f"L{t['layer']}"
    wdir.mkdir(parents=True, exist_ok=True)
    key = safe(t["name"])
    res_f = wdir / (key + ".json")
    ck = ckpt()
    try:
        shard, info, _ = ck.locate(t["name"])
        ident = identity(t, shard, info, ck)
        if res_f.exists():
            cached = json.loads(res_f.read_text())
            if cached.get("identity") == ident:
                return cached
            print(f"  {t['name']}: cached record is for another task identity (recipe rule, options or checkpoint bytes changed): recomputed", flush=True)
        dt, shape = info["dtype"], info["shape"]
        ref = wdir / (key + ".ref.f32")
        if dt == "F8_E4M3":
            with open(ref, "wb") as fh:
                N, K, tb, sb = fs.dequant_rows(ck, t["name"], fh)
            bytes_src = tb + sb
        else:
            _, _, buf = ck.raw(t["name"])
            a = fs.widen(dt, buf, int(np.prod(shape))).reshape(shape)
            a.astype("<f4").tofile(ref)
            N, K = shape
            bytes_src = len(buf)
    finally:
        ck.close()
    rec = dict(identity=ident, name=t["name"], layer=t["layer"], logical=t["logical"], class_name=re.sub(r"(?<=layers\.)\d+", "L", re.sub(r"(?<=experts\.)\d+", "E", t["name"])), shape=[N, K], precision=t["precision"], rule=t["rule"], bytes_src=bytes_src)
    if t["quantiser"] == "cast":
        w = np.fromfile(ref, dtype="<f4").reshape(N, K)
        if t["opts"]["dtype"] == "bf16":
            d = bf16_round(w)
            rec.update(bytes_dst=N * K * 2, rel_err_weight=relerr(w, d), seconds=0.0, planes=[N * K * 2])
            d.astype("<f4").tofile(wdir / (key + ".dec.f32"))
        else:
            raise Fail("only cast dtype=bf16 is measured per tensor (the f32 mHC matrices are kept as shipped)")
    else:
        args = [str(NOSYS), str(PROBE), "--in", str(ref), "--rows", str(N), "--cols", str(K), "--name", t["logical"], "--dec-out", str(wdir / (key + ".dec.f32")), "--home", str(HOME), "--", t["quantiser"]] + [f"{k}={v}" for k, v in t["opts"].items()]
        env = dict(os.environ, OMP_NUM_THREADS=str(omp))
        t0 = time.time()
        r = subprocess.run(args, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise Fail(f"probe failed for {t['name']}: {(r.stdout + r.stderr)[-600:]}")
        rep = json.loads(r.stdout.strip().split("\n")[-1])
        rec.update(bytes_dst=rep["total_bytes"], rel_err_weight=rep["rel_err"], max_abs_err=rep["max_abs_err"], seconds=rep["seconds"], wall_seconds=round(time.time() - t0, 1), planes=rep["plane_bytes"])
    rec["bpw"] = rec["bytes_dst"] * 8 / (N * K)
    res_f.write_text(json.dumps(rec))
    print(f"  {t['name']:78s} {t['precision']:8s} rel {rec['rel_err_weight']:.5f} bpw {rec['bpw']:.3f} {rec.get('wall_seconds', 0)}s", flush=True)
    return rec


def stage_quantise(work, jobs, only=None):
    h = memguard()
    if h and h.get("host_available_gib", 99) < 30:
        raise Fail(f"memguard: only {h['host_available_gib']} GiB available")
    tasks = task_list(work)
    if only:
        tasks = [t for t in tasks if re.search(only, t["name"])]
    # experts first (long), then the rest; trellis jobs get as many OpenMP threads as they have runs, capped
    def omp_for(t):
        if t["quantiser"] == "gptq":
            return max(1, min(8, 2048 // int(t["opts"].get("tcq_rows", "256"))))
        return 4
    tasks.sort(key=lambda t: (0 if t["quantiser"] == "gptq" else 1))
    out = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_task, work, t, omp_for(t)): t for t in tasks}
        for f in cf.as_completed(futs):
            out.append(f.result())
    (Path(work) / ("tensors.json" if not only else "tensors.part.json")).write_text(json.dumps(sorted(out, key=lambda r: r["name"]), indent=1))


# ------------------------------------------------------------------ stage: blocks
def load(work, L, name, which):
    p = Path(work) / f"L{L}" / (safe(name) + f".{which}.f32")
    return None if not p.exists() else np.fromfile(p, dtype="<f4")


def mat(work, L, name, shape, which):
    a = load(work, L, name, which)
    if a is None:
        raise Fail(f"missing {name} .{which}")
    return a.reshape(shape)


def stage_blocks(work):
    ck = ckpt()
    route = json.loads((Path(work) / "route.json").read_text())
    ids = route["tokens"]
    x0 = rms(embed_rows(ck, ids))
    rng = np.random.default_rng(20261123)
    recs = collect(work)
    shp = {r["name"]: r["shape"] for r in recs}
    blocks, lens = [], {}

    def M(L, suffix, which):
        n = f"model.layers.{L}.{suffix}"
        return mat(work, L, n, shp[n], which)

    def add(name, kind, domain, ref, q, extra=None, lens_hidden=None):
        b = dict(name=name, kind=kind, tokens=T, metric_domain=domain, rel_err_out=relerr(ref, q), cos_out=cosine(ref, q))
        if domain == "channel_softmax":
            r = ref.reshape(T, -1, extra["chan"])
            qq = q.reshape(T, -1, extra["chan"])
            kl = kl_rows(r.reshape(-1, extra["chan"]), qq.reshape(-1, extra["chan"]))
            b.update(kld_mean=float(kl.mean()), kld_p99=float(np.quantile(kl, 0.99)), top1=float((r.argmax(-1) == qq.argmax(-1)).mean()), heads=int(r.shape[1]))
            b["note"] = f"softmax over the {extra['chan']} channels of each of {r.shape[1]} heads, every token"
        if extra:
            b.update({k: v for k, v in extra.items() if k != "chan"})
        blocks.append(b)
        if lens_hidden is not None:
            lens[name] = lens_hidden
        return b

    mn = read_f32(ck, "model.norm.weight")

    def hidden(y):
        return (rms(y) * mn).astype(np.float32)

    # ---- layer 0: dense feed-forward
    gain = read_f32(ck, "model.layers.0.post_attention_layernorm.weight")
    xin = x0 * gain
    def ffn(which):
        g = xin @ M(0, "mlp.gate_proj.weight", which).T
        u = xin @ M(0, "mlp.up_proj.weight", which).T
        return situ(g, u) @ M(0, "mlp.down_proj.weight", which).T
    yr, yq = ffn("ref"), ffn("dec")
    add("L0.dense_ffn", "dense_ffn", "logit_lens_vocab", yr, yq, dict(precision="iq8r gate/up/down"), (hidden(yr), hidden(yq)))

    # ---- routed layers
    for L in (1, 7):
        p = f"model.layers.{L}"
        gain = read_f32(ck, f"{p}.post_attention_layernorm.weight")
        xin = x0 * gain
        bias = read_f32(ck, f"{p}.block_sparse_moe.e_score_correction_bias")
        rsf = read_f32(ck, f"{p}.block_sparse_moe.gate.routed_scaling_factor_param")
        lnorm = read_f32(ck, f"{p}.block_sparse_moe.latent_norm.weight")
        w0 = read_f32(ck, f"{p}.block_sparse_moe.gate.weight")
        w1 = read_f32(ck, f"{p}.block_sparse_moe.gate.weight_1")
        r_ref = (sig(xin @ w0.T) + sig(xin @ w1.T))
        d0 = mat(work, L, f"{p}.block_sparse_moe.gate.weight", [512, HID], "dec")
        d1 = mat(work, L, f"{p}.block_sparse_moe.gate.weight_1", [512, HID], "dec")
        r_q = (sig(xin @ d0.T) + sig(xin @ d1.T))
        top_r = np.argsort(-(r_ref + bias), axis=1, kind="stable")[:, :10]
        top_q = np.argsort(-(r_q + bias), axis=1, kind="stable")[:, :10]
        agree = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top_r, top_q)]))
        pr = r_ref / r_ref.sum(1, keepdims=True)
        pq = r_q / r_q.sum(1, keepdims=True)
        klr = (pr * (np.log(pr) - np.log(pq))).sum(1)
        b = dict(name=f"L{L}.router_iq8r", kind="router", tokens=T, metric_domain="router_topk", rel_err_out=relerr(r_ref, r_q), cos_out=cosine(r_ref, r_q),
                 kld_mean=float(klr.mean()), kld_p99=float(np.quantile(klr, 0.99)), top1=float((r_ref.argmax(1) == r_q.argmax(1)).mean()),
                 topk_set_agreement=agree, top_k=10, note="the routers are BF16 in the map; this is the IQ8R alternative (bf16 vs int8 rotated), sigmoid scores normalised over the 512 experts for the KL")
        blocks.append(b)
        # MoE block among the sampled experts, routed with the real (reference) routers
        S = route["layers"][str(L)]["experts"]
        sub = r_ref[:, S] + bias[S]
        order = np.argsort(-sub, axis=1, kind="stable")[:, :K_SUB]
        ids_t = np.array(S)[order]
        wts = np.take_along_axis(r_ref[:, S], order, axis=1)
        wts = wts / np.maximum(wts.sum(1, keepdims=True), 1e-20)
        wts = wts * rsf[ids_t]

        def moe(which):
            xl = xin @ M(L, "block_sparse_moe.fc1_latent_proj.weight", which).T
            comb = np.zeros((T, 2048), np.float32)
            for e in S:
                W1 = mat(work, L, f"{p}.block_sparse_moe.experts.{e}.w1.weight", [2048, 2048], which)
                W3 = mat(work, L, f"{p}.block_sparse_moe.experts.{e}.w3.weight", [2048, 2048], which)
                W2 = mat(work, L, f"{p}.block_sparse_moe.experts.{e}.w2.weight", [2048, 2048], which)
                tt, jj = np.nonzero(ids_t == e)
                if len(tt) == 0:
                    continue
                a = situ(xl[tt] @ W1.T, xl[tt] @ W3.T) @ W2.T
                np.add.at(comb, tt, wts[tt, jj][:, None] * a)
            nl = rms(comb) * lnorm
            ro = nl @ M(L, "block_sparse_moe.fc2_latent_proj.weight", which).T
            sy = situ(xin @ M(L, "block_sparse_moe.shared_experts.w1.weight", which).T, xin @ M(L, "block_sparse_moe.shared_experts.w3.weight", which).T) @ M(L, "block_sparse_moe.shared_experts.w2.weight", which).T
            return ro + sy
        yr, yq = moe("ref"), moe("dec")
        add(f"L{L}.moe_block", "moe_block", "logit_lens_vocab", yr, yq,
            dict(experts_in_block=len(S), top_k=K_SUB, experts=S, precision="experts iqr_v4 (uncalibrated), latent + shared iq8r", note=f"routed among the {len(S)} sampled experts ({K_SUB} per token) with the real f32 routers; the full layer routes 10 of 512"), (hidden(yr), hidden(yq)))

    # ---- attention projection paths
    for L in LAYERS:
        p = f"model.layers.{L}"
        kind = layer_kind(L)
        gain = read_f32(ck, f"{p}.input_layernorm.weight")
        xa = x0 * gain
        if kind == "hca":
            qd, qu, kv, gt, op = "self_attn.linear_q_down_proj.weight", "self_attn.linear_q_up_proj.weight", "self_attn.linear_kv_proj.weight", "self_attn.gate_proj.weight", "self_attn.o_proj.weight"
            qln = read_f32(ck, f"{p}.self_attn.q_layernorm.weight")
            chan = 512
        else:
            qd, qu, kv, gt, op = "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight", "self_attn.gate_proj.weight", "self_attn.o_proj.weight"
            qln = read_f32(ck, f"{p}.self_attn.q_a_layernorm.weight")
            chan = 256
        out = {}
        for which in ("ref", "dec"):
            qa = xa @ M(L, qd, which).T
            qn = rms(qa) * qln
            q = qn @ M(L, qu, which).T
            kvv = xa @ M(L, kv, which).T
            g = sig(qn @ M(L, gt, which).T)
            z = np.random.default_rng(777 + L).standard_normal(g.shape).astype(np.float32)
            o = (g * z) @ M(L, op, which).T
            out[which] = dict(q=q, kv=kvv, o=o, qn=qn)
            if kind == "mla":
                out[which]["iq"] = qn @ M(L, "self_attn.core_attention.indexer.wq_b.weight", which).T
        tag = "hca" if kind == "hca" else "mla"
        add(f"L{L}.{tag}_q_path", "attn_projections", "channel_softmax", out["ref"]["q"], out["dec"]["q"], dict(chan=chan, precision="iq8r q_down/q_up"))
        add(f"L{L}.{tag}_kv_proj", "attn_projections", "channel_softmax", out["ref"]["kv"], out["dec"]["kv"], dict(chan=out["ref"]["kv"].shape[-1], precision="iq8r kv"))
        add(f"L{L}.{tag}_o_path", "attn_projections", "logit_lens_vocab", out["ref"]["o"], out["dec"]["o"], dict(precision="iq8r gate + o_proj", note="o_proj input = sigmoid(gate_proj(q_norm)) times a fixed unit-normal vector: the real attention mix is not available here"),
            (hidden(out["ref"]["o"]), hidden(out["dec"]["o"])))
        if kind == "mla":
            add(f"L{L}.mla_indexer_q", "attn_projections", "channel_softmax", out["ref"]["iq"], out["dec"]["iq"], dict(chan=128, precision="iq8r wq_b"))
        # ---- the attention mix: softmax over the real q and kv, the grouped wo_a (HCA) or the absorbed kv_b (MLA), the gate and wo_b, built from the layer's own weights.
        # Approximation, stated: no rotary, no sink, no VHA post-mix; T = 96 < 128 (window) and < 2048 (top-k), so every past token is visible and the indexer's selection is the identity here.
        z_rng = np.random.default_rng(1000 + L)
        if kind == "hca":
            kvln = read_f32(ck, f"{p}.self_attn.kv_layernorm.weight")
            mixed = {}
            for which in ("ref", "dec"):
                qn_ = out[which]["qn"]
                q = (qn_ @ M(L, qu, which).T).reshape(T, 64, 512)
                kvn = rms(out[which]["kv"]) * kvln
                sc = np.einsum("thd,sd->ths", q, kvn) / np.sqrt(512.0)
                sc = np.where(np.tril(np.ones((T, T), bool))[:, None, :], sc, -1e30)
                P = np.exp(sc - sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
                att = np.einsum("ths,sd->thd", P, kvn).reshape(T, 8, 4096)
                wo = M(L, "self_attn.linear_o_group_proj", which)
                hwo = np.concatenate([att[:, g] @ wo[g * 1024:(g + 1) * 1024].T for g in range(8)], axis=1)
                mixed[which] = ((sig(qn_ @ M(L, gt, which).T) * hwo) @ M(L, op, which).T)
            add(f"L{L}.hca_attention_mix", "attn_block", "logit_lens_vocab", mixed["ref"], mixed["dec"], dict(precision="iq8r q/kv/gate/o_proj, wo_a bf16", note="causal softmax attention over the real q and the single kv head (k = v), grouped wo_a, sigmoid gate, wo_b; no rotary, sink or VHA; T=96 < window 128"),
                (hidden(mixed["ref"]), hidden(mixed["dec"])))
        else:
            kvaln = read_f32(ck, f"{p}.self_attn.kv_a_layernorm.weight")
            mixed = {}
            for which in ("ref", "dec"):
                qn_ = out[which]["qn"]
                q = (qn_ @ M(L, qu, which).T).reshape(T, 64, 256)
                lat = rms(out[which]["kv"][:, :512]) * kvaln
                kvb = M(L, "self_attn.kv_b_proj.weight", which).reshape(64, 448, 512)
                qabs = np.einsum("thd,hdc->thc", q[:, :, :192], kvb[:, :192, :])
                sc = np.einsum("thc,sc->ths", qabs, lat) / np.sqrt(256.0)
                sc = np.where(np.tril(np.ones((T, T), bool))[:, None, :], sc, -1e30)
                P = np.exp(sc - sc.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
                v = np.einsum("thc,hvc->thv", np.einsum("ths,sc->thc", P, lat), kvb[:, 192:, :]).reshape(T, 16384)
                mixed[which] = ((sig(qn_ @ M(L, gt, which).T) * v) @ M(L, op, which).T)
            add(f"L{L}.mla_attention_mix", "attn_block", "logit_lens_vocab", mixed["ref"], mixed["dec"], dict(precision="iq8r q/kv_a/gate/o_proj, kv_b bf16", note="absorbed latent attention over the real q_b, kv_a and kv_b, sigmoid gate, wo_b; no rotary tail; T=96 < top-k 2048 so the selection is everything"),
                (hidden(mixed["ref"]), hidden(mixed["dec"])))
    # ---- the DSA indexer's selection on an MLA layer: top-2048 of a causal score row at 2304 positions (the last 192 rows see more than 2048 keys)
    L = 7
    p = f"model.layers.{L}"
    T2, R2, KTOP = 2304, 192, 2048
    x2, ids2 = [], []
    xs = 777
    for _ in range(T2):
        xs = (xs * 1664525 + 1013904223) & 0xFFFFFFFF
        ids2.append(int(xs % 200000) + 100)
    xa2 = rms(embed_rows(ck, ids2)) * read_f32(ck, f"{p}.input_layernorm.weight")
    qln = read_f32(ck, f"{p}.self_attn.q_a_layernorm.weight")
    wk = read_f32(ck, f"{p}.self_attn.core_attention.indexer.wk.weight")
    kk = xa2 @ wk.T
    kg = read_f32(ck, f"{p}.self_attn.core_attention.indexer.k_norm.weight"); kb = read_f32(ck, f"{p}.self_attn.core_attention.indexer.k_norm.bias")
    kk = ((kk - kk.mean(-1, keepdims=True)) / np.sqrt(kk.var(-1, keepdims=True) + 1e-6)) * kg + kb
    wh = xa2 @ read_f32(ck, f"{p}.self_attn.core_attention.indexer.weights_proj.weight").T
    sc2 = {}
    for which in ("ref", "dec"):
        qn2 = rms(xa2 @ M(L, "self_attn.q_a_proj.weight", which).T) * qln
        iq2 = (qn2 @ M(L, "self_attn.core_attention.indexer.wq_b.weight", which).T).reshape(T2, 64, 128)
        rows = []
        for t in range(T2 - R2, T2):
            s_ = np.maximum(np.einsum("hd,sd->hs", iq2[t], kk[:t + 1]), 0.0)
            rows.append((wh[t][:, None] * s_).sum(0))
        sc2[which] = rows
    ov, kl_, t1 = [], [], []
    for a_, b_ in zip(sc2["ref"], sc2["dec"]):
        ta, tb = set(np.argpartition(-a_, KTOP)[:KTOP].tolist()), set(np.argpartition(-b_, KTOP)[:KTOP].tolist())
        ov.append(len(ta & tb) / KTOP)
        kl_.append(float(kl_rows(a_[None, :], b_[None, :])[0]))
        t1.append(float(a_.argmax() == b_.argmax()))
    flat_r = np.concatenate([np.pad(r, (0, T2 - len(r))) for r in sc2["ref"]]); flat_q = np.concatenate([np.pad(r, (0, T2 - len(r))) for r in sc2["dec"]])
    blocks.append(dict(name="L7.dsa_indexer_selection", kind="indexer_selection", tokens=R2, metric_domain="router_topk", rel_err_out=relerr(flat_r, flat_q), cos_out=cosine(flat_r, flat_q), kld_mean=float(np.mean(kl_)), kld_p99=float(np.quantile(kl_, 0.99)),
                       top1=float(np.mean(t1)), topk_set_agreement=float(np.mean(ov)), top_k=KTOP, positions=T2, note="indexer scores sum_h w_h relu(q_h . k) over the real wq_b (IQ8R vs FP8-dequant), real wk, k_norm and weights_proj (kept bf16); no rotary or Hadamard or FP8 key quantiser (4-1 rule); the last 192 positions of 2304 fixed tokens; KL of softmax(score row)"))
    ck.close()
    names = list(lens)
    H = np.concatenate([np.concatenate([lens[n][0], lens[n][1]]) for n in names]).astype(np.float32)
    np.save(Path(work) / "lens_H.npy", H)
    (Path(work) / "blocks.json").write_text(json.dumps(dict(blocks=blocks, lens_order=names, inputs=inputs_stamp(work)), indent=1))
    print(json.dumps([(b["name"], round(b["rel_err_out"], 5)) for b in blocks]))


# ------------------------------------------------------------------ stage: lens (through the real head; also the head's own quantisation)
def stage_lens(work):
    ck = ckpt()
    B = json.loads((Path(work) / "blocks.json").read_text())
    if B.get("inputs") != inputs_stamp(work):
        raise Fail("blocks.json was produced from other tensor records, routing sample or head rule than the current ones: rerun the blocks stage")
    H = np.load(Path(work) / "lens_H.npy")
    names = B["lens_order"]
    R = H.shape[0]
    shape = ck.locate("lm_head.weight")[1]["shape"]
    V = shape[0]
    rules = recipe_rules()
    hrule = ip.match(rules, "output.weight")
    ho = opts_of(hrule)
    Hhead = H[:T]          # the head block: the first lens block's reference hidden (layer-0 FFN)
    logits = np.zeros((R, V), np.float32)
    logits_qhead = np.zeros((T, V), np.float32)
    head_bytes, head_err_num, head_err_den, head_secs = 0, 0.0, 0.0, 0.0
    CH = 8192
    wdir = Path(work) / "head"
    wdir.mkdir(exist_ok=True)
    for r0 in range(0, V, CH):
        n = min(CH, V - r0)
        _, _, buf = ck.raw("lm_head.weight", r0, n)
        W = fs.widen("BF16", buf, n * HID).reshape(n, HID)
        logits[:, r0:r0 + n] = H @ W.T
        W.astype("<f4").tofile(wdir / "c.f32")
        r = subprocess.run([str(NOSYS), str(PROBE), "--in", str(wdir / "c.f32"), "--rows", str(n), "--cols", str(HID), "--name", "output.weight", "--dec-out", str(wdir / "d.f32"), "--home", str(HOME), "--", hrule["quantiser"]] + [f"{k}={v}" for k, v in ho.items()],
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="16"))
        if r.returncode != 0:
            raise Fail("head probe: " + (r.stdout + r.stderr)[-500:])
        rep = json.loads(r.stdout.strip().split("\n")[-1])
        D = np.fromfile(wdir / "d.f32", dtype="<f4").reshape(n, HID)
        logits_qhead[:, r0:r0 + n] = Hhead @ D.T
        head_bytes += rep["total_bytes"]
        head_secs += rep["seconds"]
        head_err_num += (rep["rel_err"] ** 2) * float((W.astype(np.float64) ** 2).sum())
        head_err_den += float((W.astype(np.float64) ** 2).sum())
    for f in ("c.f32", "d.f32"):
        (wdir / f).unlink(missing_ok=True)
    ck.close()
    res = {}
    for i, n in enumerate(names):
        lr, lq = logits[2 * i * T:(2 * i + 1) * T], logits[(2 * i + 1) * T:(2 * i + 2) * T]
        kl = kl_rows(lr, lq)
        res[n] = dict(kld_mean=float(kl.mean()), kld_p99=float(np.quantile(kl, 0.99)), top1=float((lr.argmax(1) == lq.argmax(1)).mean()), logit_rel_err=relerr(lr, lq))
    lr = logits[:T]
    kl = kl_rows(lr, logits_qhead)
    head = dict(name="lm_head_iq8r", kind="lm_head", tokens=T, metric_domain="logit_lens_vocab", kld_mean=float(kl.mean()), kld_p99=float(np.quantile(kl, 0.99)), top1=float((lr.argmax(1) == logits_qhead.argmax(1)).mean()),
                rel_err_out=relerr(lr, logits_qhead), cos_out=cosine(lr, logits_qhead), precision="iq8r head (rtn i8 group 128, bf16 scale)", note="bf16 head vs the same head quantised by the recipe's output.weight rule, on the layer-0 FFN branch hidden")
    for b in B["blocks"]:
        if b["name"] in res:
            r = res[b["name"]]
            b.update(kld_mean=r["kld_mean"], kld_p99=r["kld_p99"], top1=r["top1"], logit_rel_err=r["logit_rel_err"])
    B["blocks"].append(head)
    B["head"] = dict(bytes_dst=head_bytes, rows=V, bpw=head_bytes * 8 / (V * HID), rel_err_weight=math.sqrt(head_err_num / head_err_den), seconds=head_secs, bytes_src=V * HID * 2)
    B["lens_inputs"] = inputs_stamp(work)
    (Path(work) / "blocks.json").write_text(json.dumps(B, indent=1))
    print(json.dumps({b["name"]: (b.get("kld_mean"), b.get("top1")) for b in B["blocks"]}))


# ------------------------------------------------------------------ stage: ladder (the width ladder on one real expert: what the other widths cost)
LADDER = [
    ("iqr_v4_u4", "gptq", "codes=u4 tcq=16 tcq_code=mad tcq_rows={rows} block=1x* scale=f32 transform=fwht128 tcq_gain=1.12 damp=0.01 tcq_order=1"),
    ("iqr_v4_u5", "gptq", "codes=u5 tcq=16 tcq_code=mad tcq_rows={rows} block=1x* scale=f32 transform=fwht128 tcq_gain=1.16 damp=0.01 tcq_order=1"),
    ("iqr_v4_u6", "gptq", "codes=u6 tcq=16 tcq_code=mad tcq_rows={rows} block=1x* scale=f32 transform=fwht128 tcq_gain=1.24 damp=0.01 tcq_order=1"),
    ("iq8r_group128_moe_row", "rtn", "codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search"),
]


def stage_ladder(work, jobs):
    wd = Path(work) / "ladder"
    wd.mkdir(exist_ok=True)
    out_f = wd / "ladder.json"
    ck = ckpt()
    tasks = []
    for w in ("w1", "w3", "w2"):
        name = f"model.layers.1.block_sparse_moe.experts.22.{w}.weight"
        ref = wd / (w + ".ref.f32")
        if not ref.exists():
            with open(ref, "wb") as fh:
                fs.dequant_rows(ck, name, fh)
        for tag, qn, opts in LADDER:
            tasks.append((w, tag, qn, opts.format(rows=256 if w == "w2" else 512), ref))
    ck.close()

    def one(t):
        w, tag, qn, opts, ref = t
        args = [str(NOSYS), str(PROBE), "--in", str(ref), "--rows", "2048", "--cols", "2048", "--name", f"blk.1.ffn_{ {'w1': 'gate', 'w3': 'up', 'w2': 'down'}[w] }_exps.22.weight", "--home", str(HOME), "--", qn] + opts.split()
        r = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="6"))
        if r.returncode != 0:
            raise Fail("ladder probe: " + (r.stdout + r.stderr)[-400:])
        rep = json.loads(r.stdout.strip().split("\n")[-1])
        print(f"  ladder {w} {tag}: rel {rep['rel_err']:.5f} bpw {rep['total_bytes'] * 8 / 2048 / 2048:.3f} {rep['seconds']:.0f}s", flush=True)
        return dict(weight=w, variant=tag, rel_err_weight=rep["rel_err"], bytes=rep["total_bytes"], bpw=rep["total_bytes"] * 8 / 2048 / 2048, seconds=rep["seconds"])
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(one, tasks))
    out_f.write_text(json.dumps(dict(expert="layer 1 expert 22 (uncalibrated)", rows=res), indent=1))


# ------------------------------------------------------------------ stage: assemble
def stage_assemble(work, out, rss_peak):
    classes = json.loads((HERE.parent.parent / "docs/ernie55/checkpoint-classes.json").read_text())
    pm = json.loads((HERE.parent.parent / "docs/ernie55/iqr-v4-precision-map.json").read_text())
    tensors = collect(work)
    for t in tensors:      # the router IQ8R alternative is measured only: kept in the record, flagged, never part of the size
        if t["precision"] == "router_iq8r_alt":
            t["precision"], t["alt"] = "iq8r", True
            t["role"] = "router alternative (measured only: the precision map keeps the routers BF16)"
    B = json.loads((Path(work) / "blocks.json").read_text())
    cur = inputs_stamp(work)
    if B.get("inputs") != cur or B.get("lens_inputs") != cur:
        raise Fail("blocks.json (blocks or lens stage) does not belong to the current tensor records, routing sample and head rule: rerun blocks and lens")
    route = json.loads((Path(work) / "route.json").read_text())
    by_class = {}
    for t in tensors:
        if t.get("alt"):
            continue
        by_class.setdefault((t["class_name"], tuple(t["shape"])), []).append(t)
    prec = {(e["class_name"], tuple(e["shape"])): e["precision"] for e in pm["entries"]}
    iq8r_bpw = np.mean([t["bpw"] for t in tensors if t["precision"] == "iq8r" and not t.get("alt")])
    size = []
    for c in classes["classes"]:
        key = (c["name"], tuple(c["shape"]))
        p = prec[key]
        per = None
        basis, derived = None, None
        elems = int(np.prod(c["shape"])) if c["shape"] else 1
        if "scale_inv" in c["name"]:
            per, basis = 0, "dropped"
        elif key in by_class and p in ("iqr_v4", "iq8r", "bf16") and by_class[key][0]["precision"] == p:
            per = int(round(np.mean([t["bytes_dst"] for t in by_class[key]])))
            basis = "measured"
        elif c["name"] == "lm_head.weight":
            per, basis = elems, "measured"      # replaced by the chunked measurement below
        elif p in ("iqr_v4", "iq8r"):
            if c["name"].endswith(("e_proj.weight", "h_proj.weight")):
                per = int(math.ceil(elems * iq8r_bpw / 8))
                basis, derived = "derived_from_measured", f"mean measured IQ8R bits per weight ({iq8r_bpw:.4f}) x {elems} elements"
            else:
                raise Fail(f"class {c['name']} {c['shape']} at {p} was not measured")
        else:
            width = {"bf16": 2, "f32": 4}[p]
            per, basis = elems * width, "header"
        size.append(dict(class_name=c["name"], shape=c["shape"], count=c["count"], precision=p, bytes_per_tensor=per, total_bytes=per * c["count"], basis=basis, **({"derived_from": derived} if derived else {}),
                         source_bytes_per_tensor=c["bytes"] // c["count"]))
    total = sum(s["total_bytes"] for s in size)
    experts = sum(s["total_bytes"] for s in size if "experts.E" in s["class_name"])
    head = B["head"]
    hs = [s for s in size if s["class_name"] == "lm_head.weight"][0]
    # the head is measured chunk by chunk: replace its derived/header number by the measured total
    adj = head["bytes_dst"] - hs["total_bytes"]
    hs["bytes_per_tensor"], hs["total_bytes"], hs["basis"] = head["bytes_dst"], head["bytes_dst"], "measured"
    total += adj
    src_fp8 = sum(c["bytes"] for c in classes["classes"])
    table_overhead = 66048 * 65552
    pk = json.loads((Path(work) / "rss_peaks.json").read_text())
    rss_peak = max(list(pk.values()) + [rss_peak])
    res = dict(
        schema="ernie55-iqr-v4-layers/1", checkpoint="~/models/ernie55-flash-fp8", streamed=True, whole_checkpoint_loaded=False, peak_rss_gib=round(rss_peak, 2), peak_rss_by_stage=pk,
        sample=dict(layers=list(LAYERS), tokens=T, experts_quantised=sum(len(v["experts"]) for v in route["layers"].values()), experts_by_layer={k: v["experts"] for k, v in route["layers"].items()},
                    layer_kinds={str(L): layer_kind(L) for L in LAYERS}, input="fixed token ids (LCG, seed 12345), real embedding rows, rms-normalised, times the layer's real norm gain"),
        method=dict(quantiser="libquant (rtn / gptq / cast) through tools/ernie55/iqr_probe: the converter's own quantise loop, the core's decoder", recipe="data/recipes/ernie55-iqr-v4.recipe",
                    calibration="none", calibration_note="uncalibrated: identity Hessian (no ERNIE calibration Grams exist); GPTQ-class rules degrade to nearest-codeword trellis search. An upper bound on the error of a calibrated conversion (Radiance's 27B numbers: GPTQ is 20-40% lower than round to nearest)",
                    reference="FP8 block dequantisation in f32 (e4m3 x ue8m0 scale, exact)", kld="KL(softmax(reference logits) || softmax(converted logits)), mean over 96 tokens; logit lens: block output -> rms -> model.norm -> real lm_head (201216 x 4096)",
                    top1="argmax agreement of the same distributions", moe_note="MoE blocks route among the sampled experts (3 of 8 per token) with the real f32 routers, not 10 of 512"),
        ladder=(json.loads((Path(work) / "ladder/ladder.json").read_text()) if (Path(work) / "ladder/ladder.json").exists() else None),
        tensors=tensors, blocks=B["blocks"], head=head, size_by_class=size,
        total_bytes=total, total_gib=total / GIB, experts_bytes=experts, nonexpert_bytes=total - experts, source_fp8_bytes=src_fp8, source_fp8_gib=src_fp8 / GIB,
        trellis_table_overhead_bytes=table_overhead, trellis_table_overhead_note="each trellis weight carries a 65536-byte table plane and 16 bytes of mad constants (66048 weights x 65552 B); it is a pure function of the two constants, so a container that shared it would save this much",
    )
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"total {res['total_gib']:.2f} GiB, experts {experts / GIB:.2f}, non-expert {(total - experts) / GIB:.2f}, source fp8 {res['source_fp8_gib']:.2f}")


def selftest_metrics(a_path, b_path, rows, cols):
    """The metric code on two f32 matrices (the sealed tests compare it with an independent implementation): KL(softmax(a) || softmax(b)) mean, argmax agreement, relative error, cosine."""
    a = np.fromfile(a_path, dtype="<f4").reshape(rows, cols)
    b = np.fromfile(b_path, dtype="<f4").reshape(rows, cols)
    kl = kl_rows(a, b)
    print(json.dumps(dict(kld_mean=float(kl.mean()), top1=float((a.argmax(1) == b.argmax(1)).mean()), rel_err=relerr(a, b), cos=cosine(a, b))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["route", "quantise", "blocks", "lens", "ladder", "assemble", "selftest"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--rows", type=int); ap.add_argument("--cols", type=int)
    ap.add_argument("--only", help="quantise only the tensors whose name matches this regex")
    ap.add_argument("--rss-peak", type=float, default=0.0)
    a = ap.parse_args(argv)
    w = Rss()
    w.start()
    try:
        if a.stage == "route":
            stage_route(a.work)
        elif a.stage == "quantise":
            stage_quantise(a.work, a.jobs, a.only)
        elif a.stage == "blocks":
            stage_blocks(a.work)
        elif a.stage == "lens":
            stage_lens(a.work)
        elif a.stage == "ladder":
            stage_ladder(a.work, a.jobs)
        elif a.stage == "selftest":
            selftest_metrics(a.a, a.b, a.rows, a.cols)
        else:
            stage_assemble(a.work, a.out, a.rss_peak)
    except (Fail, OSError, KeyError, ValueError) as e:
        print(f"iqr_layers: {e}", file=sys.stderr)
        return 2
    finally:
        w.stop = True
        pk = Path(a.work) / "rss_peaks.json"
        try:
            d = json.loads(pk.read_text()) if pk.exists() else {}
        except ValueError:
            d = {}
        d[a.stage] = max(d.get(a.stage, 0), round(w.peak, 2))
        pk.write_text(json.dumps(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
