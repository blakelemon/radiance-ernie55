'use strict';
// The shared ERNIE CPU acceptance harness (with the fixtures beside it), so the DSpark tests are self-contained;
// the only change is that the build directory may be named by RAD_BUILD_DIR.
// Shared helpers for the 2-7 acceptance tests: build the real converter, container inspector,
// engine binary and ERNIE plugin, assemble a private RADIANCE_HOME, convert fixture checkpoints
// through the public rad-convert, serve the container with the real `radiance` binary on the host
// (libref) backend, and ask it for embeddings over HTTP, the public entry point of a pooled program.
const {spawn, spawnSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');

const repo = process.env.ERNIE_PORT_ROOT || process.cwd();
const build = process.env.RAD_BUILD_DIR || path.join(repo, 'build-ernie55');
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'attnfwd-'));
const children = new Set();
process.on('exit', () => {
  for (const c of children) try { c.kill('SIGKILL'); } catch (e) { /* gone */ }
  fs.rmSync(tmpRoot, {recursive: true, force: true});
});

function sh(cmd, args, opts = {}) {
  return spawnSync(cmd, args, {encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, ...opts});
}

const RULES = ['blk.*.attn_hc.fn=cast:dtype=f32', 'blk.*.ffn_hc.fn=cast:dtype=f32', 'head_hc.fn=cast:dtype=f32'];

let built;
function ensureBuilt() {
  if (built) return built;
  let r = sh('cmake', ['-S', repo, '-B', build, '-G', 'Ninja', '-DRAD_WITH_HIP=OFF', '-DRAD_WITH_FFMPEG=OFF'], {timeout: 120000});
  assert.equal(r.status, 0, 'configure must work: ' + r.stderr);
  r = sh('cmake', ['--build', build, '--target', 'rad-convert', 'rad-info', 'radiance', 'libref', 'libquant', 'ernie55', '-j', '8'], {timeout: 2400000});
  assert.equal(r.status, 0, 'converter, inspector, engine, kernels and the ernie55 plugin must build: ' + (r.stdout + r.stderr).slice(-2500));
  const home = path.join(tmpRoot, 'home');
  fs.mkdirSync(path.join(home, 'architectures'), {recursive: true});
  fs.symlinkSync(path.join(build, 'radiance_home', 'kernels'), path.join(home, 'kernels'));
  fs.symlinkSync(path.join(build, 'radiance_home', 'quantizers'), path.join(home, 'quantizers'));
  fs.symlinkSync(path.join(build, 'radiance_home', 'architectures', 'ernie55.so'), path.join(home, 'architectures', 'ernie55.so'));
  built = {home, convert: path.join(build, 'bin', 'rad-convert'), info: path.join(build, 'bin', 'rad-info'), radiance: path.join(build, 'bin', 'radiance')};
  return built;
}

// The public converter. keys: the --set keys (default: weights-only plus the feed-forward program).
let seq = 0;
function convert(dir, {keys = ['ernie55_weights_only=1', 'ernie55_ffn_forward=1'], rules = RULES, extra = []} = {}) {
  const b = ensureBuilt();
  const out = path.join(tmpRoot, 'm' + (seq++) + '.rad');
  const input = path.join(dir, 'model.safetensors');
  const args = ['--home', b.home, input, '--config', dir, '--tokenizer', path.join(dir, 'tokenizer.json'), '-o', out];
  for (const k of keys) args.push('--set', k);
  for (const r of rules) args.push('--quant', r);
  const r = sh(b.convert, [...args, ...extra], {timeout: 300000});
  return {status: r.status, stderr: r.stderr, stdout: r.stdout, out, produced: fs.existsSync(out)};
}

// Serve one container with the real engine on the host backend; resolves {port, log(), stop()}, or
// {failed, log} when the engine stops before it serves.
function serve(rad, {seqs = 4, tokens = 64, len = tokens} = {}) {
  const b = ensureBuilt();
  const dir = fs.mkdtempSync(path.join(tmpRoot, 'srv-'));
  const yaml = path.join(dir, 'serve.yaml');
  fs.writeFileSync(yaml, [
    'server:', '  host: 127.0.0.1', '  port: 0', '  log: info', `  radiance-home: ${b.home}`,
    '  kernels: libref', '  debug-accept-reference-kernels: true', 'cards: all',
    'models:', '  ern:', `    model: ${rad}`, `    max-num-seqs: ${seqs}`,
    `    max-num-batched-tokens: ${tokens}`, `    max-model-len: ${len}`, ''].join('\n'));
  const child = spawn(b.radiance, ['--config', yaml], {
    env: {...process.env, ROCR_VISIBLE_DEVICES: '99', RADIANCE_HOST_VRAM_MIB: '512', OMP_NUM_THREADS: '2', OMP_WAIT_POLICY: 'passive'},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  children.add(child);
  let log = '';
  child.stdout.on('data', d => { log += d; });
  child.stderr.on('data', d => { log += d; });
  return new Promise(resolve => {
    const timer = setTimeout(() => { child.kill('SIGKILL'); resolve({failed: true, log: log + '\n[timeout]'}); }, 90000);
    const tick = setInterval(() => {
      const m = /listening on http:\/\/127\.0\.0\.1:(\d+)/.exec(log);
      if (m && /model ern: ready/.test(log)) {
        clearInterval(tick); clearTimeout(timer);
        resolve({port: Number(m[1]), get log() { return log; }, stop: () => { child.kill('SIGKILL'); children.delete(child); }});
      }
    }, 50);
    child.on('exit', () => { clearInterval(tick); clearTimeout(timer); children.delete(child); resolve({failed: true, log}); });
  });
}

// POST /v1/embeddings with token-id inputs; returns one L2-normalised vector per input.
async function embed(srv, inputs) {
  const res = await fetch(`http://127.0.0.1:${srv.port}/v1/embeddings`, {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({model: 'ern', input: inputs}),
  });
  const body = await res.json();
  assert.equal(res.status, 200, 'embeddings must succeed: ' + JSON.stringify(body));
  return body.data.map(d => d.embedding);
}

// POST /v1/completions with a token-id prompt. Returns {status, body}. logprobs: alternatives asked for.
async function complete(srv, prompt, extra = {}) {
  const res = await fetch(`http://127.0.0.1:${srv.port}/v1/completions`, {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({model: 'ern', prompt, max_tokens: 1, temperature: 0, ...extra}),
  });
  return {status: res.status, body: await res.json()};
}

// What the container reader reports: {name -> {dtype, shape:[...], planes:[Buffer...]}} (as 2-6's harness).
function read(out) {
  const b = ensureBuilt();
  const v = sh(b.info, ['-v', out]);
  assert.equal(v.status, 0, 'rad-info -v must read the container: ' + v.stderr);
  const rows = {};
  for (const line of v.stdout.split('\n')) {
    const m = /^\s{6}(\S+)\s+(\S+)\s+(\d+(?:x\d+)*)\s+[\d.]+ \S+\s+@\d+/.exec(line);
    if (m) rows[m[1]] = {dtype: m[2], shape: m[3].split('x').map(Number), planes: []};
  }
  const p = sh(b.info, ['--planes', '*', out]);
  assert.equal(p.status, 0, 'rad-info --planes must read the container: ' + p.stderr);
  const data = fs.readFileSync(out);
  for (const line of p.stdout.split('\n')) {
    const m = /^(\d+) (\d+) (\S+)$/.exec(line.trim());
    if (m && rows[m[3]]) rows[m[3]].planes.push(data.subarray(Number(m[1]), Number(m[1]) + Number(m[2])));
  }
  return rows;
}

module.exports = {repo, ensureBuilt, convert, serve, embed, complete, read, sh, tmpRoot, RULES};
