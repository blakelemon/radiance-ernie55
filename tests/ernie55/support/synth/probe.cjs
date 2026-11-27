const {spawnSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const repo = process.env.ERNIE_PORT_ROOT || process.cwd();
let probeBin = null;
function command(program, args, options = {}) {
  return spawnSync(program, args, {encoding: 'utf8', timeout: 480000, maxBuffer: 512 * 1024 * 1024, ...options});
}
function plugin() {
  const build = path.join(repo, 'build-ernie55');
  const lib = process.env.ERNIE55_LIBREF || path.join(build, 'radiance_home/kernels/libref.so');
  if (!fs.existsSync(lib) && !process.env.ERNIE55_LIBREF) {
    let r = command('cmake', ['-S', repo, '-B', build, '-G', 'Ninja', '-DRAD_WITH_HIP=OFF', '-DRAD_WITH_FFMPEG=OFF']);
    assert.equal(r.status, 0, 'configure failed: ' + r.stderr);
    r = command('cmake', ['--build', build, '--target', 'libref', '-j', '4']);
    assert.equal(r.status, 0, 'libref build failed: ' + r.stdout + r.stderr);
  }
  assert.ok(fs.existsSync(lib), 'real libref is available');
  return lib;
}
function binary() {
  if (!probeBin) {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ernie-attn-'));
    process.on('exit', () => fs.rmSync(tmp, {recursive: true, force: true}));
    probeBin = path.join(tmp, 'op_probe');
    const r = command('g++', ['-std=c++20', '-O1', '-I' + path.join(repo, 'abi'), path.join(__dirname, 'op_probe.cpp'), '-ldl', '-o', probeBin]);
    assert.equal(r.status, 0, 'ABI probe compile failed: ' + r.stderr);
  }
  return probeBin;
}
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
// round to float32, then to bf16 with round-to-nearest-even (the dtype the ops store)
function bf(x) {
  f32[0] = x;
  if (!Number.isFinite(f32[0])) return f32[0];
  const bits = u32[0], lsb = (bits >>> 16) & 1;
  u32[0] = ((bits + 0x7fff + lsb) >>> 0) & 0xffff0000;
  return f32[0];
}
function word(v) { return v === -Infinity ? '-inf' : v === Infinity ? 'inf' : Number.isNaN(v) ? 'nan' : String(v); }
function contiguous(shape) { const s = Array(shape.length).fill(1); for (let i = shape.length - 2; i >= 0; i--) s[i] = s[i + 1] * shape[i + 1]; return s; }
// tensor: {dtype:'bf16'|'f32'|'i32'|'u8'|'f8' (u8 and f8 are raw byte codes), role:'in'|'out'|'inout', shape, data (flat logical), stride?}
function T(dtype, role, shape, data, stride) { return {dtype, role, shape, data, stride: stride || contiguous(shape)}; }
const zeros = n => Array(n).fill(0);
// params: object of name -> number|string; floats must be marked {f: value}
function run(op, params, tensors, tamper) {
  const lines = [String(Object.keys(params).length)];
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === 'string') lines.push('s ' + k + ' ' + v);
    else if (typeof v === 'object') lines.push('f ' + k + ' ' + v.f);
    else lines.push('i ' + k + ' ' + v);
  }
  lines.push(String(tensors.length));
  for (const t of tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    assert.equal(t.data.length, n, 'tensor data length');
    lines.push([t.dtype, t.role, t.shape.length, ...t.shape, ...t.stride, n, ...t.data.map(word)].join(' '));
  }
  if (tamper) lines.push('tamper ' + tamper[0] + ' ' + tamper[1]);
  const r = command(binary(), [plugin(), op], {input: lines.join('\n') + '\n', timeout: 120000, env: {...process.env, OMP_NUM_THREADS: '1'}});
  assert.notEqual(r.status, 3, 'Missing registered host operation ' + op);
  assert.equal(r.status, 0, 'ABI probe failed: ' + r.stdout + r.stderr);
  const got = JSON.parse(r.stdout);
  got.out = got.tensors.map(a => a.map(v => v === 'nan' ? NaN : v === 'inf' ? Infinity : v === '-inf' ? -Infinity : v));
  return got;
}
// Deterministic generator (xorshift32).
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}
// values on a coarse dyadic grid, exactly representable in bf16
function grid(n, seed, scale = 1, step = 8) { const r = rng(seed); return Array.from({length: n}, () => scale * (Math.floor(r() * (2 * step + 1)) - step) / step); }
function maxAbs(a) { return a.reduce((m, v) => Math.max(m, Math.abs(v)), 0); }
function maxDiff(a, b) { assert.equal(a.length, b.length); return a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0); }
module.exports = {run, T, bf, zeros, rng, grid, maxAbs, maxDiff, contiguous};
