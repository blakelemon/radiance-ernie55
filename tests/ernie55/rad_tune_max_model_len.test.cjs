'use strict';
// rad-tune --max-model-len on a tiny synthetic ERNIE checkpoint, CPU reference kernels, two host devices.
// The tuner used to declare every model at the file's n_ctx_train; the ERNIE reference program refuses max-num-batched-tokens x context above 2^30 scores,
// so `rad-tune --tp 8` on the real 1,048,576-token container could not declare. The flag declares at the served length instead (and --max-num-batched-tokens /
// --max-num-seqs are accepted under the server's names). A tiny checkpoint carries no training context, so the refusal is provoked by naming 1,048,576 explicitly.
// Needs node, a host build of the repo (RAD_BUILD_DIR, default <repo>/build-ernie55) and the sysfs mask:
//   nosys node --test tests/ernie55/rad_tune_max_model_len.test.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const h = require('./support/synth/harness.cjs');
const kv = require('./support/kv_ckpt.cjs');

const repo = process.env.ERNIE_PORT_ROOT || process.cwd();
const build = process.env.RAD_BUILD_DIR || path.join(repo, 'build-ernie55');
let ready;
function tuner() {
  if (ready) return ready;
  h.ensureBuilt();
  const r = h.sh('cmake', ['--build', build, '--target', 'rad-tune', '-j', '8'], {timeout: 2400000});
  assert.equal(r.status, 0, 'rad-tune must build: ' + (r.stdout + r.stderr).slice(-1500));
  return (ready = {bin: path.join(build, 'bin', 'rad-tune'), home: path.join(build, 'radiance_home')});
}
function tune(args) {
  const t = tuner();
  const rad = kv.convertTarget([]);
  const r = h.sh(t.bin, ['-m', rad, '--home', t.home, '--kernels', 'libref', '--tp', '2', '--op', 'all_reduce', '--dry-run', ...args],
    {env: {...process.env, ROCR_VISIBLE_DEVICES: '99', RADIANCE_HOST_DEVICES: '2', RADIANCE_HOST_VRAM_MIB: '512', OMP_NUM_THREADS: '2'}});
  return {status: r.status, text: r.stdout + r.stderr};
}

test('--max-model-len reaches the declare: a context the reference program refuses at 2048 tokens is refused, a servable one declares', () => {
  const bad = tune(['--max-tok', '2048', '--max-seqs', '8', '--max-model-len', '1048576']);
  assert.notEqual(bad.status, 0);
  assert.match(bad.text, /indexer score plane is max-num-batched-tokens \(2048\) x max-model-len \(1048576\)/);
  const ok = tune(['--max-tok', '2048', '--max-seqs', '8', '--max-model-len', '163840']);
  assert.equal(ok.status, 0, ok.text.slice(-800));
  assert.match(ok.text, /--dry-run: nothing measured/);
});

test('the server flag names are accepted as aliases, and a bad value is refused', () => {
  const alias = tune(['--max-num-batched-tokens', '2048', '--max-num-seqs', '8', '--max-model-len', '1048576']);
  assert.notEqual(alias.status, 0);
  assert.match(alias.text, /max-num-batched-tokens \(2048\) x max-model-len \(1048576\)/, 'the alias sets max_tok');
  const zero = tune(['--max-model-len', '0']);
  assert.equal(zero.status, 2);
  assert.match(zero.text, /--max-model-len must be >= 1/);
});

test('without the flag nothing changes: the declare runs at the file\'s own context', () => {
  const r = tune(['--max-tok', '2048', '--max-seqs', '8']);
  assert.equal(r.status, 0, r.text.slice(-800));
});
