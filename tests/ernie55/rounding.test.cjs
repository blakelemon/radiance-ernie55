const test=require('node:test');
const assert=require('node:assert/strict');
const {run}=require('./support/probe.cjs');
// The pinned reference squares each quadratic hinge in f32 before its learned coefficient scales
// it. With a non-power-of-two coefficient the two orders land an ULP apart, so the pin is exact.
// The lane probe prints 10 significant digits, which round-trips every f32 here without collapsing
// the two candidates, so the reported value is narrowed back to f32 before comparing.
const f32=Math.fround;
const coefficient=f32(.12345);
const point=f32(16/7);
for(const [tail,v,coeff] of [
  ['upper',point,[0,0,0,coefficient]],
  ['lower',-point,[0,0,coefficient,0]],
]){
  const hinge=Math.abs(v);
  const squareFirst=f32(v+f32(coefficient*f32(hinge*hinge)));
  const weightFirst=f32(v+f32(f32(coefficient*hinge)*hinge));
  test(`ERNIE MultiMax squares the ${tail} hinge before weighting in F32`,()=>{
    assert.notEqual(squareFirst,weightFirst,'fixture must separate the two rounding orders');
    const out=run('ernie_multimax',[v],{ranges:[0,0,0,0],coeff});
    assert.equal(out.status,0);
    assert.equal(out.values.length,1);
    assert.equal(f32(out.values[0]),squareFirst,`${tail} tail must round its square before weighting`);
    assert.notEqual(f32(out.values[0]),weightFirst);
  });
}
test('ERNIE MultiMax keeps a zero-weight transform the identity',()=>{
  const out=run('ernie_multimax',[point,-point],{ranges:[0,0,0,0],coeff:[0,0,0,0]});
  assert.equal(out.status,0);
  assert.deepEqual(out.values.map(f32),[point,-point]);
  assert.ok(out.input_unchanged&&out.guards_ok);
});
