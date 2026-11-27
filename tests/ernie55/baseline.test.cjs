const test=require('node:test');
const assert=require('node:assert/strict');
const {run,close,bf16}=require('./support/probe.cjs');
test('Existing SiLU host kernel keeps gate-first numerical behavior',()=>{
  const g=[-4,-1,0,1,4],u=[-3,2,5,-2,3];
  const out=run('silu_mul',[...g,...u]);
  assert.equal(out.status,0);close(out.values,g.map((x,i)=>x/(1+Math.exp(-x))*u[i]));
});
test('Existing SiLU host kernel preserves padded batch rows and inputs',()=>{
  const out=run('silu_mul',[1,-2,3,4, -1,2,-3,-4],{rows:2,n:2});
  close(out.values,[3/(1+Math.exp(-1)),-8/(1+Math.exp(2)),3/(1+Math.exp(1)),-8/(1+Math.exp(-2))]);
  assert.ok(out.input_unchanged);assert.ok(out.guards_ok);
});
test('Existing SiLU host kernel retains BF16 support',()=>{
  const out=run('silu_mul',[1,2],{dtype:'bf16'});
  assert.equal(out.status,0);close(out.values,[bf16(2/(1+Math.exp(-1)))],0.008);
});
