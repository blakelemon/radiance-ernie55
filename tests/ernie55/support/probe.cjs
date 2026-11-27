const {spawnSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const repo = process.env.ERNIE_PORT_ROOT || process.cwd();
let binary;
function run(op, input, {dtype='f32', rows=1, n=input.length/(op.endsWith('mul')?2:1), beta=4, linearBeta=25, badShape=false, ranges=[], coeff=[]}={}) {
  if (!binary) {
    const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ernie55-abi-'));
    binary=path.join(dir,'kernel-probe');
    const result=spawnSync('g++',['-std=c++20','-O1','-I'+path.join(repo,'abi'),path.join(__dirname,'kernel_probe.cpp'),'-ldl','-o',binary],{encoding:'utf8',timeout:60000});
    assert.equal(result.status,0,'ABI harness must compile: '+result.stderr);
    process.on('exit',()=>fs.rmSync(dir,{recursive:true,force:true}));
  }
  const situ=op.endsWith('mul');
  const lib=process.env.ERNIE55_LIBREF || path.join(repo,'build-ernie55','radiance_home','kernels','libref.so');
  assert.ok(fs.existsSync(lib),'Build the actual libref plugin before testing: '+lib);
  const result=spawnSync(binary,[lib,op,dtype,''+rows,''+n,''+((situ?2*n:n)+3),''+(n+5),''+beta,''+linearBeta,badShape?'bad-shape':'valid','reserved'],{input:[...input,...ranges,...coeff].join(' ')+'\n',encoding:'utf8',timeout:10000,env:{...process.env,OMP_NUM_THREADS:'1'}});
  assert.equal(result.status,0,'The registered host kernel must be callable; '+op+' returned '+result.status+': '+result.stdout+' '+result.stderr);
  return JSON.parse(result.stdout);
}
function bf16(x) {
  const b=new ArrayBuffer(4), v=new DataView(b);v.setFloat32(0,x,true);
  let i=v.getUint32(0,true); i=(i+0x7fff+((i>>>16)&1))>>>0;
  v.setUint32(0,(i>>>16)<<16,true);return v.getFloat32(0,true);
}
function close(actual,expected,tolerance=2e-5) {
  assert.equal(actual.length,expected.length);
  actual.forEach((value,i)=>{
    assert.ok(Number.isFinite(value),'Output must be finite');
    assert.ok(Math.abs(value-expected[i]) <= tolerance*Math.max(1,Math.abs(expected[i])),`${i}: ${value} != ${expected[i]}`);
  });
}
module.exports={run,bf16,close,repo};
