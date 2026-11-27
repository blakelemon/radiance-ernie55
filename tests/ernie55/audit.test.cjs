const test=require('node:test');
const assert=require('node:assert/strict');
const {spawnSync}=require('node:child_process');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');

const repo=process.env.ERNIE_PORT_ROOT||process.cwd();
const tool=path.join(repo,'tools/ernie55','audit_checkpoint.py');
const CONFIG='config.json';
const INDEX_FILE='model.safetensors.index.json';

const MLA=[7,14,21,28,35,42];
const schedule=(draft=1)=>Array.from({length:43+draft},(_,i)=>MLA.includes(i%43)?-2:128);
const BASE={model_type:'ernie5_5_moe',num_hidden_layers:43,num_nextn_predict_layers:1,hidden_size:4096,
  head_dim:512,hc_mult:4,moe_latent_size:2048,n_routed_experts:512,n_shared_experts:1,
  num_experts_per_tok:10,first_k_dense_replace:1,eos_token_id:2,compress_ratios:schedule()};
const INDEX={metadata:{total_size:7},weight_map:{t0:'model-00001.safetensors',t1:'model-00001.safetensors',
  t2:'model-00002.safetensors'}};

function audit(files={},dirs=[]){
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ernie55-audit-'));
  try{
    dirs.forEach(d=>fs.mkdirSync(path.join(dir,d)));
    Object.entries(files).forEach(([n,body])=>fs.writeFileSync(path.join(dir,n),
      Buffer.isBuffer(body)?body:(typeof body==='string'?body:JSON.stringify(body))));
    const r=spawnSync('python3',[tool,'--metadata-dir',dir],{encoding:'utf8'});
    return {status:r.status,stdout:r.stdout,stderr:r.stderr,
      json:r.status===0?JSON.parse(r.stdout):null,
      listing:fs.readdirSync(dir).sort(),
      unchanged:Object.keys(files).every(n=>fs.readFileSync(path.join(dir,n),'utf8')===files[n]||
        (typeof files[n]!=='string'&&fs.readFileSync(path.join(dir,n),'utf8')===JSON.stringify(files[n])))};
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
}
const ok=(over={},index=INDEX,dirs=[])=>audit({[CONFIG]:{...BASE,...over},[INDEX_FILE]:index},dirs);
const refuses=(over,index=INDEX)=>{
  const r=ok(over,index);
  assert.equal(r.status,2,JSON.stringify(over)+' -> '+r.stderr);
  assert.equal(r.stdout.trim(),'');
  return r;
};

test('Audit reports the documented metadata surface and nothing else',()=>{
  const r=ok();assert.equal(r.status,0,r.stderr);
  assert.deepEqual(Object.keys(r.json),['main_layers','draft_layers','attention','geometry','moe',
    'checkpoint','special_tokens']);
  assert.deepEqual(Object.keys(r.json.attention),['hca_layers','mla_dsa_layers','draft_schedule']);
  assert.deepEqual(Object.keys(r.json.geometry),['hidden_size','head_dim','hc_mult']);
  assert.deepEqual(Object.keys(r.json.moe),['dense_layers','routed_layers','routed_experts','top_k',
    'latent_size']);
  assert.deepEqual(Object.keys(r.json.checkpoint),['tensor_count','shard_count','declared_bytes',
    'tensor_shapes_available']);
  assert.deepEqual(Object.keys(r.json.special_tokens),['eos_token_id']);
  assert.equal(r.json.checkpoint.tensor_count,3);assert.equal(r.json.checkpoint.shard_count,2);
  assert.equal(r.json.checkpoint.declared_bytes,7);assert.equal(r.json.special_tokens.eos_token_id,2);
  assert.match(r.stderr,/tensor shapes are unavailable/);
});
test('Audit reads only metadata: inputs untouched, shard paths never followed',()=>{
  const r=ok({},undefined,['model-00001.safetensors']);
  assert.equal(r.status,0,r.stderr);assert.ok(r.unchanged);
  assert.deepEqual(r.listing,[CONFIG,'model-00001.safetensors',INDEX_FILE]);
});
test('A target with no draft layer reports an empty draft schedule',()=>{
  const r=ok({num_nextn_predict_layers:0,compress_ratios:schedule(0)});
  assert.equal(r.status,0,r.stderr);
  assert.equal(r.json.draft_layers,0);assert.deepEqual(r.json.attention.draft_schedule,[]);
  assert.equal(r.json.main_layers,43);assert.deepEqual(r.json.attention.mla_dsa_layers,MLA);
  assert.equal(r.json.attention.hca_layers.length,37);
});
test('Draft entries stay out of the main-layer indices',()=>{
  const r=ok({num_nextn_predict_layers:3,compress_ratios:[...schedule(0),128,-2,128]});
  assert.equal(r.status,0,r.stderr);
  assert.deepEqual(r.json.attention.draft_schedule,[128,-2,128]);
  assert.deepEqual(r.json.attention.mla_dsa_layers,MLA);
  assert.ok(r.json.attention.hca_layers.every(i=>i<43));
});
test('Layer lists follow the schedule, not a period, and the dense prefix splits them',()=>{
  const r=ok({compress_ratios:schedule(1).map((v,i)=>(i===5||i===40)?-2:v)});
  assert.equal(r.status,0,r.stderr);
  assert.deepEqual(r.json.attention.mla_dsa_layers,[5,7,14,21,28,35,40,42]);
  assert.deepEqual(r.json.moe.dense_layers,[0]);
  assert.deepEqual(r.json.moe.routed_layers,Array.from({length:42},(_,i)=>i+1));
  const none=ok({first_k_dense_replace:0});
  assert.deepEqual(none.json.moe.dense_layers,[]);
  assert.deepEqual(none.json.moe.routed_layers,Array.from({length:43},(_,i)=>i));
  const all=ok({first_k_dense_replace:43});
  assert.deepEqual(all.json.moe.routed_layers,[]);
  assert.equal(all.json.moe.dense_layers.length,43);
});
test('Boundary metadata still audits, and absent optional fields stay null',()=>{
  assert.equal(ok({num_experts_per_tok:512}).json.moe.top_k,512);
  assert.equal(ok({num_hidden_layers:1,num_nextn_predict_layers:0,first_k_dense_replace:0,
    compress_ratios:[128]}).json.main_layers,1);
  assert.equal(ok({num_nextn_predict_layers:0,compress_ratios:schedule(0)}).json.draft_layers,0);
  const sparse=ok({},{'weight_map':{t0:'model-00001.safetensors',t1:'model-00002.safetensors',
    t2:'model-00001.safetensors'}});
  assert.equal(sparse.json.checkpoint.tensor_count,3);assert.equal(sparse.json.checkpoint.shard_count,2);
  assert.equal(sparse.json.checkpoint.declared_bytes,null);
  const bare=audit({[CONFIG]:{...BASE,eos_token_id:undefined},[INDEX_FILE]:{'weight_map':
    {'t0':'model-00001.safetensors'}}});
  assert.equal(bare.status,0,bare.stderr);
  assert.equal(bare.json.special_tokens.eos_token_id,null);
  assert.equal(bare.json.checkpoint.declared_bytes,null);
});
test('Refuses booleans where an integer count is required',()=>{
  for(const field of ['num_hidden_layers','num_nextn_predict_layers','hidden_size','head_dim','hc_mult',
    'moe_latent_size','n_routed_experts','n_shared_experts','num_experts_per_tok','first_k_dense_replace',
    'eos_token_id'])
      assert.match(refuses({[field]:true}).stderr,new RegExp(field));
  assert.match(refuses({compress_ratios:schedule(1).map((v,i)=>i===3?true:v)}).stderr,
    /compress_ratios\[3\]/);
});
test('Refuses impossible counts and an out-of-range dense prefix',()=>{
  for(const over of [{num_hidden_layers:0},{hidden_size:0},{head_dim:0},{hc_mult:0},{moe_latent_size:0},
    {n_routed_experts:0},{n_shared_experts:0},{num_experts_per_tok:0},{num_experts_per_tok:513},
    {num_nextn_predict_layers:-1},{first_k_dense_replace:-1},{first_k_dense_replace:44},
    {hidden_size:4096.5},{hc_mult:null},{num_hidden_layers:'43'}])
      refuses(over);
});
test('Refuses inconsistent schedules and unsupported attention modes',()=>{
  assert.match(refuses({compress_ratios:schedule(0)}).stderr,/compress_ratios/);
  assert.match(refuses({compress_ratios:[...schedule(1),128]}).stderr,/compress_ratios/);
  for(const bad of [64,0,-1,2,127,-3])
    assert.match(refuses({compress_ratios:schedule(1).map((v,i)=>i===2?bad:v)}).stderr,
      new RegExp(`compress_ratios\\[2\\] is ${bad}`));
  for(const bad of [undefined,[],128,null,'128','x'.repeat(44),
    schedule(1).map(v=>String(v))])
      assert.match(refuses({compress_ratios:bad}).stderr,/compress_ratios/);
});
test('Refuses a tensor index that cannot describe its shards',()=>{
  for(const index of [{},{'weight_map':{}},{'weight_map':[]},{'weight_map':{'t0':1}},
    {'weight_map':{'t0':null}},{'weight_map':{'t0':['model-00001.safetensors']}},
    {'weight_map':INDEX.weight_map,'metadata':[]},
    {'weight_map':INDEX.weight_map,'metadata':{'total_size':-1}},
    {'weight_map':INDEX.weight_map,'metadata':{'total_size':true}},
    {'weight_map':INDEX.weight_map,'metadata':{'total_size':1.5}}])
      refuses({},index);
});
test('Refuses shard names that are not plain relative safetensors filenames',()=>{
  for(const shard of ['../outside.safetensors','..','.','/tmp/outside.safetensors','./model.safetensors',
    'sub/model.safetensors','sub/../../outside.safetensors','model-00001.safetensors.bak','model.bin',
    '.safetensors','https://example.com/model.safetensors','file:///tmp/model.safetensors',
    'C:\\shards\\model.safetensors','C:model.safetensors','\\\\server\\share\\model.safetensors',
    ' model.safetensors',
    'model.safetensors ','model\x00.safetensors','~model.safetensors','model.SAFETENSORS']){
    const r=refuses({},{'weight_map':{'lm_head.weight':shard}});
    assert.match(r.stderr,/shard/);
  }
});
test('A named shard need not exist next to the metadata',()=>{
  const r=ok({},{'weight_map':{'lm_head.weight':'not-present-00001.safetensors'}});
  assert.equal(r.status,0,r.stderr);assert.equal(r.json.checkpoint.shard_count,1);
});
test('Refuses absent or malformed metadata, naming the file',()=>{
  let r=audit({[INDEX_FILE]:INDEX});assert.equal(r.status,2);assert.match(r.stderr,/config\.json/);
  r=audit({[CONFIG]:BASE});assert.equal(r.status,2);assert.match(r.stderr,/model\.safetensors\.index\.json/);
  assert.equal(audit({}).status,2);
  r=audit({[CONFIG]:'{ not json',[INDEX_FILE]:INDEX});assert.equal(r.status,2);
  assert.match(r.stderr,/config\.json is not valid JSON/);
  r=audit({[CONFIG]:BASE,[INDEX_FILE]:'[]'});assert.equal(r.status,2);
  assert.match(r.stderr,/index\.json is not a JSON object/);
  r=audit({[CONFIG]:'[1,2]',[INDEX_FILE]:INDEX});assert.equal(r.status,2);
  assert.match(r.stderr,/config\.json is not a JSON object/);
  r=audit({[CONFIG]:Buffer.from([0xff,0xfe,0x00,0xff]),[INDEX_FILE]:INDEX});assert.equal(r.status,2);
  assert.match(r.stderr,/config\.json cannot be read/);assert.doesNotMatch(r.stderr,/Traceback/);
  assert.equal(r.stdout.trim(),'');
  assert.match(refuses({model_type:'ernie4_5_moe'}).stderr,/ernie5_5_moe/);
  assert.match(refuses({model_type:null}).stderr,/model_type/);
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ernie55-audit-'));
  try{
    const file=path.join(dir,'config.json');fs.writeFileSync(file,'{}');
    const notDir=spawnSync('python3',[tool,'--metadata-dir',file],{encoding:'utf8'});
    assert.equal(notDir.status,2);assert.match(notDir.stderr,/metadata directory/);
    assert.equal(notDir.stdout.trim(),'');
    const gone=spawnSync('python3',[tool,'--metadata-dir',path.join(dir,'nope')],{encoding:'utf8'});
    assert.equal(gone.status,2);assert.match(gone.stderr,/metadata directory/);
    const noArg=spawnSync('python3',[tool],{encoding:'utf8'});
    assert.equal(noArg.status,2);assert.equal(noArg.stdout.trim(),'');
    assert.match(noArg.stderr,/metadata-dir/);
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});
test('Exposes audit_checkpoint(metadata_dir) and main(argv) for reuse',()=>{
  const dir=fs.mkdtempSync(path.join(os.tmpdir(),'ernie55-audit-'));
  try{
    fs.writeFileSync(path.join(dir,CONFIG),JSON.stringify(BASE));
    fs.writeFileSync(path.join(dir,INDEX_FILE),JSON.stringify(INDEX));
    const py='import contextlib,importlib.util,inspect,io,json,pathlib,sys\n'
      +'HERE=sys.argv[1]\n'
      +`spec=importlib.util.spec_from_file_location("audit_checkpoint",${JSON.stringify(tool)})\n`
      +'mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)\n'
      +'assert list(inspect.signature(mod.audit_checkpoint).parameters)==["metadata_dir"]\n'
      +'assert list(inspect.signature(mod.main).parameters)==["argv"]\n'
      +'buf=io.StringIO()\n'
      +'with contextlib.redirect_stdout(buf):\n'
      +'    assert mod.main(["--metadata-dir",HERE])==0\n'
      +'report=json.loads(buf.getvalue())\n'
      +'assert report["main_layers"]==43\n'
      +'d=mod.audit_checkpoint(pathlib.Path(HERE))\n'
      +'assert isinstance(d,dict) and d["checkpoint"]["tensor_shapes_available"] is False\n'
      +'assert mod.audit_checkpoint(HERE)==d\n'
      +'try:\n'
      +'    mod.audit_checkpoint(pathlib.Path(HERE).parent)\n'
      +'except mod.AuditError:\n'
      +'    pass\n'
      +'else:\n'
      +'    raise AssertionError("AuditError expected for a directory without metadata")\n'
      +'print("signatures-ok")\n';
    const r=spawnSync('python3',['-c',py,dir],{encoding:'utf8'});
    assert.equal(r.status,0,r.stderr);assert.match(r.stdout,/signatures-ok/);
  } finally {fs.rmSync(dir,{recursive:true,force:true});}
});
