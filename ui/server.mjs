// Loopback-only viewer server. Static files, plus one narrow write endpoint for the grasp station:
// POST /api/command {"cmd": one of COMMANDS, place: x_mm,y_mm,yaw_deg} -> data/command.json (grasp_app.py polls it). Nothing else is writable.
import http from 'node:http';
import {readFile,writeFile,rename} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.dirname(fileURLToPath(import.meta.url));
const types={'.html':'text/html; charset=utf-8','.css':'text/css','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml'};
const COMMANDS=new Set(['home','locate','grasp','place','stop','open','release','apply_offset']);
let counter=0;
async function command(req,res){
  const host=req.headers.host||'',origin=req.headers.origin;
  if(!/^(127\.0\.0\.1|localhost)(:\d+)?$/.test(host)||(origin&&!/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin))){res.writeHead(403);res.end();return;}
  let body='';for await(const chunk of req){body+=chunk;if(body.length>200){res.writeHead(413);res.end();return;}}
  let input={};try{input=JSON.parse(body);}catch{}
  const cmd=input.cmd;
  if(!COMMANDS.has(cmd)){res.writeHead(400);res.end('unknown command');return;}
  const out={cmd,id:`${Date.now()}-${++counter}`};
  if(cmd==='place'){   // target cube pose in table mm; the station re-checks reachability
    const ok=(v,lo,hi)=>typeof v==='number'&&Number.isFinite(v)&&v>=lo&&v<=hi;
    if(!ok(input.x_mm,-300,600)||!ok(input.y_mm,-300,600)||!ok(input.yaw_deg,-360,360)){res.writeHead(400);res.end('bad target');return;}
    Object.assign(out,{x_mm:input.x_mm,y_mm:input.y_mm,yaw_deg:input.yaw_deg});
  }
  const file=path.join(root,'data','command.json'),tmp=file+'.tmp';
  await writeFile(tmp,JSON.stringify(out));await rename(tmp,file);
  res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify({ok:true,cmd}));
}
const server=http.createServer(async(req,res)=>{
  if(req.method==='POST'&&req.url==='/api/command'){try{await command(req,res);}catch{res.writeHead(500);res.end();}return;}
  if(!['GET','HEAD'].includes(req.method)) {res.writeHead(405);res.end();return;}
  try{
    const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    const requested=name==='/'?'/index.html':name;
    if(!/^\/(index\.html|app\.mjs|scene\.mjs|style\.css|data\/[\w.-]+|node_modules\/three\/(build|examples\/jsm)\/[\w/.-]+)$/.test(requested)) throw new Error('not found');
    const filename=path.resolve(root,'.'+requested);
    if(!filename.startsWith(root+path.sep)) throw new Error('not found');
    const bytes=await readFile(filename);
    res.writeHead(200,{'Content-Type':types[path.extname(filename)]||'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
    res.end(req.method==='HEAD'?undefined:bytes);
  }catch{res.writeHead(404);res.end('Not found');}
});
server.listen(Number(process.env.PORT||4173),'127.0.0.1',()=>console.log(`Local: http://127.0.0.1:${server.address().port}`));
