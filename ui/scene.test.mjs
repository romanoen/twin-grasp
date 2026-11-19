import test from 'node:test';
import assert from 'node:assert/strict';
import {voxelizeCube,createTarget,moveTarget,validateTarget} from './scene.mjs';

const grid={edge_m:.005,origin_m:[0,0,0]};
const object={id:'red',edge_m:.025,position_m:[.0125,.0125,.0125],yaw_deg:0};
const scene={source:{pose_sha256:'fixed-source'},objects:[object],grid,table:{display_bounds_m:{x:[-.1,.4],y:[-.2,.25]}}};
test('aligned 25 mm cube fills exactly 125 cells; touching boundary excluded',()=>{
  const cells=voxelizeCube(object,grid);assert.equal(cells.length,125);assert(cells.every(c=>c.every(v=>v>=0&&v<5)));
});
test('rotated cube fills its interior across all five layers',()=>{
  const cells=voxelizeCube({...object,yaw_deg:45},grid);assert(cells.length>125);assert.equal(new Set(cells.map(c=>c[2])).size,5);
  assert(cells.some(([x,y,z])=>x===2&&y===2&&z===2));
});
test('translated grid origin preserves occupied indices and supports negative coordinates',()=>{
  const shift=[-.107,.203,-.02],moved={...object,position_m:object.position_m.map((v,i)=>v+shift[i])};
  assert.deepEqual(voxelizeCube(moved,{...grid,origin_m:shift}),voxelizeCube(object,grid));
  const negative=voxelizeCube({...object,position_m:[-.0125,-.0125,.0125]},grid);
  assert.equal(negative.length,125);assert(negative.every(([x,y])=>x<0&&y<0));
});
test('snap changes target only and preserves exact observed pose',()=>{
  const before=JSON.stringify(scene),target=createTarget(scene),next=moveTarget(target,scene,.147,.033,32,{snap:true});
  assert.deepEqual(next.objects[0].position_m,[.145,.035,.0125]);assert.equal(JSON.stringify(scene),before);
  assert.deepEqual(target.objects[0].position_m,object.position_m);assert.equal(next.robot_execution_requested,false);
});
test('export/import preserves meters and angle; rejects other source, nonfinite or elevated targets',()=>{
  const target=moveTarget(createTarget(scene),scene,.1512,.0367,81.3);
  assert.deepEqual(validateTarget(JSON.parse(JSON.stringify(target)),scene),target);
  assert.throws(()=>validateTarget({...target,source_pose_sha256:'different'},scene));
  for(const invalid of [[NaN,0,.0125],[0,0,.05],[5,0,.0125]]) {
    const input=structuredClone(target);input.objects[0].position_m=invalid;assert.throws(()=>validateTarget(input,scene));
  }
});
test('target snapping respects an offset raster origin',()=>{
  const shifted={...scene,grid:{...grid,origin_m:[.002,.003,0]}};
  const result=moveTarget(createTarget(shifted),shifted,.148,.036,0,{snap:true});
  assert(Math.abs(result.objects[0].position_m[0]-.147)<1e-12);
  assert(Math.abs(result.objects[0].position_m[1]-.038)<1e-12);
});
