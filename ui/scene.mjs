// Pure scene geometry. Coordinates are meters, with Z pointing up.
const EPS = 1e-12;
export const clone = value => JSON.parse(JSON.stringify(value));

export function footprint(object) {
  const half = object.edge_m / 2;
  const a = object.yaw_deg * Math.PI / 180;
  return [[-half,-half],[half,-half],[half,half],[-half,half]].map(([x,y]) => [
    object.position_m[0] + x*Math.cos(a)-y*Math.sin(a),
    object.position_m[1] + x*Math.sin(a)+y*Math.cos(a)
  ]);
}

function overlap(a,b,axes) {
  return axes.every(([x,y]) => {
    const pa=a.map(p=>p[0]*x+p[1]*y), pb=b.map(p=>p[0]*x+p[1]*y);
    return Math.min(Math.max(...pa),Math.max(...pb))-Math.max(Math.min(...pa),Math.min(...pb)) > EPS;
  });
}

export function voxelizeCube(object,grid) {
  const size=grid.edge_m, origin=grid.origin_m;
  if (!(size>0) || !(object.edge_m>0)) throw new Error('Ungültige Rastergröße');
  const polygon=footprint(object), a=object.yaw_deg*Math.PI/180;
  const axes=[[1,0],[0,1],[Math.cos(a),Math.sin(a)],[-Math.sin(a),Math.cos(a)]];
  const lo=[Math.min(...polygon.map(p=>p[0])),Math.min(...polygon.map(p=>p[1])),object.position_m[2]-object.edge_m/2];
  const hi=[Math.max(...polygon.map(p=>p[0])),Math.max(...polygon.map(p=>p[1])),object.position_m[2]+object.edge_m/2];
  const first=lo.map((v,i)=>Math.floor((v-origin[i])/size+EPS));
  const last=hi.map((v,i)=>Math.ceil((v-origin[i])/size-EPS)-1);
  const cells=[];
  for(let x=first[0];x<=last[0];x++) for(let y=first[1];y<=last[1];y++) {
    const bx=origin[0]+x*size, by=origin[1]+y*size;
    if(!overlap(polygon,[[bx,by],[bx+size,by],[bx+size,by+size],[bx,by+size]],axes)) continue;
    for(let z=first[2];z<=last[2];z++) {
      const bottom=origin[2]+z*size;
      if(Math.min(bottom+size,hi[2])-Math.max(bottom,lo[2])>EPS) cells.push([x,y,z]);
    }
  }
  return cells;
}

export function createTarget(scene) {
  return {schema_version:1,kind:'desired_scene',units:'m',source_pose_sha256:scene.source.pose_sha256,
    objects:scene.objects.map(({id,edge_m,position_m,yaw_deg})=>({id,edge_m,position_m:[...position_m],yaw_deg})),
    robot_execution_requested:false,scene_valid_for_planning:false};
}

export function validateTarget(input,scene) {
  if(input?.schema_version!==1 || input.kind!=='desired_scene' || input.units!=='m') throw new Error('Keine unterstützte Ziel-Datei.');
  if(input.source_pose_sha256!==scene.source.pose_sha256) throw new Error('Dieses Ziel gehört zu einer anderen Aufnahme.');
  if(!Array.isArray(input.objects)||input.objects.length!==scene.objects.length) throw new Error('Die Objektliste passt nicht zur Aufnahme.');
  const result=createTarget(scene), bounds=scene.table.display_bounds_m;
  for(let i=0;i<result.objects.length;i++) {
    const obj=input.objects[i], expected=result.objects[i];
    if(obj.id!==expected.id || obj.edge_m!==expected.edge_m || !Array.isArray(obj.position_m)||obj.position_m.length!==3 ||
       !obj.position_m.every(Number.isFinite)||!Number.isFinite(obj.yaw_deg)) throw new Error('Ungültige Würfelposition.');
    const [x,y,z]=obj.position_m;
    if(x<bounds.x[0]||x>bounds.x[1]||y<bounds.y[0]||y>bounds.y[1]||Math.abs(z-expected.edge_m/2)>1e-9) throw new Error('Das Ziel muss innerhalb der Ansicht auf dem Tisch liegen.');
    expected.position_m=[x,y,z]; expected.yaw_deg=obj.yaw_deg;
  }
  return result;
}

export function moveTarget(target,scene,x,y,yaw,{snap=false}={}) {
  if(![x,y,yaw].every(Number.isFinite)) throw new Error('Bitte gültige Zahlen eingeben.');
  const bounds=scene.table.display_bounds_m, step=scene.grid.edge_m;
  if(snap) {x=scene.grid.origin_m[0]+Math.round((x-scene.grid.origin_m[0])/step)*step;y=scene.grid.origin_m[1]+Math.round((y-scene.grid.origin_m[1])/step)*step;}
  const result=clone(target),object=result.objects[0];
  object.position_m=[Math.max(bounds.x[0],Math.min(bounds.x[1],x)),Math.max(bounds.y[0],Math.min(bounds.y[1],y)),object.edge_m/2];
  object.yaw_deg=yaw;
  return validateTarget(result,scene);
}
