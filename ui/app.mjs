// Viewer for grasp_app.py: 3D scene (table, live arm, work area, cube from the map, placement target), wrist camera,
// station commands and a message log. The page only reads data/*.json and posts commands to /api/command.
import * as THREE from 'three';
import {STLLoader} from 'three/addons/loaders/STLLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {clone, createTarget, moveTarget, validateTarget} from './scene.mjs';

const $ = id => document.getElementById(id);
const fmt = (v, d = 1) => v.toFixed(d);
const CUBE = .025;
const DEFAULT_TARGET = [.02, .23, 0];   // middle of the work area (m, m, deg)
const BUSY = new Set(['moving', 'locating', 'grasping', 'placing']);
const TONE = {ready: 'ok', moving: 'busy', locating: 'busy', grasping: 'busy', placing: 'busy', stopped: 'bad', error: 'bad'};

let data, target, scene, camera, renderer, controls, targetMesh, robot, snapshot;
let state = null;                       // last grasp-state.json
let areaCells = [], areaStep = 10;
const viewport = $('viewport');

// ---------------- scene ----------------
function lines(points, color, opacity = 1) {
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({color, transparent: opacity < 1, opacity}));
}

function addTable() {
  const b = data.table.display_bounds_m, w = b.x[1] - b.x[0], h = b.y[1] - b.y[0];
  const table = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshLambertMaterial({color: 0x161c24}));
  table.position.set((b.x[0] + b.x[1]) / 2, (b.y[0] + b.y[1]) / 2, -.001); table.name = 'table';
  scene.add(table);
  for (const [step, color, opacity] of [[.01, 0x2a3440, .45], [.05, 0x3a4756, 1]]) {
    const p = [];
    for (let x = Math.ceil(b.x[0] / step) * step; x <= b.x[1] + 1e-9; x += step) p.push(x, b.y[0], 0, x, b.y[1], 0);
    for (let y = Math.ceil(b.y[0] / step) * step; y <= b.y[1] + 1e-9; y += step) p.push(b.x[0], y, 0, b.x[1], y, 0);
    scene.add(lines(p, color, opacity));
  }
  scene.add(lines([b.x[0], b.y[0], 0, b.x[1], b.y[0], 0, b.x[1], b.y[0], 0, b.x[1], b.y[1], 0,
                   b.x[1], b.y[1], 0, b.x[0], b.y[1], 0, b.x[0], b.y[1], 0, b.x[0], b.y[0], 0], 0x55677a));
  const tag = new THREE.Mesh(new THREE.PlaneGeometry(data.tag.edge_m, data.tag.edge_m),
    new THREE.MeshBasicMaterial({map: new THREE.TextureLoader().load('data/tag.png'), transparent: true, opacity: .85}));
  tag.position.set(...data.tag.center_table_m); tag.position.z += .0005; scene.add(tag);
  // table frame axes at the tag origin
  scene.add(lines([0, 0, .001, .06, 0, .001], 0xff5a4e), lines([0, 0, .001, 0, .06, .001], 0x45d483), lines([0, 0, .001, 0, 0, .06], 0x4ea3ff));
  return table;
}

async function addArea() {
  const r = await fetch('data/grasp-area.json', {cache: 'no-store'});
  if (!r.ok) return;
  const area = await r.json(), s = area.step_mm / 1000;
  areaCells = area.cells.map(c => c.xy); areaStep = area.step_mm;
  const mesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(s * .9, s * .9),
    new THREE.MeshBasicMaterial({color: 0x3d7a5a, transparent: true, opacity: .35, depthWrite: false}), areaCells.length);
  const m = new THREE.Matrix4();
  areaCells.forEach(([x, y], i) => { m.makeTranslation(x / 1000, y / 1000, .0006); mesh.setMatrixAt(i, m); });
  mesh.name = 'work-area'; scene.add(mesh);
}

const box = new THREE.BoxGeometry(CUBE, CUBE, CUBE);
const cubeBox = new THREE.Group();
cubeBox.add(new THREE.Mesh(box, new THREE.MeshStandardMaterial({color: 0xc2453b, roughness: .6})),
            new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({color: 0x7a241e})));
cubeBox.visible = false;

const targetBox = new THREE.Group();
{
  const ring = new THREE.Mesh(new THREE.RingGeometry(CUBE * .95, CUBE * 1.05, 48), new THREE.MeshBasicMaterial({color: 0x6f9fc4, transparent: true, opacity: .55, side: THREE.DoubleSide}));
  ring.position.z = -CUBE / 2 + .0008;
  const tick = lines([-CUBE * 1.3, 0, 0, -CUBE * .7, 0, 0, CUBE * .7, 0, 0, CUBE * 1.3, 0, 0, 0, -CUBE * 1.3, 0, 0, -CUBE * .7, 0, 0, CUBE * .7, 0, 0, CUBE * 1.3, 0], 0x6f9fc4, .8);
  tick.position.z = -CUBE / 2 + .0008;
  targetBox.add(new THREE.Mesh(box, new THREE.MeshBasicMaterial({color: 0x6f9fc4, transparent: true, opacity: .14, depthWrite: false})),
                new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({color: 0x6f9fc4})), ring, tick);
}

function home(top = false) {
  // Fit arm, work area and target into the view.
  camera.aspect = viewport.clientWidth / Math.max(viewport.clientHeight, 1); camera.updateProjectionMatrix();
  const bounds = new THREE.Box3().expandByObject(targetBox);
  if (robot) bounds.expandByObject(robot);
  for (const [x, y] of areaCells) bounds.expandByPoint(new THREE.Vector3(x / 1000, y / 1000, 0));
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  const fov = Math.min(camera.fov, 2 * Math.atan(Math.tan(camera.fov * Math.PI / 360) * camera.aspect) * 180 / Math.PI) * Math.PI / 180;
  const dist = THREE.MathUtils.clamp(sphere.radius * 1.05 / Math.sin(fov / 2), controls.minDistance, controls.maxDistance);
  const dir = top ? new THREE.Vector3(0, -.001, 1) : new THREE.Vector3(.75, -.9, .75);
  controls.target.copy(sphere.center); camera.position.copy(sphere.center).add(dir.normalize().multiplyScalar(dist)); controls.update();
  $('view-orbit').classList.toggle('on', !top); $('view-top').classList.toggle('on', top);
}

// ---------------- placement target ----------------
const storeKey = () => `so101-target:${data.source.pose_sha256}`;
const targetXY = () => target.objects[0].position_m.slice(0, 2);
const targetYaw = () => ((Math.round(target.objects[0].yaw_deg) % 90) + 90) % 90;

function setTarget(x, y, yaw) {
  try { target = moveTarget(target, data, x, y, yaw); } catch { return; }
  try { localStorage.setItem(storeKey(), JSON.stringify(target)); } catch {}
  showTarget();
}

function showTarget() {
  const [x, y] = targetXY(), yaw = targetYaw();
  targetBox.position.set(x, y, CUBE / 2); targetBox.rotation.z = yaw * Math.PI / 180;
  targetMesh.position.copy(targetBox.position);
  if (document.activeElement !== $('target-x')) $('target-x').value = Math.round(x * 1000);
  if (document.activeElement !== $('target-y')) $('target-y').value = Math.round(y * 1000);
  if (document.activeElement !== $('target-yaw')) $('target-yaw').value = yaw;
  updateButtons();
}

function targetInArea() {
  const [x, y] = targetXY().map(v => v * 1000);
  return areaCells.some(([cx, cy]) => Math.abs(cx - x) <= areaStep / 2 + .01 && Math.abs(cy - y) <= areaStep / 2 + .01);
}

function initPointer(table) {
  const fromFields = () => setTarget(Number($('target-x').value) / 1000, Number($('target-y').value) / 1000, Number($('target-yaw').value));
  for (const id of ['target-x', 'target-y', 'target-yaw']) $(id).addEventListener('change', fromFields);

  const canvas = renderer.domElement, ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -CUBE / 2), offset = new THREE.Vector3();
  let dragging = false;
  const aim = e => { const r = canvas.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera); };
  canvas.addEventListener('pointerdown', e => {
    if (e.button !== 0) return; aim(e);
    if (!ray.intersectObject(targetMesh).length) return;
    const p = ray.ray.intersectPlane(plane, new THREE.Vector3()); if (!p) return;
    dragging = true; controls.enabled = false; offset.copy(p).sub(targetMesh.position); canvas.setPointerCapture(e.pointerId); canvas.focus();
  }, {capture: true});
  canvas.addEventListener('pointermove', e => {
    aim(e);
    const hit = ray.intersectObject(table)[0];
    $('cursor').textContent = hit ? `x ${fmt(hit.point.x * 1000).padStart(6)}  y ${fmt(hit.point.y * 1000).padStart(6)}  mm` : 'x ---.-  y ---.-  mm';
    if (!dragging) { canvas.style.cursor = ray.intersectObject(targetMesh).length ? 'grab' : ''; return; }
    const p = ray.ray.intersectPlane(plane, new THREE.Vector3()); if (!p) return;
    p.sub(offset); setTarget(Math.round(p.x * 200) / 200, Math.round(p.y * 200) / 200, targetYaw());
  });
  const drop = () => { dragging = false; controls.enabled = true; canvas.style.cursor = ''; };
  for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) canvas.addEventListener(ev, drop);
}

// ---------------- arm ----------------
async function loadRobot() {
  const r = await fetch('data/robot.json'); if (!r.ok) throw new Error('robot.json missing');
  snapshot = await r.json();
  const loader = new STLLoader(), cache = new Map();
  robot = new THREE.Group(); robot.name = 'so101';
  const meshes = await Promise.all(snapshot.visuals.map(async v => {
    if (!cache.has(v.mesh)) cache.set(v.mesh, loader.loadAsync(v.mesh));
    const c = new THREE.Color(...v.color.slice(0, 3)), dark = c.getHSL({}).l < .4;
    const mesh = new THREE.Mesh(await cache.get(v.mesh), new THREE.MeshStandardMaterial({color: dark ? 0x2b323b : 0xc9d0d8, roughness: .6, metalness: .05}));
    mesh.name = v.link; mesh.matrixAutoUpdate = false; return mesh;
  }));
  robot.add(...meshes);   // Promise.all keeps the source order: visual i <-> matrix i
  scene.add(robot);
  applyPose(snapshot); robot.updateMatrixWorld(true); home();
  pollArm();
}

function applyPose(packet) {
  if (!packet.model_id || packet.model_id !== snapshot.model_id) throw new Error('model changed - reload');
  const rows = packet.matrices ?? packet.visuals?.map(v => v.matrix);
  if (rows?.length !== robot.children.length) throw new Error('arm model mismatch');
  rows.forEach((m, i) => {
    if (!Array.isArray(m) || m.flat().length !== 16 || !m.flat().every(Number.isFinite)) throw new Error('invalid transform');
    robot.children[i].matrix.set(...m.flat()).scale(new THREE.Vector3(...snapshot.visuals[i].scale));
  });
}

let armLive = false;
async function pollArm() {
  try {
    const r = await fetch('data/live-robot.json', {cache: 'no-store'}); if (!r.ok) throw new Error();
    const live = await r.json();
    if (Math.abs(Date.now() - live.timestamp_ms) > 1500 || live.urdf_sha256 !== snapshot.urdf_sha256) throw new Error();
    applyPose(live); armLive = true;
  } catch { armLive = false; }
  setTimeout(pollArm, 50);
}

// ---------------- station ----------------
async function send(cmd, extra = {}) {
  try {
    const r = await fetch('api/command', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({cmd, ...extra})});
    if (!r.ok) throw new Error();
  } catch { log('command not sent - is server.mjs running?', 'bad'); }
}

const active = () => !!state && Math.abs(Date.now() - state.timestamp_ms) < 2000 && state.phase !== 'ended';
const busy = () => active() && BUSY.has(state.phase);

function updateButtons() {
  const on = active(), idle = on && !busy(), inArea = targetInArea();
  $('cmd-stop').disabled = !on;
  $('cmd-home').disabled = !idle;
  $('cmd-locate').disabled = !(idle && state.can_locate);
  $('cmd-grasp').disabled = !(idle && state.can_grasp);
  $('cmd-place').disabled = !(idle && state.can_place && inArea);
  $('hint').textContent = on && state.can_place && !inArea ? '! target outside the work area' : '';
}

const clock = () => new Date().toLocaleTimeString('en-GB', {hour12: false});
let lastLogged = '';
function log(text, tone = '') {
  const li = document.createElement('li'); li.className = tone;
  li.innerHTML = '<time></time><span></span>'; li.firstChild.textContent = clock(); li.lastChild.textContent = text;
  const list = $('log'); list.prepend(li);
  while (list.children.length > 60) list.lastChild.remove();
}

let imageBusy = false, imageAt = 0;
async function pollStation() {
  try {
    const r = await fetch('data/grasp-state.json', {cache: 'no-store'});
    state = r.ok ? await r.json() : null;
  } catch { state = null; }
  const on = active(), tone = on ? TONE[state.phase] ?? '' : '';
  $('led-link').classList.toggle('on', on);
  $('led-arm').classList.toggle('on', armLive);
  $('led-cam').classList.toggle('on', Date.now() - imageAt < 1500);
  $('status').dataset.tone = tone;
  $('phase').textContent = on ? state.phase.toUpperCase() : 'OFF';
  $('message').textContent = on ? state.message : 'Station not running (python grasp_app.py).';
  const key = on ? `${state.phase}|${state.message}` : 'off';
  if (key !== lastLogged) { lastLogged = key; log(on ? state.message : 'station offline', tone === 'bad' ? 'bad' : state?.phase === 'ready' ? 'ok' : ''); }
  updateButtons();

  const img = $('camera');
  if (on && !imageBusy) { imageBusy = true; img.src = `data/camera.jpg?t=${Date.now()}`; }
  if (!on && img.hasAttribute('src')) img.removeAttribute('src');
  $('cam-age').textContent = Date.now() - imageAt < 1500 ? 'live' : '';

  const c = on ? state.cube : null;
  cubeBox.visible = !!c;
  if (c) { cubeBox.position.set(c.q[0] / 1000, c.q[1] / 1000, (c.z_mm ?? 12.5) / 1000); cubeBox.rotation.z = c.q[2] * Math.PI / 180; }
  $('cube-x').textContent = c ? fmt(c.q[0]) : '—';
  $('cube-y').textContent = c ? fmt(c.q[1]) : '—';
  $('cube-yaw').textContent = c ? fmt(((c.q[2] % 90) + 90) % 90, 0) : '—';
  setTimeout(pollStation, 200);
}

function initStation() {
  const place = () => { const [x, y] = targetXY(); send('place', {x_mm: x * 1000, y_mm: y * 1000, yaw_deg: targetYaw()}); };
  const actions = {stop: () => send('stop'), locate: () => send('locate'), home: () => send('home'), grasp: () => send('grasp'), place};
  for (const [name, fn] of Object.entries(actions)) $(`cmd-${name}`).onclick = fn;
  const keys = {l: 'locate', r: 'home', g: 'grasp', p: 'place'};
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { send('stop'); return; }
    if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (keys[k] && !$(`cmd-${keys[k]}`).disabled) { e.preventDefault(); actions[keys[k]](); return; }
    const turn = {q: 1, e: -1}[k];
    if (turn) { e.preventDefault(); setTarget(...targetXY(), targetYaw() + turn * (e.shiftKey ? 1 : 5)); }
  });
  const img = $('camera');
  img.onload = () => { imageBusy = false; imageAt = Date.now(); };
  img.onerror = () => { imageBusy = false; img.removeAttribute('src'); };
  const tick = () => { $('clock').textContent = clock(); }; tick(); setInterval(tick, 500);
  pollStation();
}

// ---------------- start ----------------
async function start() {
  const r = await fetch('data/scene.json'); if (!r.ok) throw new Error('scene.json missing');
  data = await r.json();
  target = moveTarget(createTarget(data), data, ...DEFAULT_TARGET);
  try { const s = localStorage.getItem(storeKey()); if (s) target = validateTarget(JSON.parse(s), data); } catch {}

  scene = new THREE.Scene(); scene.background = new THREE.Color(0x0d1117);
  camera = new THREE.PerspectiveCamera(35, 1, .001, 8); camera.up.set(0, 0, 1);
  renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.domElement.tabIndex = 0;
  viewport.appendChild(renderer.domElement);
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a1f26, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(-.3, -.4, .8); scene.add(key);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.minDistance = .1; controls.maxDistance = 1.8; controls.maxPolarAngle = Math.PI / 2 - .02;

  const table = addTable();
  targetMesh = new THREE.Mesh(new THREE.BoxGeometry(CUBE * 1.3, CUBE * 1.3, CUBE * 1.3), new THREE.MeshBasicMaterial({visible: false}));
  scene.add(targetBox, targetMesh, cubeBox);
  await addArea().catch(() => {});
  showTarget(); home();
  initPointer(table); initStation();
  loadRobot().catch(e => log(e.message, 'bad'));

  new ResizeObserver(() => {
    const w = viewport.clientWidth, h = viewport.clientHeight;
    camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix(); renderer.setSize(w, h);
  }).observe(viewport);
  $('loading').remove();
  $('view-orbit').onclick = () => home(); $('view-top').onclick = () => home(true);
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

  // Read-only hooks for the browser tests.
  window.station = {
    state: () => clone(state), target: () => ({xy_mm: targetXY().map(v => v * 1000), yaw_deg: targetYaw()}), targetInArea,
    cube: () => ({visible: cubeBox.visible, position: cubeBox.position.toArray()}), areaCells: () => areaCells.length,
    robotParts: () => robot?.children.length ?? 0, armLive: () => armLive,
    targetScreenPoint: () => { const p = targetBox.position.clone().project(camera), r = renderer.domElement.getBoundingClientRect(); return {x: r.left + (p.x * .5 + .5) * r.width, y: r.top + (-p.y * .5 + .5) * r.height}; },
  };
}
start().catch(e => { $('loading').textContent = `3D view failed to start: ${e.message}`; console.error(e); });
