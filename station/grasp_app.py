"""Grasp station behind the viewer: locate the cube once, then grasp and place it purely from the model.

Stop live_robot_view.py first; this program owns bus and camera and publishes the live joint packet itself.
Viewer files (ui/data): live-robot.json (joints), camera.jpg (wrist camera: detected outline cyan, model
cube green, projected with the separate camera-chain calibration camera_chain.py), grasp-state.json (phase, cube, messages), grasp-area.json (10 observation poses and placeable area).
Commands (viewer -> server.mjs -> ui/data/command.json): home (UI: Reset), locate, grasp, place{x_mm,y_mm,yaw_deg},
stop, open, release, apply_offset. Every motion runs in small steps and checks for 'stop'; stop holds the arm where it is.

Model cube ("Karte"): after 'locate' (all observation poses that see the cube, joint silhouette fit) the cube is
locked in the map and persisted (data/scene_state.json). 'grasp' uses that pose without measuring; after
closing, the cube sits against the fixed jaw, a known place in the tool frame, so the arm motion tells where it is
while held and where it is after 'place'. The arm is repeatable to ~0.7 mm, so poses it produced itself stay
consistent. A quick check from the current view (single image) refuses a grasp if the cube was moved by hand.
"""
import json, threading, time
import cv2
import numpy as np
import arm_io
from arm_io import ARM_JOINTS, ROOT, SETTLE_S, Abort, JointBus, detect, move_to
# Extended arm + cube: the shoulder lags ~60 steps against gravity at 10 deg/s. Collisions still exceed this quickly.
arm_io.FOLLOW_LIMIT_RAW = 110
from cube_detect import extract
from cube_fit import cube_vertices, fit, sample_boundary, view_residual
from grasp_area import load_or_compute
import grasp_measure as finecal
from grasp_plan import CUBE_EDGE_MM, GRASP_IN_TOOL_MM, GRIPPER_OPEN_RAW, GRIPPER_SQUEEZE_RAW, path_clear, plan, solve
from live_robot_view import LiveWriter
from camera_chain import OBS as CAMERA_OBS, CameraChain
from robot_model import load_robot_model

DATA = ROOT/'ui/data'
SCENE = ROOT/'data/scene_state.json'
LIFT_MM = 50.
STABLE_N, STABLE_MM = 4, 2.5           # single-view preview is 'stable' when the last N agree within this
MAX_SILHOUETTE_RMS_PX = 4.
LOCATE_PERIOD_S = .4
HELD_MARGIN_RAW = 12                   # gripper stalled this far before the squeeze goal -> something is in the jaws
MOVED_MM = 5.
CHECK_MM = 3.                          # after placing: correct the map when the cameras see the cube this far off
RELIEF_RAW = 30                        # gripper counts (~3 mm at the jaw tip) to relieve the squeeze before lifting
RELEASE_LIFT_MM = 20.                  # lift this far before opening the gripper fully                          # preview (single image, ~+-2.5 mm) this far from the map -> cube was moved
VIEW_OUTLIER_MM = 5.                   # per-view estimate this far from the median is not used in the joint fit
# Cube centre in the tool frame after closing: pushed against the fixed jaw (inner face x = 0), grasp height.
CUBE_IN_TOOL_MM = np.array([-CUBE_EDGE_MM/2, 0., GRASP_IN_TOOL_MM[2]])
# Lamp light: the wooden table (hue ~9, saturation ~110) and the cube's red-tinted shadow (hue 6-9, saturation ~140) fall
# inside the broad cube filter and widen the silhouette (fit rms 11-13 px). The cube itself is hue ~0 (19.11. images):
# station mask = strict red hue only (cube_detect.red_mask seed, no growth). A pale specular strip on the
# edge between two faces split the cube into two blobs (19.11., rms 6-9 px): lower saturation floor + 9 px closing.
CUBE_SEED = dict(seed_hue_low_max=5, seed_min_saturation=85, seed_min_value=60, seed_grow_px=0, seed_close_px=9)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]



def write_json(path, payload):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, default=float)); tmp.replace(path)


class Camera(threading.Thread):
    """Grabs frames continuously so motion loops never block on the camera."""
    def __init__(self, index=0):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION); self.cap.set(3, 1280); self.cap.set(4, 720)
        self.frame, self.stamp, self.running = None, 0., True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok: self.frame, self.stamp = frame, time.monotonic()

    def latest(self):
        return None if self.frame is None or time.monotonic()-self.stamp > 1. else self.frame.copy()


class Station:
    def __init__(self):
        self.model = load_robot_model(); self.area = load_or_compute(self.model)
        self.views = self.area['views_raw']; self.home = self.views[0]; self.view = None
        self.cell_views = {tuple(c['xy']): c['views'] for c in self.area['cells']}
        intr = json.loads((ROOT/'calibration/camera_intrinsics.json').read_text()); self.intr = intr
        self.K, self.dist = np.array(intr['camera_matrix']), np.array(intr['distortion_coefficients'])
        self.feat = dict(json.loads((ROOT/'config/cube_features.json').read_text()), **CUBE_SEED)
        params = cv2.aruco.DetectorParameters(); params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.finder = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
        self.cam = Camera(); self.cam.start()
        self.arm = JointBus(self.model); self.live = LiveWriter(self.model)
        self.camchain = CameraChain(self.model)   # camera-only calibration for the green model cube (grasping uses self.model)
        self.phase, self.message = 'ready', 'Ready - press Reset to move to the home pose'
        self.preview, self.estimates, self.outline, self.precise = None, [], None, None
        self.command_id = self._command()[2]; self.last = dict(live=0., jpg=0., locate=0.)
        self.log = []; self.folder = ROOT/'data/grasp'/time.strftime('%Y%m%d_%H%M%S')
        self.offset = finecal.load_offset(); self.last_measure = None
        g = int(self.arm.bus.read('Present_Position', 'gripper', normalize=False))
        gripping = bool(self.arm.torque_at_connection['gripper'] and GRIPPER_SQUEEZE_RAW+HELD_MARGIN_RAW <= g <= GRIPPER_SQUEEZE_RAW+90)
        self.cube, self.holding = None, gripping                     # cube: dict(q=[x,y,yaw] true table mm, state, source)
        saved = json.loads(SCENE.read_text()) if SCENE.exists() else {}
        if saved.get('model_id') == self.model.model_id and saved.get('cube'):
            if saved['cube']['state'] == 'locked' and not gripping:   # from an earlier run: may have been moved meanwhile
                self.cube = dict(saved['cube'], verified=False, source=saved['cube'].get('source', '')+' - previous run, unconfirmed')
            if saved['cube']['state'] == 'held' and gripping: self.cube = saved['cube']

    # ---------- viewer I/O ----------
    def _command(self):
        try:
            c = json.loads((DATA/'command.json').read_text()); return c.get('cmd'), c, c.get('id')
        except Exception:
            return None, {}, None

    def poll_command(self):
        cmd, c, cid = self._command()
        if cid is not None and cid != self.command_id:
            self.command_id = cid; return cmd, c
        return None, {}

    def raw(self):
        return self.arm.read()['raw_positions']

    def tool(self, raw):
        return self.model.table_base@self.model.link_transform(self.model.joint_degrees(raw))

    def camera_table_arm(self, raw):
        return self.camchain.camera_table(raw)

    def record_tag(self, view, frames=6, timeout=1.5):
        """Tag observations at an observation pose: data for refitting the camera-chain calibration (camera_chain.py)."""
        poses, t0, raw = [], time.monotonic(), self.raw()
        while len(poses) < frames and time.monotonic()-t0 < timeout:
            frame = self.cam.latest()
            found = detect(self.finder, frame, self.intr)[0] if frame is not None else None
            if found: poses.append(found[1][0]['T_camera_tag'])
            self.publish(); time.sleep(.06)
        if len(poses) >= 3:
            folder = CAMERA_OBS/time.strftime('%Y%m%d'); folder.mkdir(parents=True, exist_ok=True)
            write_json(folder/f"{time.strftime('%H%M%S')}_v{view+1:02d}.json", dict(view=view, raw=raw, raw_after=self.raw(), T_camera_tag=poses, model_id=self.model.model_id))

    def held_cube_pose(self, raw):
        """True-table pose of the cube in the gripper from the arm model (+ fine-calibration bias)."""
        T = self.tool(raw); c = (T@np.r_[CUBE_IN_TOOL_MM/1000, 1])[:3]*1000
        return [float(c[0]+self.offset[0]), float(c[1]+self.offset[1]), float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))], float(c[2])

    def save_scene(self):
        write_json(SCENE, dict(saved=time.strftime('%Y-%m-%d %H:%M:%S'), model_id=self.model.model_id, cube=self.cube, offset_mm=self.offset.tolist()))

    def publish(self, raw=None, force=False):
        now = time.monotonic(); raw = raw or self.raw()
        if self.cube and self.cube['state'] == 'held':
            self.cube['q'], self.cube['z_mm'] = self.held_cube_pose(raw)
        if force or now-self.last['live'] > .05:
            self.live.write(raw, self.arm.torque_at_connection, held=self.arm.held); self.last['live'] = now
        if force or now-self.last['jpg'] > .15:
            frame = self.cam.latest()
            if frame is not None:
                if self.outline is not None and self.view is not None and not self.holding:
                    cv2.polylines(frame, [self.outline.astype(np.int32)], True, (255, 255, 0), 2)
                if self.cube is not None:
                    if self.cube['state'] == 'held':
                        # Held cube: camera and gripper are rigidly joined -> tool frame straight into the camera (no arm model).
                        c, h = CUBE_IN_TOOL_MM/1000, CUBE_EDGE_MM/2000
                        verts = np.array([[c[0]+sx*h, c[1]+sy*h, c[2]+sz*h] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
                        T = np.linalg.inv(self.model.camera_transform)
                    else:
                        T = self.camera_table_arm(raw)   # cube on the table: camera-chain calibration
                        verts = cube_vertices(self.cube['q'], .025, 0.)
                    cam = T[:3, :3]@verts.T+T[:3, 3:4]
                    if np.all(cam[2] > .02):
                        pts = cv2.projectPoints(cam.T.copy(), np.zeros(3), np.zeros(3), self.K, self.dist)[0].reshape(-1, 2)
                        if np.all(np.isfinite(pts)) and np.abs(pts).max() < 1e4:
                            for a, b in EDGES: cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (60, 220, 60), 3)
                small = cv2.resize(frame, (640, 360)); ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    tmp = DATA/'camera.tmp'; tmp.write_bytes(buf.tobytes()); tmp.replace(DATA/'camera.jpg')
            write_json(DATA/'grasp-state.json', dict(timestamp_ms=time.time()*1000, model_id=self.model.model_id, phase=self.phase,
                       message=self.message,
                       held=self.arm.held, view=self.view, views=len(self.views), at_home=self.view == 0, holding_cube=self.holding,
                       cube=self.cube, preview=self.preview, precise=self.precise, can_grasp=self.can_grasp(), can_place=self.can_place(),
                       can_locate=not self.holding, finecal=self.finecal_state()))
            self.last['jpg'] = now

    def tick(self):
        """Pump for motions: publish, and True if the viewer asked to stop."""
        self.publish(); return self.poll_command()[0] == 'stop'

    # ---------- cube ----------
    def stable(self):
        if len(self.estimates) < STABLE_N: return False
        xy = np.array([e[:2] for e in self.estimates[-STABLE_N:]]); return bool(np.ptp(xy, axis=0).max() <= STABLE_MM)

    def can_grasp(self):
        return bool(self.arm.held and not self.holding and self.cube and self.cube['state'] == 'locked' and self.cube.get('reachable'))

    def can_place(self):
        return bool(self.arm.held and self.holding and self.cube and self.cube['state'] == 'held')

    def reachable(self, xy, yaw):
        return bool(plan(self.model, self.home, np.array(xy)-self.offset, yaw)['ok'])

    def single_view(self, frame):
        found = detect(self.finder, frame, self.intr)[0]
        if not found: return None, None
        info, _, _ = extract(frame, self.feat, cv2.contourArea(np.array(found[0], np.float32)))
        if 'silhouette_contour_px' not in info: return None, found
        c = np.array(info['silhouette_contour_px'], float)
        return dict(number=1, role='fit', contour=c, observed_samples=sample_boundary(c), transform=np.array(found[1][0]['T_camera_tag']), matrix=self.K, distortion=self.dist), found

    def observe(self, frames=5, timeout=2.5):
        """Fit-ready cube views (tag-based camera pose) from the current, still pose."""
        out, t0 = [], time.monotonic()
        while len(out) < frames and time.monotonic()-t0 < timeout:
            self.publish(); frame = self.cam.latest()
            if frame is None: continue
            v, _ = self.single_view(frame)
            if v: v['number'] = len(out)+1; out.append(v); time.sleep(.08)
        return out

    def locate_preview(self):
        """Single-image estimate from the current view: live preview, and the 'was it moved?' check."""
        frame = self.cam.latest()
        if frame is None: return
        v, found = self.single_view(frame)
        if v is None:
            self.outline = None; self.estimates.clear(); self.preview = None; return
        self.outline = v['contour']
        q, _, _ = fit([v, dict(v, number=2)], .025, 0.)
        if float(np.sqrt(np.mean(view_residual(q, v, .025, 0.)**2))) > MAX_SILHOUETTE_RMS_PX: return
        self.estimates = (self.estimates+[q])[-8:]
        med = np.median(np.array(self.estimates[-STABLE_N:]), axis=0)
        dev = float(np.linalg.norm(med[:2]-self.cube['q'][:2])) if self.cube and self.cube['state'] == 'locked' else None
        self.preview = dict(q=[float(med[0]), float(med[1]), float(q[2])], stable=self.stable(), deviation_from_map_mm=dev)

    def views_for(self, xy):
        cells = np.array(self.area['cells_mm'])
        i = int(np.argmin(np.linalg.norm(cells-np.array(xy[:2]), axis=1)))
        return self.cell_views[tuple(cells[i])] if np.linalg.norm(cells[i]-xy[:2]) <= 15 else list(range(len(self.views)))

    # ---------- motions ----------
    def gripper(self, goal):
        bus = self.arm.bus
        present = int(bus.read('Present_Position', 'gripper', normalize=False))
        bus.write('Goal_Position', 'gripper', present, normalize=False, num_retry=1); bus.enable_torque(['gripper'], num_retry=1)
        steps = max(1, int(abs(goal-present)/(4096/360*1.2)))
        for i in range(1, steps+1):
            bus.write('Goal_Position', 'gripper', int(round(present+(goal-present)*i/steps)), normalize=False, num_retry=1)
            time.sleep(.04)
            if self.tick():
                bus.write('Goal_Position', 'gripper', int(bus.read('Present_Position', 'gripper', normalize=False)), normalize=False, num_retry=1)
                raise Abort('Stopped')
        self.wait(.6)
        return int(bus.read('Present_Position', 'gripper', normalize=False))

    def wait(self, seconds):
        t0 = time.monotonic()
        while time.monotonic()-t0 < seconds:
            if self.tick(): raise Abort('Stopped')
            time.sleep(.02)

    def go(self, goal, label):
        start = self.raw(); reason = path_clear(self.model, start, dict(start, **{k: goal[k] for k in ARM_JOINTS}))
        if reason: raise Abort(f'{label}: {reason}')
        move_to(self.arm, goal, self.tick)

    def go_view(self, i):
        self.view = None; self.go(self.views[i], f'observation pose {i+1}'); self.wait(SETTLE_S); self.view = i
        self.estimates.clear(); self.preview = None; self.record_tag(i)

    def hold(self):
        if not self.arm.held: self.arm.hold()

    def do_home(self):
        self.hold(); self.phase, self.message = 'moving', 'Moving to home pose ...'; self.go_view(0)
        self.phase = 'ready'
        self.message = ('Cube in gripper - set a target, then Place' if self.holding else
                        'Cube in map - ready to grasp' if self.cube else 'Put the cube in the work area, then Locate')

    def do_locate(self):
        """Visit observation poses, use every pose that sees the cube, fit all images jointly, lock in the map."""
        if self.holding: raise Abort('Cube is in the gripper')
        self.hold(); self.phase = 'locating'
        start = self.view if self.view is not None else 0
        todo = [start]+[i for i in range(len(self.views)) if i != start]
        per_view, obs_by_view, narrowed = [], {}, False
        while todo:
            i = todo.pop(0); self.message = f'Locating: pose {i+1} (cube seen in {len(per_view)}) ...'
            if i != self.view: self.go_view(i)
            obs = self.observe(); frame = self.cam.latest()
            if frame is not None:
                self.folder.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(self.folder/f'locate_{time.strftime("%H%M%S")}_v{i:02d}.jpg'), frame)
            if len(obs) >= 3:
                q, _, _ = fit(obs, .025, 0.); per_view.append(dict(view=i, q=q.tolist(), frames=len(obs))); obs_by_view[i] = obs
                if not narrowed:   # first sighting: only visit poses that can see the cube there
                    narrowed = True; seeing = set(self.views_for(q)); todo = [j for j in todo if j in seeing]
        if not per_view:
            self.go_view(0); raise Abort('Cube not seen from any pose - put it in the green area')
        med = np.median(np.array([v['q'][:2] for v in per_view]), axis=0)
        used = [v for v in per_view if np.linalg.norm(np.array(v['q'][:2])-med) <= VIEW_OUTLIER_MM] or per_view
        allobs = [o for v in used for o in obs_by_view[v['view']]]
        q, _, _ = fit(allobs, .025, 0.)
        rms = float(np.sqrt(np.mean(np.concatenate([view_residual(q, o, .025, 0.) for o in allobs])**2)))
        diffs = [float(np.linalg.norm(np.array(v['q'][:2])-q[:2])) for v in used]
        self.precise = dict(q=q.tolist(), views=[v['view'] for v in used], rejected=[v['view'] for v in per_view if v not in used],
                            per_view=per_view, rms_px=rms, max_view_difference_mm=max(diffs), mean_view_difference_mm=float(np.mean(diffs)))
        self.log.append(dict(event='locate', **self.precise))
        if rms > MAX_SILHOUETTE_RMS_PX: self.go_view(0); raise Abort(f'Poor silhouette fit ({rms:.1f} px) - map not updated')
        self.cube = dict(q=[float(v) for v in q], state='locked', source=f'located from {len(used)} poses', time=time.strftime('%H:%M:%S'),
                         reachable=self.reachable(q[:2], q[2]))
        self.save_scene(); self.go_view(0); self.phase = 'ready'
        self.message = (f"Cube located at x {q[0]:.1f} / y {q[1]:.1f} mm from {len(used)} poses (view spread max {max(diffs):.1f} mm)"
                        + ('' if self.cube['reachable'] else ' - not graspable there'))

    def check_not_moved(self):
        """If the current view sees the cube, it must still be where the map says."""
        if self.view is None:
            if self.cube.get('verified') is False: raise Abort('Map is from a previous run - please Locate')
            return
        self.estimates.clear(); t0 = time.monotonic()
        while time.monotonic()-t0 < 2.5 and not (self.preview and self.preview['stable']):
            self.locate_preview(); self.wait(.15)
        if self.preview and self.preview['stable'] and (self.preview['deviation_from_map_mm'] or 0) > MOVED_MM:
            raise Abort(f"Cube is {self.preview['deviation_from_map_mm']:.0f} mm off the map - moved? Please Locate again")
        if self.cube.get('verified') is False:   # map from an earlier run: only grasp if the camera confirms it
            if not (self.preview and self.preview['stable']): raise Abort('Map is from a previous run and the camera cannot confirm it - please Locate')
            self.cube['verified'] = True; self.save_scene()

    def do_grasp(self):
        if not self.can_grasp(): raise Abort('Locate first (cube must be in the map and graspable)')
        self.check_not_moved()
        q = np.array(self.cube['q']); p = plan(self.model, self.raw(), q[:2]-self.offset, q[2])   # fine-calibration bias
        if not p['ok']: raise Abort('Not graspable: '+p['reason'])
        self.log.append(dict(event='grasp', cube=self.cube, plan={k: p[k] for k in ('hover', 'descent', 'grasp', 'cube_centre_mm', 'yaw_deg')}))
        self.view = None; self.phase = 'grasping'
        self.message = 'Moving above the cube ...'; self.go(p['hover']['raw'], 'hover'); self.gripper(GRIPPER_OPEN_RAW); self.wait(SETTLE_S)
        self.message = 'Descending ...'
        for raw in p['descent']: move_to(self.arm, raw, self.tick)
        self.wait(SETTLE_S)
        self.wait(.3); before, grasp_raw = self.cam.latest(), self.raw()
        self.message = 'Closing gripper ...'; reached = self.gripper(GRIPPER_SQUEEZE_RAW)
        self.wait(.3); after = self.cam.latest(); self.record_push(before, after, grasp_raw, q.tolist())
        self.holding = reached-GRIPPER_SQUEEZE_RAW >= HELD_MARGIN_RAW; self.outline = None
        self.log.append(dict(event='closed', gripper_raw=reached, holding=self.holding))
        if self.holding:
            self.cube = dict(state='held', source='in gripper (from arm motion)', time=time.strftime('%H:%M:%S'))
            self.cube['q'], self.cube['z_mm'] = self.held_cube_pose(self.raw())
        else:
            self.cube = None
        self.save_scene()
        self.message = 'Lifting ...'
        up = solve(self.model, p['grasp'], np.array(p['cube_centre_mm'])+[0, 0, LIFT_MM], p['yaw_deg'])
        if not up['ok']: raise Abort('No solution for lifting')
        move_to(self.arm, up['raw'], self.tick); self.wait(.5)
        self.do_home()
        if not self.holding: self.message = 'Gripper closed on nothing - please Locate again'

    def do_place(self, target):
        """Put the held cube down so that its centre lands on target (true table mm) with the given yaw."""
        if not self.can_place(): raise Abort('No cube in the gripper')
        x, y, yaw = float(target['x_mm']), float(target['y_mm']), float(target['yaw_deg'])
        want = np.array([x, y])-self.offset          # model frame
        seed = self.raw(); goal = want.copy(); p = None
        for _ in range(3):   # grasp point that puts the held cube (pushed to the fixed jaw) onto the target
            p = plan(self.model, seed, goal, yaw if p is None else p['yaw_deg'])
            if not p['ok']: raise Abort('Target unreachable: '+p['reason'])
            err = (self.tool(p['grasp'])@np.r_[CUBE_IN_TOOL_MM/1000, 1])[:2]*1000-want
            if np.linalg.norm(err) < .5: break
            goal = goal-err
        self.log.append(dict(event='place', target=target, plan={k: p[k] for k in ('hover', 'descent', 'grasp', 'cube_centre_mm', 'yaw_deg')}))
        self.view = None; self.phase = 'placing'
        self.message = 'Moving to target ...'; self.go(p['hover']['raw'], 'to target')
        self.message = 'Descending ...'
        for raw in p['descent']: move_to(self.arm, raw, self.tick)
        self.wait(SETTLE_S)
        q, _ = self.held_cube_pose(self.raw()); place_raw = self.raw()
        # Gentle release: relieve the squeeze by ~3 mm (nothing dragged along), lift, open fully in the air.
        self.message = 'Releasing ...'; before = self.cam.latest()
        present = int(self.arm.bus.read('Present_Position', 'gripper', normalize=False))
        self.gripper(present+RELIEF_RAW); self.wait(.3); after = self.cam.latest(); self.holding = False
        slip = self.release_slip(before, after, place_raw)
        self.cube = dict(q=q, state='locked', source='placed (from arm motion)', time=time.strftime('%H:%M:%S'), reachable=self.reachable(q[:2], q[2]))
        self.save_scene()
        for h in (RELEASE_LIFT_MM, LIFT_MM):
            up = solve(self.model, p['grasp'], np.array(p['cube_centre_mm'])+[0, 0, h], p['yaw_deg'])
            if up['ok']: move_to(self.arm, up['raw'], self.tick); self.wait(.3)
            if h == RELEASE_LIFT_MM: self.gripper(GRIPPER_OPEN_RAW)
        check = self.recheck(q)
        self.log.append(dict(event='placed', model_q=q, cube=self.cube, release_slip=slip, check=check))
        self.do_home()
        self.message = (f'Placed at x {self.cube["q"][0]:.1f} / y {self.cube["q"][1]:.1f} mm'
                        + (f' - check: {check["deviation_mm"]:.1f} mm from model, map {"corrected" if check["corrected"] else "confirmed"}' if check and check.get('deviation_mm') is not None
                           else ' - check inconclusive (poor silhouette), map from arm motion' if check else ' - check: cube not in view, map from arm motion')
                        + (f'; slipped {slip["shift_mm"]:.1f} mm on release' if slip else ''))

    def release_slip(self, before, after, raw):
        """Cube shift (mm) between the images just before and after relieving the grip (same arm pose)."""
        if before is None or after is None: return None
        g = finecal.push_geometry(self.model, raw, self.K, self.dist)
        mb, ma = finecal.cube_mask(before, self.feat, g['point']), finecal.cube_mask(after, self.feat, g['point'])
        self.folder.mkdir(parents=True, exist_ok=True); n = len(list(self.folder.glob('release_*_before.jpg')))+1
        cv2.imwrite(str(self.folder/f'release_{n:02d}_before.jpg'), before); cv2.imwrite(str(self.folder/f'release_{n:02d}_after.jpg'), after)
        if mb is None or ma is None: return None
        cb, ca = np.argwhere(mb).mean(0)[::-1], np.argwhere(ma).mean(0)[::-1]; d = (ca-cb)/g['px_per_mm']
        return dict(shift_mm=float(np.linalg.norm(d)), along_push_axis_mm=float(d@g['axis']), images=f'release_{n:02d}', area_ratio=float(ma.sum()/max(mb.sum(), 1)))

    def recheck(self, q):
        """After placing: look from up to 2 nearby observation poses that see the cube; correct the map if > CHECK_MM off."""
        cur = self.raw(); views = sorted(self.views_for(np.array(q)), key=lambda i: sum(abs(self.views[i][k]-cur[k]) for k in ARM_JOINTS))[:2]
        obs, used = [], []
        for i in views:
            self.message = f'Checking from pose {i+1} ...'; self.go_view(i); o = self.observe()
            if len(o) >= 3: obs += o; used.append(i)
        if not obs: return None
        q2, _, _ = fit(obs, .025, 0.)
        rms = float(np.sqrt(np.mean(np.concatenate([view_residual(q2, o, .025, 0.) for o in obs])**2)))
        if rms > MAX_SILHOUETTE_RMS_PX: return dict(views=used, rms_px=rms, deviation_mm=None, corrected=False)
        dev = float(np.linalg.norm(np.array(q2[:2])-np.array(q[:2]))); corrected = dev > CHECK_MM
        if corrected:
            self.cube.update(q=[float(v) for v in q2], source=f'placed, re-located ({dev:.1f} mm corrected)', reachable=self.reachable(q2[:2], q2[2]))
            self.save_scene()
        return dict(views=used, measured_q=[float(v) for v in q2], rms_px=rms, deviation_mm=dev, corrected=corrected)

    def finecal_state(self):
        fit_ = finecal.solve_bias(finecal.load_samples())
        return dict(samples=fit_['n'], fit=fit_, applied_mm=self.offset.tolist(), last=self.last_measure, planned_clearance_mm=finecal.FIXED_CLEARANCE_MM)

    def record_push(self, before, after, raw, q):
        if before is None or after is None: return
        self.folder.mkdir(parents=True, exist_ok=True); n = len(list(self.folder.glob('push_*_before.jpg')))+1
        cv2.imwrite(str(self.folder/f'push_{n:02d}_before.jpg'), before); cv2.imwrite(str(self.folder/f'push_{n:02d}_after.jpg'), after)
        r = finecal.measure(self.model, raw, before, after, self.feat, self.K, self.dist)
        r.update(cube_q=q, raw=raw, offset_applied_mm=self.offset.tolist(), images=str((self.folder/f'push_{n:02d}').relative_to(ROOT)), model_id=self.model.model_id)
        finecal.add_sample(r); self.last_measure = {k: r[k] for k in ('valid', 'clearance_mm', 'reason') if k in r}
        self.log.append(dict(event='push_measure', **{k: v for k, v in r.items() if k != 'raw'}))

    def do_apply_offset(self):
        fit_ = finecal.solve_bias(finecal.load_samples())
        if not fit_['ready']: raise Abort('Correction not ready: '+fit_['reason'])
        self.offset = finecal.apply_offset(fit_, self.model.model_id)
        self.message = f'Correction applied: x {-self.offset[0]:+.1f} / y {-self.offset[1]:+.1f} mm'

    def do_open(self):
        self.gripper(GRIPPER_OPEN_RAW)
        if self.holding: self.cube = None; self.save_scene()     # taken out by hand: position unknown
        self.holding = False; self.phase, self.message = 'ready', 'Gripper open' + (' - put the cube down, then Locate' if not self.cube else '')

    def do_release(self):
        self.arm.release(); self.view = None; self.phase, self.message = 'off', 'Motors off'

    # ---------- main ----------
    def run(self):
        print(f'Grasp station running. {len(self.views)} poses, work area {len(self.area["cells_mm"])} cells. Map: {self.cube}. Torque: {self.arm.torque_at_connection}', flush=True)
        actions = dict(home=self.do_home, locate=self.do_locate, grasp=self.do_grasp, open=self.do_open, release=self.do_release, apply_offset=self.do_apply_offset)
        try:
            while True:
                cmd, c = self.poll_command()
                if cmd in actions or cmd == 'place':
                    try:
                        self.do_place(c) if cmd == 'place' else actions[cmd]()
                    except Abort as error:
                        self.phase, self.message = 'stopped', str(error)+' - arm holding. Reset returns home.'
                        self.view = None
                    except Exception as error:   # programming/communication error: stop the motion, keep holding, stay alive
                        import traceback; traceback.print_exc()
                        try: self.arm.bus.sync_write('Goal_Position', {k: v for k, v in self.raw().items() if k in ARM_JOINTS}, normalize=False)
                        except Exception: pass
                        self.phase, self.message = 'error', f'Internal error ({type(error).__name__}) - arm holding. Reset returns home.'
                        self.view = None
                    self.log.append(dict(event=cmd, phase=self.phase, message=self.message, raw=self.raw()))
                    self.folder.mkdir(parents=True, exist_ok=True); (self.folder/'log.json').write_text(json.dumps(self.log, indent=1, default=float)+'\n')
                    print(f'{cmd}: {self.message}', flush=True)
                elif cmd == 'stop':
                    self.message = 'Halted.'
                if self.view is not None and not self.holding and self.arm.held and time.monotonic()-self.last['locate'] > LOCATE_PERIOD_S:
                    self.locate_preview(); self.last['locate'] = time.monotonic()
                self.publish(); time.sleep(.02)
        finally:
            self.cam.running = False
            self.phase, self.message = 'ended', 'Grasp station ended'; self.publish(force=True)
            if self.arm.held: print('Note: arm torque stays on. To release: hold the arm, then python station/arm_io.py --release', flush=True)
            self.arm.close()


if __name__ == '__main__':
    import signal, sys
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # background jobs ignore SIGINT; SIGTERM runs the cleanup
    try: Station().run()
    except KeyboardInterrupt: pass
