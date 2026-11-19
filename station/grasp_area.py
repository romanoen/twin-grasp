"""Compute once and save the five observation poses and the placeable table area for grasp_app.py (no hardware).

Reachable cell: a vertical grasp plan exists for cube rotations 0 and 45 deg. Observation pose ("view"): the tag
is fully visible (precise camera pose per image) and the arm is >= 15 cm above the table. View 0 is the proven
home pose; views 1-4 are found by greedy random search to (a) cover reachable cells not seen yet and (b) give
cells a second view from a direction >= MIN_VIEW_ANGLE_DEG different (multi-view cube fit). A cell is seen when
the whole cube (0 and 45 deg) projects inside the image, MARGIN_PX from the border and above the jaws.
Output: ui/data/grasp-area.json (keyed to model id, home pose and settings).
"""
import hashlib, json
from pathlib import Path
import cv2
import numpy as np
from cube_fit import cube_vertices
from grasp_plan import clearance, path_clear, plan
from robot_model import load_robot_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'ui/data/grasp-area.json'
# View 0 (home): searched 09.11. to see the tag plus much of the area; used in the first grasps. Tool ~20 cm high.
HOME_RAW = {'shoulder_pan': 2338, 'shoulder_lift': 1198, 'elbow_flex': 2047, 'wrist_flex': 3126, 'wrist_roll': 1514, 'gripper': 2464}
VIEWS = 10
STEP_MM = 10
X_RANGE, Y_RANGE = (-120, 260), (50, 320)
IMAGE = (1280, 720); MARGIN_PX = 50; JAW_TOP_PX = 400     # margin covers ~27 px model camera error; jaws below ~420 px
MIN_VIEW_ANGLE_DEG = 30.
TAG_MARGIN_PX = 15        # tag only needs to be fully detectable
MIN_TOOL_HEIGHT_MM = 150.
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
TAG = np.array([[-.04, -.04, 0], [.04, -.04, 0], [.04, .04, 0], [-.04, .04, 0]])


def camera_table(model, raw):
    """T_camera_table from the arm model."""
    return np.linalg.inv(model.table_base@model.link_transform(model.joint_degrees(raw))@model.camera_transform)


def in_view(points, T, K, dist, margin=MARGIN_PX):
    cam = (T[:3, :3]@points.T+T[:3, 3:4]).T
    if np.any(cam[:, 2] <= .05): return False
    px = cv2.projectPoints(points, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0].reshape(-1, 2)
    return bool(np.all(px[:, 0] > margin) and np.all(px[:, 0] < IMAGE[0]-margin) and np.all(px[:, 1] > margin) and np.all(px[:, 1] < JAW_TOP_PX))


def key(model):
    return hashlib.sha256(json.dumps(dict(model=model.model_id, home=HOME_RAW, views=VIEWS, step=STEP_MM, x=X_RANGE, y=Y_RANGE,
                                          img=[MARGIN_PX, JAW_TOP_PX, TAG_MARGIN_PX], angle=MIN_VIEW_ANGLE_DEG, schema=3)).encode()).hexdigest()


class Coverage:
    def __init__(self, model, cells):
        intr = json.loads((ROOT/'calibration/camera_intrinsics.json').read_text())
        self.K, self.dist, self.model, self.cells = np.array(intr['camera_matrix']), np.array(intr['distortion_coefficients']), model, cells
        self.cubes = [np.vstack([cube_vertices((x, y, a), .025, 0.) for a in (0, 45)]) for x, y in cells]
        cal = model.calibration
        self.limits = {k: (cal[k]['range_min']+60, cal[k]['range_max']-60) for k in JOINTS}

    def seen(self, raw):
        """None if the pose is unusable, else (bool array of seen cells, unit viewing directions camera->cell)."""
        if any(not lo <= raw[k] <= hi for k, (lo, hi) in self.limits.items()) or clearance(self.model, raw): return None
        if (self.model.table_base@self.model.link_transform(self.model.joint_degrees(raw)))[2, 3]*1000 < MIN_TOOL_HEIGHT_MM: return None
        T = camera_table(self.model, raw)
        if not in_view(TAG, T, self.K, self.dist, TAG_MARGIN_PX): return None
        cam = np.linalg.inv(T)[:3, 3]
        mask = np.array([in_view(c, T, self.K, self.dist) for c in self.cubes])
        d = np.array([[x/1000, y/1000, .0125] for x, y in self.cells])-cam
        return mask, d/np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def score(views):
        """Per cell: seen (1) + second direction (0.7) + third direction (0.4), directions pairwise >= MIN_VIEW_ANGLE_DEG apart."""
        n = len(views[0][0]); c = np.cos(np.radians(MIN_VIEW_ANGLE_DEG))
        once = np.zeros(n, bool); twice = np.zeros(n, bool); thrice = np.zeros(n, bool)
        far = {}
        for i in range(len(views)):
            once |= views[i][0]
            for j in range(i):
                far[i, j] = views[i][0] & views[j][0] & (np.einsum('ij,ij->i', views[i][1], views[j][1]) < c); twice |= far[i, j]
                for k in range(j):
                    if (j, k) in far: thrice |= far[i, j] & far[j, k] & far[i, k]
        return once.sum()+.7*twice.sum()+.4*thrice.sum(), once, twice, thrice

def search_views(model, cov, seeds, rng, iterations=2200):
    views = [HOME_RAW]; data = [cov.seen(HOME_RAW)]
    for _ in range(VIEWS-1):
        best = None
        for seed in seeds:
            cur = dict(seed); s = cov.seen(cur)
            if s is None: continue
            cur_score = Coverage.score(data+[s])[0]
            for it in range(iterations//len(seeds)):
                cand = dict(cur)
                for k in JOINTS: cand[k] = int(cur[k]+rng.normal(0, 70 if it < iterations//len(seeds)//2 else 25))
                s2 = cov.seen(cand)
                if s2 is None or path_clear(model, HOME_RAW, cand) or path_clear(model, cand, HOME_RAW): continue
                sc = Coverage.score(data+[s2])[0]
                if sc >= cur_score: cur, s, cur_score = cand, s2, sc
            if best is None or cur_score > best[0]: best = (cur_score, cur, s)
        views.append(best[1]); data.append(best[2])
    return views, data


def compute(model):
    cache = OUT.with_name('grasp-reach.json'); ckey = hashlib.sha256(json.dumps([model.model_id, HOME_RAW, STEP_MM, X_RANGE, Y_RANGE]).encode()).hexdigest()
    if cache.exists() and json.loads(cache.read_text())['key'] == ckey:
        reach = json.loads(cache.read_text())['cells']
    else:
        reach = [[x, y] for y in range(Y_RANGE[0], Y_RANGE[1]+1, STEP_MM) for x in range(X_RANGE[0], X_RANGE[1]+1, STEP_MM)
                 if all(plan(model, HOME_RAW, (x, y), a)['ok'] for a in (0., 45.))]
        cache.write_text(json.dumps(dict(key=ckey, cells=reach)))
    cov = Coverage(model, reach)
    seeds = [HOME_RAW]+[dict(p['target_raw'], gripper=HOME_RAW['gripper'])
                        for p in json.loads((ROOT/'data/multipose_confirm_20261108/plan.json').read_text())['poses']]
    views, data = search_views(model, cov, seeds, np.random.default_rng(1109))
    _, once, twice, thrice = Coverage.score(data)
    seen_by = [[i for i, (m, _) in enumerate(data) if m[c]] for c in range(len(reach))]
    pairs = {f'{i}-{j}': path_clear(model, views[i], views[j]) is None for i in range(VIEWS) for j in range(VIEWS) if i != j}
    cells = [dict(xy=reach[c], views=seen_by[c], multi=bool(twice[c]), directions=int(once[c])+int(twice[c])+int(thrice[c])) for c in range(len(reach)) if once[c]]
    return dict(schema_version=2, key=key(model), model_id=model.model_id, home_raw=HOME_RAW, views_raw=views, direct_paths=pairs,
                step_mm=STEP_MM, reachable_cells=len(reach), cells=cells, cells_mm=[c['xy'] for c in cells],
                meaning='Cube centre cells (table/tag frame, mm) graspable from above and visible (with the tag) from >= 1 of the views; multi/directions = seen from two/three directions pairwise >= 30 deg apart.')


def load_or_compute(model):
    if OUT.exists():
        area = json.loads(OUT.read_text())
        if area.get('key') == key(model):
            return area
    area = compute(model); OUT.write_text(json.dumps(area, indent=1)+'\n'); return area


if __name__ == '__main__':
    a = load_or_compute(load_robot_model())
    print(f"reachable {a['reachable_cells']} cells, visible {len(a['cells'])}, from 2 directions {sum(c['multi'] for c in a['cells'])}, from 3 directions {sum(c['directions'] == 3 for c in a['cells'])}")
    for i, v in enumerate(a['views_raw']):
        print(i, v, 'sees', sum(i in c['views'] for c in a['cells']))
    print('direct paths blocked:', [k for k, ok in a['direct_paths'].items() if not ok])
