"""Separate calibration of the arm->camera chain, used only for camera-based model views (e.g. the green model cube
in the wrist-camera image). The grasping/tool chain stays on the active tip+camera calibration (robot_model).

Why separate: the tip-touch calibration (v3) trades camera consistency (4.8 mm) for tip accuracy; a camera-only fit of
the shared parameters (3 joint zero offsets, camera mount, table<-base) is ~2 mm on held-out camera poses. Both
cannot hold at once with the URDF model - the known, unresolved camera-vs-tip conflict.

fit:  python camera_chain.py  -> config/camera_chain.json
Data: multipose_20261108 (fit), multipose_confirm_20261108 (check) and tag observations the grasp station records at
its observation poses (data/camera_obs/*/*.json; every 3rd visit held out as check).
"""
import glob, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from robot_model import load_robot_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'config/camera_chain.json'
OBS = ROOT/'data/camera_obs'
OJ = ('shoulder_lift', 'elbow_flex', 'wrist_flex')


def pose(vector):
    """4x4 transform from a rotation vector (3) + translation (3)."""
    t = np.eye(4); t[:3, :3] = cv2.Rodrigues(np.asarray(vector[:3], dtype=float))[0]; t[:3, 3] = vector[3:]; return t


def mean_pose(transforms):
    """Chordal mean rotation, median translation."""
    transforms = np.array(transforms); t = np.eye(4)
    u, _, vt = np.linalg.svd(np.sum(transforms[:, :3, :3], axis=0)); t[:3, :3] = u@np.diag([1, 1, np.linalg.det(u@vt)])@vt
    t[:3, 3] = np.median(transforms[:, :3, 3], axis=0); return t


class CameraChain:
    """T_camera_table(raw) from the camera calibration; falls back to the active model if none is stored/valid."""
    def __init__(self, model):
        self.model, self.data = model, None
        if OUT.exists():
            d = json.loads(OUT.read_text())
            if d.get('model_id') == model.model_id: self.data = d
        if self.data:
            self.delta = np.array(self.data['joint_offset_delta_deg']); self.X = np.array(self.data['T_tool_camera'])
            self.table_base = np.array(self.data['T_table_base'])

    def camera_table(self, raw):
        m = self.model
        if not self.data:
            return np.linalg.inv(m.table_base@m.link_transform(m.joint_degrees(raw))@m.camera_transform)
        a = m.joint_degrees(raw)
        for n, d in zip(OJ, self.delta): a[n] += d
        return np.linalg.inv(self.table_base@m.link_transform(a)@self.X)


def session(folder):
    out = []
    for f in sorted(glob.glob(f'{ROOT}/data/{folder}/pose_*.json')):
        s = json.loads(Path(f).read_text())['samples']
        out.append(dict(raw={k: float(np.median([x['raw_positions'][k] for x in s])) for k in s[0]['raw_positions']},
                        obs=mean_pose([x['pose_candidates'][0]['T_camera_tag'] for x in s]), source=f))
    return out


def station_obs():
    out = []
    for f in sorted(glob.glob(f'{OBS}/*/*.json')):
        d = json.loads(Path(f).read_text())
        if len(d['T_camera_tag']) >= 3: out.append(dict(raw=d['raw'], obs=mean_pose(d['T_camera_tag']), source=f, view=d.get('view')))
    return out


def residual(model, p, views):
    X = pose(p[3:9])@model.camera_transform; B = pose(p[9:15])@np.linalg.inv(model.table_base); e = []
    for v in views:
        a = model.joint_degrees(v['raw'])
        for n, d in zip(OJ, p[:3]): a[n] += np.degrees(d)
        est = model.link_transform(a)@X@v['obs']
        e.append(np.r_[(est[:3, 3]-B[:3, 3])*1000, Rotation.from_matrix(B[:3, :3].T@est[:3, :3]).as_rotvec()*100])
    return np.array(e).reshape(-1, 6)


def stats(e):
    if not len(e): return None
    n = np.linalg.norm(e[:, :3], axis=1); return dict(n=len(e), rms_mm=float(np.sqrt(np.mean(n*n))), max_mm=float(n.max()))


def main():
    model = load_robot_model()
    fit_v, check_v = session('multipose_20261108'), session('multipose_confirm_20261108')
    st = station_obs(); fit_v += [v for i, v in enumerate(st) if i % 3 != 2]; check_st = [v for i, v in enumerate(st) if i % 3 == 2]
    r = least_squares(lambda p: residual(model, p, fit_v).ravel(), np.zeros(15), x_scale='jac', max_nfev=300); p = r.x
    zero = np.zeros(15)
    report = dict(before=dict(fit=stats(residual(model, zero, fit_v)), check_poses=stats(residual(model, zero, check_v)), check_station=stats(residual(model, zero, check_st)) if check_st else None),
                  after=dict(fit=stats(residual(model, p, fit_v)), check_poses=stats(residual(model, p, check_v)), check_station=stats(residual(model, p, check_st)) if check_st else None))
    out = dict(schema_version=1, created=time.strftime('%Y-%m-%d %H:%M'), model_id=model.model_id, purpose='camera-based model views only; grasping uses robot_model',
               joint_offset_delta_deg=np.degrees(p[:3]).tolist(), T_tool_camera=(pose(p[3:9])@model.camera_transform).tolist(),
               T_table_base=np.linalg.inv(pose(p[9:15])@np.linalg.inv(model.table_base)).tolist(), report=report,
               station_observations=len(st), sources_sha256={Path(v['source']).name+('' if 'camera_obs' not in v['source'] else f"@{Path(v['source']).parent.name}"): hashlib.sha256(Path(v['source']).read_bytes()).hexdigest()[:16] for v in fit_v+check_v+check_st})
    OUT.write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
