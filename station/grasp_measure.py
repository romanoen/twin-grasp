"""Grasp fine calibration from the wrist camera, independent of the camera-mount calibration.

While the gripper closes, the moving jaw pushes the cube against the fixed jaw: the cube travels exactly its
clearance to the fixed jaw (planned FIXED_CLEARANCE_MM = 7.5 mm) along the tool x axis, and the arm does not move.
The cube shift between the image just before and just after closing therefore measures the real clearance.
Over several grasps with different jaw directions, the clearances determine a constant 2D table-frame bias b of
the grasp point: b . x_i = s_i - 7.5 (x_i = tool x axis in the table plane). Correction: command c - b.
The model is used only for the image direction of the push and the pixel/mm scale at the cube (small effect).
"""
import json, time
from pathlib import Path
import cv2
import numpy as np
from grasp_plan import FIXED_CLEARANCE_MM, GRASP_IN_TOOL_MM

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT/'data/grasp_finecal/samples.json'
OFFSET = ROOT/'config/grasp_offset.json'
MIN_SAMPLES = 6
MIN_DIRECTION_EIG = .15        # smallest eigenvalue of mean(x x^T): jaw directions must span the plane
VALID_SHIFT_MM = (.5, 16.)
# Close-up cube colour (09.11. images): cube hue 0-1/179, saturation ~190-220; wooden table hue 10-15, saturation 60-140.
# The broad scene filter (config/cube_features.json, hue <= 12, sat >= 100) also takes the table in close-ups.
CUBE_HUE_MAX, CUBE_HUE_WRAP, CUBE_SAT_MIN, CUBE_VAL_MIN = 5, 172, 150, 40


def cube_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (((h <= CUBE_HUE_MAX) | (h >= CUBE_HUE_WRAP)) & (s >= CUBE_SAT_MIN) & (v >= CUBE_VAL_MIN)).astype(np.uint8)*255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def push_geometry(model, raw, K, dist):
    """Image point of the nominal cube top centre, unit image direction of +x_tool (push direction) and px per mm."""
    X = model.camera_transform; tool = model.link_transform(model.joint_degrees(raw))
    cam_tool = np.linalg.inv(X)            # tool frame -> camera frame
    top = GRASP_IN_TOOL_MM/1000+[0, 0, -.0125]   # cube top centre (tool z points down towards the table)
    pts = np.array([top, top+[.001, 0, 0]])
    cam = (cam_tool[:3, :3]@pts.T+cam_tool[:3, 3:4]).T
    px = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
    d = px[1]-px[0]
    x_table = (model.table_base@tool)[:2, 0]
    return dict(point=px[0], axis=d/np.linalg.norm(d), px_per_mm=float(np.linalg.norm(d)), x_table=(x_table/np.linalg.norm(x_table)).tolist())


def cube_mask(frame, feat, near, radius=300):
    """Red mask of the largest red component near the predicted cube position."""
    mask = cube_red(frame); n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)
    good = [i for i in range(1, n) if stats[i, 4] >= feat['min_component_area_px']]
    if not good: return None
    good = [i for i in good if np.linalg.norm(cent[i]-near) <= radius]   # model prediction may be ~100 px off
    if not good: return None
    i = max(good, key=lambda i: stats[i, 4])                                 # the cube: largest red blob nearby
    return (labels == i).astype(np.float32)


def push_shift_px(before, after, axis, search=(-10., 60.), perp=4):
    """Shift (px) along axis that best maps the 'before' mask onto 'after' (overlap maximum, sub-pixel)."""
    h, w = before.shape; best = []
    for t in np.arange(search[0], search[1]+.01, 1.):
        score = 0.
        for p in range(-perp, perp+1, 2):
            shift = axis*t+np.array([-axis[1], axis[0]])*p
            M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            score = max(score, float((cv2.warpAffine(before, M, (w, h))*after).sum()))
        best.append((t, score))
    ts, ss = np.array(best).T; k = int(np.argmax(ss))
    if 0 < k < len(ss)-1:   # parabola through the peak
        a, b, c = ss[k-1], ss[k], ss[k+1]; den = a-2*b+c
        return float(ts[k]+(.5*(a-c)/den if den else 0.)), float(ss[k]/max(before.sum(), 1))
    return float(ts[k]), float(ss[k]/max(before.sum(), 1))


def measure(model, raw, before_img, after_img, feat, K, dist):
    g = push_geometry(model, raw, K, dist)
    mb, ma = cube_mask(before_img, feat, g['point']), cube_mask(after_img, feat, g['point'])
    if mb is None or ma is None: return dict(valid=False, reason='cube not found in grasp image')
    ys, xs = np.nonzero(mb+ma); m = int(18*g['px_per_mm'])+10            # crop: speed, room for a 16 mm push
    y0, y1, x0, x1 = max(ys.min()-m, 0), ys.max()+m, max(xs.min()-m, 0), xs.max()+m
    t, overlap = push_shift_px(mb[y0:y1, x0:x1], ma[y0:y1, x0:x1], g['axis'], search=(-1.*g['px_per_mm'], (VALID_SHIFT_MM[1]+2)*g['px_per_mm']))
    s = t/g['px_per_mm']
    valid = VALID_SHIFT_MM[0] <= s <= VALID_SHIFT_MM[1] and overlap > .5
    return dict(valid=bool(valid), clearance_mm=float(s), shift_px=t, overlap=overlap, px_per_mm=g['px_per_mm'], x_table=g['x_table'],
                reason=None if valid else f'implausible measurement (clearance {s:.1f} mm, overlap {overlap:.2f})')


def solve_bias(samples):
    v = [s for s in samples if s['valid']]
    if len(v) < MIN_SAMPLES: return dict(ready=False, n=len(v), reason=f'{len(v)}/{MIN_SAMPLES} valid grasps')
    # Absolute grasp-point bias: each sample was commanded at c - offset_applied, so s - 7.5 = (b - offset_applied) . x
    A = np.array([s['x_table'] for s in v])
    r = np.array([s['clearance_mm']-FIXED_CLEARANCE_MM+np.dot(s.get('offset_applied_mm', [0, 0]), s['x_table']) for s in v])
    resid0 = np.array([s['clearance_mm']-FIXED_CLEARANCE_MM for s in v])
    eig = float(np.linalg.eigvalsh(A.T@A/len(v)).min())
    if eig < MIN_DIRECTION_EIG: return dict(ready=False, n=len(v), reason='vary the cube rotation (grasp directions too similar)')
    b = np.linalg.lstsq(A, r, rcond=None)[0]
    loo = [float(r[i]-A[i]@np.linalg.lstsq(np.delete(A, i, 0), np.delete(r, i), rcond=None)[0]) for i in range(len(v))]
    return dict(ready=True, n=len(v), bias_mm=b.tolist(), before_rms_mm=float(np.sqrt(np.mean(resid0**2))),
                after_rms_mm=float(np.sqrt(np.mean((r-A@b)**2))), loo_rms_mm=float(np.sqrt(np.mean(np.square(loo)))), direction_eig=eig)


def load_samples():
    return json.loads(SAMPLES.read_text()) if SAMPLES.exists() else []


def add_sample(sample):
    SAMPLES.parent.mkdir(parents=True, exist_ok=True); s = load_samples()+[sample]; SAMPLES.write_text(json.dumps(s, indent=1, default=float)+'\n'); return s


def load_offset():
    return np.array(json.loads(OFFSET.read_text())['bias_mm']) if OFFSET.exists() else np.zeros(2)


def apply_offset(result, model_id):
    """The fit is already absolute (accounts for the offset each sample was taken with)."""
    total = np.array(result['bias_mm'])
    OFFSET.write_text(json.dumps(dict(bias_mm=total.tolist(), applied=time.strftime('%Y-%m-%d %H:%M'), model_id=model_id, from_fit=result,
                                      meaning='Grasp point bias in table frame (mm); grasp_app commands cube centre minus this.'), indent=2)+'\n')
    return total
