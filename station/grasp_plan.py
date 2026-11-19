"""Grasp geometry and inverse kinematics for a 25 mm cube on the table (pure computation, no hardware).

Tool frame (gripper_frame_link, from the URDF meshes): +z points out along the jaws, the fixed jaw's inner face is
the plane x = 0 with the jaw body at x > 0, the moving jaw opens towards -x. The cube centre is placed at
GRASP_IN_TOOL_MM: 7.5 mm clearance to the fixed jaw (covers ~5 mm calibration error), jaw tips 3 mm above the table.
Closing pushes the cube against the fixed jaw.
"""
import numpy as np
from scipy.optimize import least_squares

ARM = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
CUBE_EDGE_MM = 25.
FIXED_CLEARANCE_MM = 7.5
TIP_ABOVE_TABLE_MM = 3.
FIXED_TIP_Z_MM = 6.3                       # FIXED_JAW_TIP_IN_TOOL_M z
GRASP_IN_TOOL_MM = np.array([-(CUBE_EDGE_MM/2+FIXED_CLEARANCE_MM), 0., FIXED_TIP_Z_MM-(CUBE_EDGE_MM/2-TIP_ABOVE_TABLE_MM)])
HOVER_MM = 60.
MAX_TILT_DEG = 12.                         # tool axis from vertical
MAX_YAW_ERR_DEG = 4.
LIMIT_MARGIN_RAW = 40
# Gripper raw counts (follower calibration closed stop 2044); moving inner face x ~ -3 - 0.107*(raw-2044) mm.
GRIPPER_OPEN_RAW = 2470                    # inner gap ~48 mm
GRIPPER_SQUEEZE_RAW = 2215                 # inner face at ~ -21 mm: 3-4 mm squeeze on a 25 mm cube
FLOORS_MM = dict(wrist_link=25., gripper_link=12.)


def tool(model, raw):
    return model.table_base@model.link_transform(model.joint_degrees(raw))


def grasp_point(model, raw):
    return (tool(model, raw)@np.r_[GRASP_IN_TOOL_MM/1000, 1])[:3]*1000


def _residual(model, base_raw, values, target_mm, yaw):
    raw = dict(base_raw, **dict(zip(ARM, values)))
    T = tool(model, raw); p = (T@np.r_[GRASP_IN_TOOL_MM/1000, 1])[:3]*1000
    z = T[:3, 2]; x = T[:3, 0]
    return np.r_[p-target_mm, (z-[0, 0, -1])*60, (x[:2]-[np.cos(yaw), np.sin(yaw)])*40]


def solve(model, seed_raw, target_mm, yaw_deg):
    """Best of the four symmetric jaw directions (cube is 90 deg symmetric). Returns dict or None."""
    cal = model.calibration
    lo = [cal[k]['range_min']+LIMIT_MARGIN_RAW for k in ARM]; hi = [cal[k]['range_max']-LIMIT_MARGIN_RAW for k in ARM]
    best = None
    # Roll seed per jaw direction: yaw changes ~linearly with roll, slope measured at the seed (sign depends on mounting).
    yaw_at = lambda raw: np.degrees(np.arctan2(tool(model, raw)[1, 0], tool(model, raw)[0, 0]))
    probe = dict(seed_raw, wrist_roll=seed_raw['wrist_roll']+50)
    slope = ((yaw_at(probe)-yaw_at(seed_raw)+180) % 360-180)/50
    for k in range(4):
        yaw = np.radians(yaw_deg+90*k)
        x0 = np.array([seed_raw[j] for j in ARM], float)
        if abs(slope) > 1e-3:
            x0[4] += ((np.degrees(yaw)-yaw_at(seed_raw)+180) % 360-180)/slope
        if not lo[4] <= x0[4] <= hi[4]:
            continue
        x0 = np.clip(x0, lo, hi)
        r = least_squares(lambda v: _residual(model, seed_raw, v, np.asarray(target_mm, float), yaw), x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=300)
        raw = dict(seed_raw, **{j: int(round(v)) for j, v in zip(ARM, r.x)})
        T = tool(model, raw)
        out = dict(raw=raw, yaw_deg=float(np.degrees(yaw)), position_error_mm=float(np.linalg.norm(grasp_point(model, raw)-target_mm)),
                   tilt_deg=float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))),
                   yaw_error_deg=float(abs((np.degrees(np.arctan2(T[1, 0], T[0, 0]))-np.degrees(yaw)+180) % 360-180)),
                   roll_travel_raw=abs(raw['wrist_roll']-seed_raw['wrist_roll']))
        out['ok'] = out['position_error_mm'] < 1. and out['tilt_deg'] <= MAX_TILT_DEG and out['yaw_error_deg'] <= MAX_YAW_ERR_DEG
        key = (not out['ok'], out['roll_travel_raw'] if out['ok'] else out['position_error_mm'])
        if best is None or key < best[0]:
            best = (key, out)
    return best[1]


def clearance(model, raw):
    """Lowest heights above the table (mm) for the checked links; None if fine, else a reason."""
    a = model.joint_degrees(raw)
    for link, floor in FLOORS_MM.items():
        h = (model.table_base@model.link_transform(a, link))[2, 3]*1000
        if h < floor:
            return f'{link} {h:.0f} mm above table (< {floor:.0f})'
    tip = (tool(model, raw)@np.r_[0, 0, FIXED_TIP_Z_MM/1000, 1])[2]*1000
    if tip < TIP_ABOVE_TABLE_MM-1:
        return f'jaw tip {tip:.1f} mm above table'
    return None


def plan(model, seed_raw, cube_xy_mm, cube_yaw_deg, steps=6):
    """Hover above the cube, then a straight vertical descent (waypoints) to the grasp pose."""
    centre = np.r_[cube_xy_mm, CUBE_EDGE_MM/2]
    hover = solve(model, seed_raw, centre+[0, 0, HOVER_MM], cube_yaw_deg)
    if not hover['ok']:
        return dict(ok=False, reason=f"hover pose unreachable (error {hover['position_error_mm']:.1f} mm, tilt {hover['tilt_deg']:.0f} deg)", hover=hover)
    path, previous = [], hover['raw']
    for h in np.linspace(HOVER_MM, 0, steps+1)[1:]:
        s = solve(model, previous, centre+[0, 0, h], hover['yaw_deg'])
        if not s['ok'] or abs(s['yaw_deg']-hover['yaw_deg']) > 1:
            return dict(ok=False, reason=f'no clean descent at {h:.0f} mm', hover=hover)
        path.append(s['raw']); previous = s['raw']
    for raw in [hover['raw']]+path:
        reason = clearance(model, raw)
        if reason:
            return dict(ok=False, reason='too close to table: '+reason, hover=hover)
    return dict(ok=True, hover=hover, descent=path, grasp=path[-1], cube_centre_mm=centre.tolist(), yaw_deg=hover['yaw_deg'])


def path_clear(model, start, goal, n=24):
    """Joint-interpolated move: links may not dip below their floor, unless both ends are already lower
    (then not below the lower end - 3 mm). Returns None or a reason."""
    a0, a1 = model.joint_degrees(start), model.joint_degrees(goal)
    h = lambda a, link: (model.table_base@model.link_transform(a, link))[2, 3]*1000
    ends = {link: min(h(a0, link), h(a1, link)) for link in FLOORS_MM}
    for s in np.linspace(0, 1, n+1)[1:-1]:
        a = model.joint_degrees({k: start[k]+(goal[k]-start[k])*s for k in start})
        for link, floor in FLOORS_MM.items():
            if h(a, link) < min(floor, ends[link]-3): return f'{link} would touch the table on the way'
    return None
