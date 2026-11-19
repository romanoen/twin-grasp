"""Arm and tag I/O for the SO-101 follower.

Owns the follower bus (stop live_robot_view.py first). Motor writes happen only in JointBus.hold()/release()
and move_to(): the five arm joints; the gripper is never commanded. Torque is never disabled on exit,
so a held arm cannot drop. `python arm_io.py --release` switches arm torque off (hold the arm first).
"""
import argparse, os, time
from pathlib import Path
import cv2
import numpy as np
from robot_model import load_robot_model
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROOT = Path(__file__).resolve().parents[1]
PORT = os.environ.get('SO101_PORT', '/dev/cu.usbmodem5B901033421')
ARM_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
STILL_DEG = .3
RAW_PER_DEG = 4096/360
STEP_DEG = .25              # per 25 ms command step -> about 10 deg/s
FOLLOW_LIMIT_RAW = 60       # abort if any joint lags the command by more than this
SETTLE_S = 1.5


class JointBus:
    """Direct joint access. Writes happen only in hold()/release(), triggered by the user."""
    def __init__(self, model):
        self.model = model
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id='follower', calibration_dir=ROOT/'calibration', disable_torque_on_disconnect=False))
        self.robot = robot  # keep alive: dropping it disconnects the bus
        self.bus = robot.bus; self.bus.connect(); self.held = False
        if not self.bus.is_calibrated:
            raise RuntimeError('Motor calibration mismatch')
        self.torque_at_connection = {k: int(v) for k, v in self.bus.sync_read('Torque_Enable', normalize=False, num_retry=1).items()}
        modes = self.bus.sync_read('Operating_Mode', normalize=False, num_retry=1)
        self.position_mode = all(int(modes[k]) == 0 for k in ARM_JOINTS)
        self.held = all(self.torque_at_connection[k] for k in ARM_JOINTS)

    def read(self):
        raw = {k: int(v) for k, v in self.bus.sync_read('Present_Position', normalize=False, num_retry=1).items()}
        return dict(timestamp_ms=time.time()*1000, raw_positions=raw, joint_degrees=self.model.joint_degrees(raw), model_id=self.model.model_id)

    def hold(self):
        if not self.position_mode:
            raise RuntimeError('Arm joints not in position mode; refusing to hold')
        raw = self.read()['raw_positions']
        goal = {k: raw[k] for k in ARM_JOINTS}
        self.bus.sync_write('Goal_Position', goal, normalize=False)
        self.bus.enable_torque(list(ARM_JOINTS), num_retry=1)
        self.bus.sync_write('Goal_Position', goal, normalize=False)
        self.held = True
        after = self.read()['raw_positions']
        return max(abs(after[k]-goal[k]) for k in ARM_JOINTS)

    def release(self):
        self.bus.disable_torque(list(ARM_JOINTS), num_retry=1); self.held = False

    def close(self):
        # Never drop a held arm on exit: holding is left active on purpose.
        self.bus.disconnect(disable_torque=False)


def tag_pose_candidates(corners, edge, matrix, distortion):
    """Both IPPE solutions for a square tag, best reprojection first."""
    half = edge / 2
    points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    _, rotations, translations, _ = cv2.solvePnPGeneric(points, pixels, matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    for rotation, translation in zip(rotations, translations):
        rmat = cv2.Rodrigues(rotation)[0]
        if np.min((rmat @ points.T + translation)[2]) <= 0:
            continue
        projected = cv2.projectPoints(points, rotation, translation, matrix, distortion)[0].reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
        transform = np.eye(4)
        transform[:3, :3] = rmat
        transform[:3, 3] = translation.ravel()
        candidates.append({"reprojection_rms_px": error, "T_camera_tag": transform.tolist(),
                           "camera_position_tag_m": (-rmat.T @ translation).ravel().tolist(),
                           "tag_distance_m": float(np.linalg.norm(translation))})
    return sorted(candidates, key=lambda item: item["reprojection_rms_px"])


def detect(finder, frame, settings):
    corners, ids, _ = finder.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    matches = [] if ids is None else np.flatnonzero(ids.flatten() == 1)
    if len(matches) != 1:
        return None, corners, ids
    c = corners[int(matches[0])]
    candidates = tag_pose_candidates(c, .08, np.array(settings['camera_matrix']), np.array(settings['distortion_coefficients']))
    if not candidates or candidates[0]['reprojection_rms_px'] > 1:
        return None, corners, ids
    if len(candidates) > 1 and candidates[1]['reprojection_rms_px'] < 1:
        return None, corners, ids   # ambiguous planar pose
    return (c.reshape(4, 2).tolist(), candidates), corners, ids


class Abort(Exception):
    pass


def move_to(arm, goal, pump):
    """Linear joint interpolation. pump() refreshes the window and returns True if a key was pressed."""
    start = arm.read()['raw_positions']
    span = max(abs(goal[k]-start[k]) for k in ARM_JOINTS)
    steps = max(1, int(np.ceil(span/(STEP_DEG*RAW_PER_DEG))))
    for step in range(1, steps+1):
        command = {k: round(start[k]+(goal[k]-start[k])*step/steps) for k in ARM_JOINTS}
        arm.bus.sync_write('Goal_Position', command, normalize=False)
        time.sleep(.025)
        present = arm.read()['raw_positions']
        lag = max(abs(present[k]-command[k]) for k in ARM_JOINTS)
        if pump() or lag > FOLLOW_LIMIT_RAW:
            arm.bus.sync_write('Goal_Position', {k: present[k] for k in ARM_JOINTS}, normalize=False)
            raise Abort('Stopped' if lag <= FOLLOW_LIMIT_RAW else f'Following error {lag} steps - arm halted')


def release_only():
    arm = JointBus(load_robot_model())
    try:
        input('Hold the arm, then press Enter to switch torque off ...'); arm.release(); print('Arm torque off.')
    finally:
        arm.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true', help='Switch arm torque off (hold the arm first)')
    if parser.parse_args().release:
        release_only()
