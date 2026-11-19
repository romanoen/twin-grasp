"""Model-only jaw conversion; never changes motor calibration."""
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def gripper_limits_deg():
    urdf = Path(__file__).resolve().parents[1] / 'models/so101/so101_new_calib.urdf'
    joint = ET.parse(urdf).getroot().find("joint[@name='gripper']")
    limit = joint.find('limit')
    return math.degrees(float(limit.get('lower'))), math.degrees(float(limit.get('upper')))


def gripper_model_angle(raw, calibration):
    """Align the measured closed stop with the URDF closed limit, preserving servo degrees."""
    lower, upper = gripper_limits_deg()
    closed = calibration['gripper']['range_min']
    angle = lower + (raw['gripper'] - closed) * 360 / 4095
    return min(upper, max(lower, angle))


# All active viewer, tool and camera transforms use this shared model.
import hashlib
import json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
from so101_kinematics import degrees_from_raw, forward_kinematics, joint_chain, rotation_rpy


# Foremost point of the fixed jaw (wrist_roll_follower mesh) in gripper_frame_link coordinates, metres.
# Derived from the URDF visual mesh; the physical tip is the reference for touch tests.
FIXED_JAW_TIP_IN_TOOL_M = (0.0010, 0.0000, 0.0063)


class RobotModel:
    def __init__(self, calibration, handeye, config, model_id, registration=None):
        self.calibration = calibration
        # Table<-base registration from the active calibration (None: derive from a scene capture).
        self.table_base = np.array(registration['T_table_base']) if registration else None
        self.handeye = handeye
        self.config = config
        self.model_id = model_id
        self.roll_offset_deg = float(config['wrist_roll_zero_offset_deg'])
        if not math.isfinite(self.roll_offset_deg) or not -180 <= self.roll_offset_deg <= 180:
            raise ValueError('Invalid model roll offset')
        # G_new = G_old F^-1 Rz(delta) F, where F is the fixed tool origin.
        # Apply the inverse frame change to the camera mount, preserving G X exactly.
        fixed_tool = joint_chain('gripper_frame_link')[-1][1]
        self.tool_from_gripper = fixed_tool.copy()
        unrotate = np.eye(4)
        unrotate[:3, :3] = rotation_rpy([0., 0., -math.radians(self.roll_offset_deg)])
        self.camera_transform = np.linalg.inv(fixed_tool) @ unrotate @ fixed_tool @ np.array(handeye['T_gripper_camera'])

    def joint_degrees(self, raw):
        angles = degrees_from_raw(raw, self.calibration)
        for name, offset in self.handeye['joint_zero_offsets_deg'].items():
            angles[name] += offset
        angles['wrist_roll'] += self.roll_offset_deg
        angles['gripper'] = gripper_model_angle(raw, self.calibration)
        return angles

    def link_transform(self, angles, target='gripper_frame_link'):
        if target == 'base_link':
            return np.eye(4)
        return forward_kinematics(angles, target)

    def frames(self, raw, table_base):
        angles = self.joint_degrees(raw)
        tool = np.array(table_base) @ self.link_transform(angles)
        return angles, tool, tool @ self.camera_transform


def load_robot_model(calibration_path=None, handeye_path=None, config_path=None):
    config_path = Path(config_path or ROOT/'config/robot_model.json')
    contents = config_path.read_bytes()
    config = json.loads(contents)
    paths = {'calibration':Path(calibration_path or ROOT/config['references']['calibration']['path']),
             'handeye':Path(handeye_path or ROOT/config['references']['handeye']['path']),
             'urdf':ROOT/config['references']['urdf']['path']}
    if 'registration' in config['references']:
        paths['registration'] = ROOT/config['references']['registration']['path']
    blobs = {}
    for name, path in paths.items():
        blobs[name] = path.read_bytes()
        if hashlib.sha256(blobs[name]).hexdigest() != config['references'][name]['sha256']:
            raise ValueError('Model reference changed: '+name)
    identity = hashlib.sha256(contents).hexdigest()
    registration = json.loads(blobs['registration']) if 'registration' in blobs else None
    return RobotModel(json.loads(blobs['calibration']),json.loads(blobs['handeye']),config,identity,registration)
