"""URDF forward kinematics only; does not command hardware."""
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def rotation_rpy(rpy):
    roll, pitch, yaw = rpy
    rx, _ = cv2.Rodrigues(np.array([roll, 0.0, 0.0]))
    ry, _ = cv2.Rodrigues(np.array([0.0, pitch, 0.0]))
    rz, _ = cv2.Rodrigues(np.array([0.0, 0.0, yaw]))
    return rz @ ry @ rx


@lru_cache(maxsize=None)
def joint_chain(target="gripper_frame_link"):
    tree = ET.parse(ROOT / "models/so101/so101_new_calib.urdf").getroot()
    by_child = {node.find("child").get("link"): node for node in tree.findall("joint")}
    chain = []
    child = target
    while child != "base_link":
        node = by_child[child]
        chain.append(node)
        child = node.find("parent").get("link")
    parsed = []
    for node in reversed(chain):
        origin = node.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        origin_transform = np.eye(4)
        origin_transform[:3, :3] = rotation_rpy(rpy)
        origin_transform[:3, 3] = xyz
        if node.get("type") != "fixed":
            if node.get("type") != "revolute":
                raise ValueError("Unsupported joint type")
            axis = np.fromstring(node.find("axis").get("xyz"), sep=" ")
            axis /= np.linalg.norm(axis)
        else:
            axis = None
        parsed.append((node.get("name"), origin_transform, axis))
    return parsed


def forward_kinematics(joint_degrees, target="gripper_frame_link", *, joint_origin_overrides=None):
    """Compute a link pose; optional origin translations support offline geometry checks."""
    transform = np.eye(4)
    for name, origin, axis in joint_chain(target):
        local = origin.copy()
        if joint_origin_overrides and name in joint_origin_overrides:
            translation = np.asarray(joint_origin_overrides[name], dtype=float)
            if translation.shape != (3,) or not np.all(np.isfinite(translation)):
                raise ValueError("Joint origin must be three finite coordinates")
            local[:3, 3] = translation
        if axis is not None:
            rotation, _ = cv2.Rodrigues(axis * np.deg2rad(joint_degrees[name]))
            local[:3, :3] = local[:3, :3] @ rotation
        transform = transform @ local
    return transform


def degrees_from_raw(raw, calibration):
    # Matches installed LeRobot's MotorNormMode.DEGREES conversion.
    return {name: (raw[name] - (calibration[name]["range_min"] + calibration[name]["range_max"]) / 2) * 360 / 4095
            for name in JOINTS}
