"""Read-only joint stream for local viewer. Never enables/disables torque or writes goals.

LiveWriter is shared with grasp_app.py, which owns the bus while it runs and publishes the same packet.
"""
import json
import time
from pathlib import Path
import numpy as np
from robot_model import load_robot_model
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'ui/data'


class LiveWriter:
    def __init__(self, model):
        self.snapshot = json.loads((DATA/'robot.json').read_text())
        if self.snapshot.get('model_id') != model.model_id:
            raise RuntimeError('Snapshot model differs; regenerate snapshot first')
        self.model = model
        # Fixed registration from the active calibration (snapshot): valid while base/tag setup is unchanged.
        self.base = np.array(self.snapshot['T_table_base'])
        self.local = [np.array(v['T_link_visual']) for v in self.snapshot['visuals']]

    def write(self, raw, torque, **extra):
        model, base = self.model, self.base
        angles, tool, camera = model.frames(raw, base)
        matrices = [(base@model.link_transform(angles, v['link'])@local).tolist() for v, local in zip(self.snapshot['visuals'], self.local)]
        payload = dict(timestamp_ms=time.time()*1000, model_id=model.model_id, model_roll_zero_deg=model.roll_offset_deg, T_table_tool=tool.tolist(),
                       T_table_camera=camera.tolist(), tool_reference_physically_validated=False, raw_positions={k: int(v) for k, v in raw.items()},
                       torque_at_connection={k: int(v) for k, v in torque.items()}, joint_degrees=angles, matrices=matrices,
                       registration='saved_scene_unverified', read_only=True, urdf_sha256=self.snapshot['urdf_sha256'], **extra)
        temporary = DATA/'live-robot.tmp'
        temporary.write_text(json.dumps(payload, allow_nan=False))
        temporary.replace(DATA/'live-robot.json')


def main():
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from arm_io import PORT
    model = load_robot_model(); writer = LiveWriter(model)
    calibration_path = ROOT/'calibration/follower.json'
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id='follower', calibration_dir=calibration_path.parent))
    bus = robot.bus
    bus.connect()
    try:
        if not bus.is_calibrated:
            raise RuntimeError('Motor calibration mismatch')
        torque = bus.sync_read('Torque_Enable', normalize=False, num_retry=1)
        print('Read-only live joints connected; no torque or goal writes.', flush=True)
        while True:
            started = time.monotonic()
            writer.write(bus.sync_read('Present_Position', normalize=False, num_retry=1), torque)
            time.sleep(max(0, .05-(time.monotonic()-started)))
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: pass
