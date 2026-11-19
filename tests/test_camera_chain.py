"""Camera-chain calibration: separate from the tool chain, falls back to the active model, reproduces its check value."""
import json, unittest
import numpy as np
import camera_chain as C
from robot_model import load_robot_model


class CameraChainTest(unittest.TestCase):
    def setUp(self):
        self.model = load_robot_model(); self.raw = {'shoulder_pan': 2338, 'shoulder_lift': 1198, 'elbow_flex': 2047, 'wrist_flex': 3126, 'wrist_roll': 1514, 'gripper': 2464}

    def test_fallback_is_the_active_model(self):
        c = C.CameraChain(self.model); c.data = None
        m = self.model; ref = np.linalg.inv(m.table_base@m.link_transform(m.joint_degrees(self.raw))@m.camera_transform)
        np.testing.assert_allclose(c.camera_table(self.raw), ref, atol=1e-12)

    def test_stored_calibration_beats_active_model_on_check_poses(self):
        d = json.loads(C.OUT.read_text()); self.assertEqual(d['model_id'], self.model.model_id)
        self.assertLess(d['report']['after']['check_poses']['rms_mm'], .6*d['report']['before']['check_poses']['rms_mm'])
        c = C.CameraChain(self.model); views = C.session('multipose_confirm_20261108')
        e = []
        for v in views:   # camera_table applied to the check poses: implied tag pose vs. the table origin (tag = table frame)
            T = np.linalg.inv(c.camera_table(v['raw']))@v['obs']; e.append(np.linalg.norm(T[:3, 3])*1000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(np.square(e)))), d['report']['after']['check_poses']['rms_mm'], delta=.05)


if __name__ == '__main__':
    unittest.main()
