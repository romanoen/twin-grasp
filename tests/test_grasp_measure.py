"""Offline tests for grasp_measure (synthetic push on a real pre-close image; bias solver)."""
import json, unittest
from pathlib import Path
import cv2
import numpy as np
from grasp_measure import FIXED_CLEARANCE_MM, cube_red, measure, push_geometry, solve_bias
from robot_model import load_robot_model

ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT/'tests/fixtures/grasp_20261109_101521'


def pushed(img, feat, shift):
    """Move the red cube by `shift` pixels (inpaint the old place)."""
    mask = (cube_red(img) > 0).astype(np.uint8)
    out = cv2.inpaint(img, cv2.dilate(mask, np.ones((7, 7), np.uint8)), 5, cv2.INPAINT_TELEA)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]); h, w = mask.shape
    moved = cv2.warpAffine(img, M, (w, h)); mm = cv2.warpAffine(mask, M, (w, h))
    out[mm > 0] = moved[mm > 0]; return out


class MeasureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_model(); cls.feat = json.loads((ROOT/'config/cube_features.json').read_text())
        intr = json.loads((ROOT/'calibration/camera_intrinsics.json').read_text()); cls.K, cls.d = np.array(intr['camera_matrix']), np.array(intr['distortion_coefficients'])
        cls.raw = next(e for e in json.loads((RUN/'log.json').read_text()) if e['event'] == 'descend')['raw']
        cls.img = cv2.imread(str(RUN/'descend.jpg'))

    def test_synthetic_push_is_measured(self):
        g = push_geometry(self.model, self.raw, self.K, self.d)
        for mm in (3., 7.5, 11.):
            after = pushed(self.img, self.feat, g['axis']*mm*g['px_per_mm'])
            r = measure(self.model, self.raw, self.img, after, self.feat, self.K, self.d)
            self.assertTrue(r['valid'], r); self.assertAlmostEqual(r['clearance_mm'], mm, delta=.4)

    def test_push_direction_points_to_fixed_jaw(self):
        # In the pre-close image the cube sits left, near the fixed jaw: the push must point left (towards it).
        self.assertLess(push_geometry(self.model, self.raw, self.K, self.d)['axis'][0], -.5)

    def test_bias_solver_recovers_constant_offset(self):
        b = np.array([2., -3.]); rng = np.random.default_rng(1); samples = []
        for a in rng.uniform(0, 2*np.pi, 8):
            x = np.array([np.cos(a), np.sin(a)]); samples.append(dict(valid=True, x_table=x.tolist(), clearance_mm=FIXED_CLEARANCE_MM+x@b+rng.normal(0, .2)))
        r = solve_bias(samples); self.assertTrue(r['ready']); np.testing.assert_allclose(r['bias_mm'], b, atol=.4)

    def test_needs_varied_directions(self):
        same = [dict(valid=True, x_table=[1., 0.], clearance_mm=6.) for _ in range(8)]
        self.assertFalse(solve_bias(same)['ready'])


if __name__ == '__main__':
    unittest.main()
