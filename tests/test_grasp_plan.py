"""Offline tests of the grasp planner (no hardware)."""
import json, unittest
from pathlib import Path
import numpy as np
from grasp_plan import CUBE_EDGE_MM, HOVER_MM, clearance, grasp_point, plan, tool
from robot_model import load_robot_model

SEED = json.loads((Path(__file__).resolve().parents[1]/'data/multipose_confirm_20261108/plan.json').read_text())['poses'][0]['target_raw']


class GraspPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_robot_model(); cls.seed = dict(SEED, gripper=2048)

    def test_reachable_cube(self):
        p = plan(self.model, self.seed, (80, 200), 20.)
        self.assertTrue(p['ok'], p.get('reason'))
        centre = np.r_[80, 200, CUBE_EDGE_MM/2]
        np.testing.assert_allclose(grasp_point(self.model, p['grasp']), centre, atol=1.)
        np.testing.assert_allclose(grasp_point(self.model, p['hover']['raw']), centre+[0, 0, HOVER_MM], atol=1.)
        self.assertLess(np.degrees(np.arccos(-tool(self.model, p['grasp'])[2, 2])), 12)
        yaw = np.degrees(np.arctan2(*tool(self.model, p['grasp'])[[1, 0], 0]))
        self.assertLess(abs((yaw-20+45) % 90-45), 4)          # jaws square to a cube face
        for raw in p['descent']:
            self.assertIsNone(clearance(self.model, raw))

    def test_out_of_reach_is_refused(self):
        self.assertFalse(plan(self.model, self.seed, (145, 11), 7.)['ok'])


if __name__ == '__main__':
    unittest.main()
