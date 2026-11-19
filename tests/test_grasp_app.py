"""Offline tests of the model-only cube bookkeeping in grasp_app (no hardware)."""
import unittest
import numpy as np
import grasp_app as G
from grasp_plan import plan
from robot_model import load_robot_model
from grasp_area import load_or_compute


def station(offset=(0., 0.)):
    s = G.Station.__new__(G.Station); s.model = load_robot_model(); s.area = load_or_compute(s.model)
    s.views = s.area['views_raw']; s.home = s.views[0]; s.offset = np.array(offset)
    return s


class BookkeepingTest(unittest.TestCase):
    def test_held_cube_after_grasp_sits_at_fixed_jaw(self):
        s = station(); c = np.array([40., 200.])
        p = plan(s.model, s.home, c, 20.); self.assertTrue(p['ok'])
        q, z = s.held_cube_pose(p['grasp'])
        T = s.tool(p['grasp']); x_axis = T[:2, 0]/np.linalg.norm(T[:2, 0])
        # planned centre 7.5 mm from the fixed jaw; after the push the cube is 7.5 mm further along +x_tool
        np.testing.assert_allclose(np.array(q[:2])-c, 7.5*x_axis, atol=.6)
        self.assertAlmostEqual(z, 12.5, delta=.6)

    def test_place_plan_puts_cube_on_target(self):
        for offset, yaw in (((0., 0.), 30.), ((-5.86, 1.31), 30.), ((0., 0.), 0.), ((0., 0.), 62.)):
            s = station(offset); target = np.array([20., 230.]); want = target-s.offset; goal = want.copy(); p = None
            for _ in range(3):   # same iteration as do_place
                p = plan(s.model, s.home, goal, yaw if p is None else p['yaw_deg']); self.assertTrue(p['ok'])
                err = (s.tool(p['grasp'])@np.r_[G.CUBE_IN_TOOL_MM/1000, 1])[:2]*1000-want
                if np.linalg.norm(err) < .5: break
                goal = goal-err
            q, _ = s.held_cube_pose(p['grasp'])
            np.testing.assert_allclose(q[:2], target, atol=1.)   # model says: cube lands on the target (true frame)
            self.assertLess(abs((q[2]-yaw+45) % 90-45), 4.)      # and with the target rotation (cube is 90 deg symmetric)


if __name__ == '__main__':
    unittest.main()
