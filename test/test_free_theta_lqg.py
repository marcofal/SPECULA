import specula
specula.init(-1)  # CPU

import unittest

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib import adaptive_lqg_mimo as am
from specula.processing_objects.free_theta_lqg import FreeThetaLqg
from specula.processing_objects.integrator import Integrator


def ar2(T, seed, std=50.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 1000)
    d = np.zeros_like(e)
    for k in range(2, len(e)):
        d[k] = (0.995 + 0.9) * d[k - 1] - 0.995 * 0.9 * d[k - 2] + e[k]
    d = d[1000:]
    return std * d / d.std()


def loop(obj, d, M=None):
    """y_k = d_k - M out_comm_{k-1}: controller delay 1 + DM read one frame later."""
    T, n = d.shape
    M = np.eye(n) if M is None else M
    meas = BaseValue(value=np.zeros(n), target_device_idx=-1)
    obj.inputs['delta_comm'].set(meas)
    obj.setup()
    prev, comm = np.zeros(n), np.zeros((T, n))
    for k in range(T):
        meas.value = d[k] - M @ prev
        meas.generation_time = obj.seconds_to_t(k * 0.001)
        obj.check_ready(meas.generation_time)
        obj.trigger()
        obj.post_trigger()
        comm[k] = cpuArray(obj.outputs['out_comm'].value)
        prev = comm[k]
    return comm


class TestFreeThetaLqg(unittest.TestCase):

    def test_parameters(self):
        sp = SimulParams(time_step=0.001)
        obj = FreeThetaLqg(sp, n_modes=6, lqg_modes=[0, 1, 2], train_s=2.0, redesign_s=0.5,
                           dither_std=3.0, target_device_idx=-1)
        self.assertEqual(obj.method, 'mimo_free_theta')
        c = obj.mode_ctrl[0]
        self.assertIsInstance(c, am.AdaptiveMimoFreeTheta)
        self.assertEqual((c.m, c.N, c.structure), (3, 11, 'full'))
        self.assertEqual((c.min_samples, c.redesign_every, c.rls.window), (2000, 500, 2000))
        np.testing.assert_allclose(c.dither_after, 3.0)            # the dither stays on
        self.assertEqual(c.rho_grid, (0.0, 0.1, 0.3, 1.0, 10.0))
        self.assertTrue(c.refit_u)
        self.assertIsNone(c.Lam)                                   # time-shift register
        f = FreeThetaLqg(sp, n_modes=6, lqg_modes=[0, 1], poles=[0, 0, 0.5, 0.5], train_s=1.0,
                         target_device_idx=-1)
        self.assertEqual(f.mode_ctrl[0].N, 4)
        self.assertIsNotNone(f.mode_ctrl[0].Lam)
        with self.assertRaises(ValueError):
            FreeThetaLqg(sp, n_modes=6, structure='diag', target_device_idx=-1)
        with self.assertRaises(ValueError):
            FreeThetaLqg(sp, n_modes=6, train_s=2.0, window_s=1.0, target_device_idx=-1)

    def test_closes_the_loop_with_plant_cross_talk(self):
        """Two LQG modes on a plant whose commands are rotated by 30 deg: the full Theta
        holds the cross-talk, on the register and on stable filters."""
        sp = SimulParams(time_step=0.001)
        T = 6000
        d = np.column_stack([ar2(T, s) for s in (1, 2, 3)])
        a = np.deg2rad(30.0)
        M = np.eye(3)
        M[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = loop(ref, d, M)
        res_int = d[1:, :2] - (M @ comm_ref[:-1].T).T[:, :2]
        for poles in (None, [0, 0, 0.5, 0.5, 0.5, 0.5]):
            obj = FreeThetaLqg(sp, n_modes=3, lqg_modes=[0, 1], n_theta=6, poles=poles,
                               int_gain=0.4, train_s=1.5, window_s=2.5, redesign_s=0.25,
                               dither_std=2.0, target_device_idx=-1)
            comm = loop(obj, d, M)
            np.testing.assert_allclose(comm[:, 2], comm_ref[:, 2], rtol=1e-4, atol=1e-3)
            ctrl = obj.mode_ctrl[0]
            self.assertTrue([1 for _, e, _ in ctrl.log if e == 'accepted'], f"{poles}: {ctrl.log}")
            res = d[1:, :2] - (M @ comm[:-1].T).T[:, :2]
            self.assertLess(np.sqrt(np.mean(res[-2000:] ** 2)), np.sqrt(np.mean(res_int[-2000:] ** 2)),
                            str(poles))


if __name__ == "__main__":
    unittest.main()
