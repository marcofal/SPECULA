import specula
specula.init(-1)  # CPU

import unittest

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib import adaptive_lqg as al
from specula.lib import adaptive_lqg_var as avar
from specula.lib.dither_design import gain_rel_std, suggest_dither_std
from specula.processing_objects.dd_lqg import DdLqg


def ar2(T, seed, std=50.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 1000)
    d = np.zeros_like(e)
    for k in range(2, len(e)):
        d[k] = (0.995 + 0.9) * d[k - 1] - 0.995 * 0.9 * d[k - 2] + e[k]
    d = d[1000:]
    return std * d / d.std()


def loop(obj, d):
    """y_k = d_k - out_comm_{k-1}: controller delay 1 + DM read one frame later."""
    T, n = d.shape
    meas = BaseValue(value=np.zeros(n), target_device_idx=-1)
    obj.inputs['delta_comm'].set(meas)
    obj.setup()
    prev, comm = np.zeros(n), np.zeros((T, n))
    for k in range(T):
        meas.value = d[k] - prev
        meas.generation_time = obj.seconds_to_t(k * 0.001)
        obj.check_ready(meas.generation_time)
        obj.trigger()
        obj.post_trigger()
        comm[k] = cpuArray(obj.outputs['out_comm'].value)
        prev = comm[k]
    return comm


class TestDdLqg(unittest.TestCase):

    def test_parameters_map_to_the_two_methods(self):
        sp = SimulParams(time_step=0.001)
        mimo = DdLqg(sp, n_modes=6, method='mimo', lqg_modes=[0, 1, 2, 3], train_s=2.0,
                     redesign_s=0.5, block_size=2, dither_std=3.0, target_device_idx=-1)
        self.assertEqual(mimo.method, 'var_lqg')
        self.assertEqual([c.m for c in mimo.mode_ctrl], [2, 2])
        self.assertTrue(all(isinstance(c, avar.AdaptiveVarLQG) for c in mimo.mode_ctrl))
        c = mimo.mode_ctrl[0]
        self.assertEqual((c.min_samples, c.redesign_every), (2000, 500))
        np.testing.assert_allclose(c.dither_after, 3.0)            # mimo keeps the dither on
        siso = DdLqg(sp, n_modes=6, method='siso', lqg_modes=[0, 1], train_s=2.0,
                     redesign_s=0.25, n_theta=9, target_device_idx=-1)
        self.assertEqual(siso.method, 'free_theta')
        self.assertTrue(all(isinstance(c, al.AdaptiveModeFreeTheta) for c in siso.mode_ctrl))
        self.assertEqual(siso.mode_ctrl[0].N, 9)
        self.assertEqual(siso.mode_ctrl[0].dither_after, 0.0)      # siso switches it off
        with self.assertRaises(ValueError):
            DdLqg(sp, n_modes=6, method='lqg', target_device_idx=-1)

    def test_both_methods_close_the_loop(self):
        sp = SimulParams(time_step=0.001)
        T = 5000
        d = np.column_stack([ar2(T, s) for s in (1, 2, 3)])
        for method in ('siso', 'mimo'):
            obj = DdLqg(sp, n_modes=3, method=method, lqg_modes=[0, 1], int_gain=0.4,
                        train_s=1.5, window_s=2.5 if method == 'mimo' else None,
                        redesign_s=0.25, dither_std=5.0, p=4, target_device_idx=-1)
            comm = loop(obj, d)
            for ctrl in obj.mode_ctrl:
                acc = [e for _, e, _ in ctrl.log if e == 'accepted']
                self.assertTrue(acc, f"{method}: no design accepted in {ctrl.name}")
            res = d[1:, :2] - comm[:-1, :2]
            self.assertLess(np.sqrt(np.mean(res[-1000:] ** 2)), np.sqrt(np.mean(d[-1000:, :2] ** 2)),
                            method)


class TestDitherDesign(unittest.TestCase):

    def test_suggestion_meets_the_target_on_its_own_model(self):
        """On a pure delay plant with AR(1) turbulence, the suggested dither inverts
        gain_rel_std (margin 1) and scales as 1 / sqrt(window)."""
        rng = np.random.default_rng(0)
        T, g = 20000, 0.7
        e = rng.standard_normal((T, 2)) * [3.0, 6.0]
        phi = np.zeros((T, 2))
        for k in range(1, T):
            phi[k] = 0.95 * phi[k - 1] + e[k]
        u = rng.standard_normal((T, 2))
        y = phi.copy()
        y[2:] -= g * u[:-2]
        d, se = suggest_dither_std(y, u, window=5000, gain_guess=g, rel_std=0.1, margin=1.0,
                                   d_min=0.0)
        np.testing.assert_allclose(se, [3.0, 6.0], rtol=0.05)
        np.testing.assert_allclose(gain_rel_std(se, d, 5000, g), 0.1, rtol=1e-9)
        d4, _ = suggest_dither_std(y, u, window=20000, gain_guess=g, rel_std=0.1, margin=1.0,
                                   d_min=0.0)
        np.testing.assert_allclose(d4, d / 2, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
