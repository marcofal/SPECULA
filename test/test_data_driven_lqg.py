import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib import adaptive_lqg as al
from specula.processing_objects.data_driven_lqg import DataDrivenLqg
from specula.processing_objects.integrator import Integrator


def ar2_turbulence(T, seed=1, std=100.0):
    """Low-pass AR(2) sequence with poles 0.995 and 0.9, scaled to `std`."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 1000)
    d = np.zeros_like(e)
    a1, a2 = 0.995 + 0.9, -0.995 * 0.9
    for k in range(2, len(e)):
        d[k] = a1 * d[k - 1] + a2 * d[k - 2] + e[k]
    d = d[1000:]
    return std * d / d.std()


def fractional_taps(delay, gain):
    """Taps (g_1, g_2, g_3) of -gain ((1 - f) z^-m + f z^-(m+1))."""
    m, f = int(np.floor(delay)), delay - np.floor(delay)
    g = np.zeros(3)
    g[m - 1], g[m] = -gain * (1 - f), -gain * f
    return g


def run_fir_loop(step, d, taps):
    """y_k = d_k + sum_i g_i u_{k-i}, u_k = step(y_k). Returns the residual."""
    u = np.zeros(len(d))
    y = np.zeros(len(d))
    for k in range(len(d)):
        past = u[max(0, k - len(taps)):k][::-1]
        y[k] = d[k] + taps[:len(past)] @ past
        u[k] = step(y[k])
    return y


class TestAdaptiveLqgLib(unittest.TestCase):

    def test_sliding_window_iv_matches_batch(self):
        rng = np.random.default_rng(0)
        n, W, T = 3, 200, 700
        Phi, Z = rng.standard_normal((T, n)), rng.standard_normal((T, n))
        Y = Phi @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.standard_normal(T)
        rls = al.SlidingWindowRLS(n, W, delta=1e-9, refresh_every=10**9)
        for k in range(T):
            rls.update(Phi[k], Y[k], z=Z[k])
        batch = np.linalg.solve(Z[-W:].T @ Phi[-W:], Z[-W:].T @ Y[-W:])
        np.testing.assert_allclose(rls.theta, batch, rtol=1e-7, atol=1e-9)

    def test_lqg_nominal_plant_beats_open_loop(self):
        a = np.array([0.995 + 0.9, -0.995 * 0.9])
        d = al.tune_lqg(np.array([0.0, -1.0, 0.0]), a, q=1.0, r=0.0, dt=1e-3,
                        delay_margin=0.5)
        # open-loop std of the AR(2) from its impulse response
        h = np.zeros(20000)
        h[0], h[1] = 1.0, a[0]
        for k in range(2, len(h)):
            h[k] = a[0] * h[k - 1] + a[1] * h[k - 2]
        ol_std = np.sqrt(np.sum(h ** 2))
        # 2-frame minimum variance is sqrt(1 + a_1^2) = 2.14; margins cost some of it
        pred = d.predicted_output_std()
        self.assertGreater(pred, np.sqrt(1 + a[0] ** 2) - 1e-6)
        self.assertLess(pred, 0.1 * ol_std)
        self.assertTrue(np.all(np.abs(np.linalg.eigvals(
            al.closed_loop_matrix(al.fir_plant([0.0, -1.0, 0.0]), d.ctrl))) < 1))

    def test_adaptive_mode_identifies_plant_and_beats_integrator(self):
        T = 7000
        d = ar2_turbulence(T)
        taps = fractional_taps(2.3, 0.6)                 # true plant, unknown to the controller

        ctrl = al.AdaptiveModeLQG(dt=1e-3, plant_window=3000, dist_window=2000,
                                  dither_std=2.0, warmup_gain=0.4,
                                  rng=np.random.default_rng(3))
        y_lqg = run_fir_loop(lambda y: ctrl.step(y)[0], d, taps)

        state = [0.0]

        def integrator(y):
            state[0] += 0.4 * y
            return state[0]
        y_int = run_fir_loop(integrator, d, taps)

        accepted = [info for _, ev, info in ctrl.log if ev == "accepted"]
        self.assertTrue(accepted, f"no design accepted: {ctrl.log}")
        g = np.array(accepted[-1]["g"])
        self.assertAlmostEqual(g.sum(), taps.sum(), delta=0.15 * abs(taps.sum()))
        rms_lqg = np.sqrt(np.mean(y_lqg[-2000:] ** 2))
        rms_int = np.sqrt(np.mean(y_int[-2000:] ** 2))
        self.assertLess(rms_lqg, rms_int)


class TestDataDrivenLqgObject(unittest.TestCase):

    def _loop(self, obj, d, t_step):
        """Drive the object in a SPECULA-like loop: controller delay 1 inside
        the object, DM layer read one frame later (dm.out_layer:-1), so
        y_k = d_k - out_comm_{k-1}."""
        T, n = d.shape
        meas = BaseValue(value=np.zeros(n), target_device_idx=-1)
        obj.inputs['delta_comm'].set(meas)
        obj.setup()
        prev_comm = np.zeros(n)
        comm = np.zeros((T, n))
        for k in range(T):
            t = k * t_step
            meas.value = d[k] - prev_comm
            meas.generation_time = t
            obj.check_ready(t)
            obj.trigger()
            obj.post_trigger()
            comm[k] = cpuArray(obj.outputs['out_comm'].value)
            prev_comm = comm[k]
        return comm

    def test_object_integrator_modes_and_plant_timing(self):
        simul_params = SimulParams(time_step=0.001)
        T = 5000
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2, 3)])

        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[1], int_gain=0.4,
                            plant_window=2500, dist_window=1500, dither_std=2.0,
                            delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001))

        # integrator modes behave like the SPECULA Integrator in the same loop
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))
        np.testing.assert_allclose(comm[:, [0, 2]], comm_ref[:, [0, 2]], rtol=1e-4, atol=1e-3)

        # the LQG mode sees u_k -> y as -z^-2: tap 2 is about -1
        ctrl = obj.mode_ctrl[0]
        accepted = [info for _, ev, info in ctrl.log if ev == "accepted"]
        self.assertTrue(accepted, f"no design accepted: {ctrl.log}")
        np.testing.assert_allclose(accepted[-1]["g"], [0.0, -1.0, 0.0], atol=0.15)


if __name__ == '__main__':
    unittest.main()
