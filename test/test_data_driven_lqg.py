import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib import adaptive_lqg as al
from specula.lib import adaptive_lqg_mimo as am
from specula.lib import adaptive_lqg_var as avar
from specula.processing_objects.data_driven_lqg import DataDrivenLqg
from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.iir_filter import IirFilter
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


    def test_model_lqg_on_free_theta_equals_fir_ar_lqg(self):
        """The free-theta design on the theta of a FIR + AR model is the structured
        Kalman + LQR (same controller transfer function)."""
        g, a, q = np.array([0.0, -0.7, -0.3]), np.array([1.2, -0.25]), 4.0
        theta = al.theta_from_fir_ar(g, a)
        f = np.linspace(1.0, 450.0, 60)
        for s, rho in ((0.0, 1.0), (1.0, 1.0), (100.0, 10.0)):
            d_free = al.ModelLQG(*al.buffer_model(theta), q, rho, s)
            d_fir = al.FirArLQG(g, a, q, r=0.0, rho=rho, r_kalman=max(s * q, 1e-9))
            c_free = al.frequency_response(*d_free.ctrl, f, 1e-3)
            c_fir = al.frequency_response(*d_fir.ctrl, f, 1e-3)
            np.testing.assert_allclose(c_free, c_fir, rtol=1e-6, atol=1e-8)

    def test_stabilize_theta(self):
        theta = al.theta_from_fir_ar([0.0, -1.0, 0.0], [2.1, -1.2])      # A roots 1.2 +- ... (unstable)
        self.assertGreater(np.abs(np.roots(np.r_[1.0, -theta[:5]])).max(), 1.0)
        th_s, moved = al.stabilize_theta(theta, 0.99)
        self.assertEqual(moved, 2)
        self.assertLessEqual(np.abs(np.roots(np.r_[1.0, -th_s[:5]])).max(), 0.99 + 1e-9)
        np.testing.assert_array_equal(th_s[5:], theta[5:])
        th_ok, moved_ok = al.stabilize_theta(al.theta_from_fir_ar([0.0, -1.0, 0.0], [1.2, -0.25]))
        self.assertEqual(moved_ok, 0)

    def test_free_theta_mode_beats_integrator_without_dither(self):
        T = 9000
        d = ar2_turbulence(T)
        taps = fractional_taps(2.3, 0.6)                 # true plant, unknown to the controller

        ctrl = al.AdaptiveModeFreeTheta(dt=1e-3, N=6, window=3000, min_samples=2000,
                                        dither_std=2.0, dither_after=0.0, warmup_gain=0.4,
                                        rng=np.random.default_rng(3))
        y_free = run_fir_loop(lambda y: ctrl.step(y)[0], d, taps)

        state = [0.0]

        def integrator(y):
            state[0] += 0.4 * y
            return state[0]
        y_int = run_fir_loop(integrator, d, taps)

        events = [ev for _, ev, _ in ctrl.log]
        self.assertIn("accepted", events, f"no design accepted: {ctrl.log}")
        self.assertNotIn("reverted", events)
        self.assertEqual(ctrl.dither_level, 0.0)
        rms_free = np.sqrt(np.mean(y_free[-2000:] ** 2))
        rms_int = np.sqrt(np.mean(y_int[-2000:] ** 2))
        self.assertLess(rms_free, rms_int)


class TestAdaptiveLqgMimoLib(unittest.TestCase):

    def test_m1_equals_siso(self):
        """With one mode the MIMO model, design and stabilization are the SISO ones."""
        theta = al.theta_from_fir_ar([0.0, -1.0, -0.1], [1.4, -0.6, 0.1])
        N = theta.size // 2
        f = np.linspace(1.0, 450.0, 40)
        for M1, Mm in zip(al.buffer_model(theta), am.buffer_model(theta[:, None], N, 1)):
            np.testing.assert_array_equal(M1, Mm)
        for s, rho in ((0.0, 1.0), (10.0, 0.0), (100.0, 10.0)):
            d1 = al.ModelLQG(*al.buffer_model(theta), 3.0, rho, s)
            dm = am.MimoModelLQG(*am.buffer_model(theta[:, None], N, 1), [[3.0]], rho, s)
            np.testing.assert_allclose(al.frequency_response(*d1.ctrl, f, 1e-3),
                                       am.frequency_response(*dm.ctrl, f, 1e-3)[:, 0, 0], rtol=1e-8, atol=1e-10)
            self.assertAlmostEqual(d1.predicted_output_std(), dm.predicted_output_std(), places=8)
        unstable = al.theta_from_fir_ar([0.0, -1.0, 0.0], [2.1, -1.2])
        th1, n1 = al.stabilize_theta(unstable, 0.99)
        thm, nm = am.stabilize_theta(unstable[:, None], 1, 0.99)
        self.assertEqual(n1, nm)
        np.testing.assert_allclose(th1, thm[:, 0], atol=1e-10)

    def test_masked_rls_matches_batch(self):
        rng = np.random.default_rng(0)
        n, m, W, T = 6, 3, 300, 900
        Phi = rng.standard_normal((T, n))
        Y = Phi @ rng.standard_normal((n, m)) + 0.1 * rng.standard_normal((T, m))
        masks = am.own_ar_masks(1, 3)                  # 3 modes, N = 1: y_i and all u
        rls = am.MimoSlidingRLS(n, m, W, refresh_every=10**9, masks=masks)
        for k in range(T):
            rls.update(Phi[k], Y[k])
        for j, ix in enumerate(masks):
            X = Phi[-W:, ix]
            b = np.linalg.solve(1e-6 * np.eye(len(ix)) + X.T @ X, X.T @ Y[-W:, j])
            np.testing.assert_allclose(rls.theta[ix, j], b, rtol=1e-6, atol=1e-8)
            self.assertEqual(np.abs(np.delete(rls.theta[:, j], ix)).max(), 0.0)

    def test_stabilize_theta_filters(self):
        """Shift filters: the buffer move. Cascade filters: the model poles outside the
        radius land on it, the others stay."""
        from specula.lib.nonminimal_observer import cascade_filter
        rng = np.random.default_rng(0)
        m, N = 2, 5
        Th = 0.4 * rng.standard_normal((2 * N * m, m))
        Th[:m] += 1.2 * np.eye(m)
        a, na = am.stabilize_theta(Th, m, 0.99)
        b, nb = am.stabilize_theta_filters(Th, np.eye(N, k=-1), np.eye(N)[0], m, 0.99)
        self.assertEqual(na, nb)
        np.testing.assert_allclose(a, b, atol=1e-12)
        Lam, ell = cascade_filter([0, 0, 0.5, 0.5, 0.5])
        F, Gy, Gu = am.filter_matrices(Lam, ell, m)
        t2, n = am.stabilize_theta_filters(Th, Lam, ell, m, 0.99)
        ev0 = np.linalg.eigvals(am.filter_model(Th, F, Gy, Gu)[0])
        ev1 = np.linalg.eigvals(am.filter_model(t2, F, Gy, Gu)[0])
        self.assertGreater(n, 0)
        self.assertLessEqual(np.abs(ev1).max(), 0.99 + 1e-9)
        for x in ev0[np.abs(ev0) <= 0.99]:
            self.assertLess(np.min(np.abs(ev1 - x)), 1e-8)

    def test_own_plant_masks(self):
        """'diag_plant': mode i sees every past measurement and only its own commands."""
        N, m = 3, 4
        masks = am.own_plant_masks(N, m)
        for i, ix in enumerate(masks):
            np.testing.assert_array_equal(ix[:N * m], np.arange(N * m))
            np.testing.assert_array_equal((ix[N * m:] - N * m) % m, i)
            self.assertEqual(len(ix), N * (m + 1))

    def test_block_diagonal_theta_gives_siso_controllers(self):
        N = 5
        tha = al.theta_from_fir_ar([0.0, -1.0], [1.5, -0.7, 0.1])
        thb = al.theta_from_fir_ar([0.0, -0.8], [0.9, 0.0, 0.0])
        Th = np.zeros((4 * N, 2))
        Th[0:2 * N:2, 0], Th[2 * N::2, 0] = tha[:N], tha[N:]
        Th[1:2 * N:2, 1], Th[2 * N + 1::2, 1] = thb[:N], thb[N:]
        dm = am.MimoModelLQG(*am.buffer_model(Th, N, 2), np.diag([2.0, 0.5]), 1.0, 10.0)
        f = np.linspace(1.0, 450.0, 40)
        C = am.frequency_response(*dm.ctrl, f, 1e-3)
        for i, (t_, q) in enumerate(((tha, 2.0), (thb, 0.5))):
            d1 = al.ModelLQG(*al.buffer_model(t_), q, 1.0, 10.0)
            np.testing.assert_allclose(C[:, i, i], al.frequency_response(*d1.ctrl, f, 1e-3), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(C[:, 0, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(C[:, 1, 0], 0.0, atol=1e-10)


class TestAdaptiveLqgVarLib(unittest.TestCase):

    def test_structured_model_is_the_siso_model_for_one_mode(self):
        """m = 1: the structured model gives the controller of the free-theta design on
        theta_from_fir_ar (same innovations model, other realization)."""
        g, a = np.array([0.0, -0.8, -0.2]), np.array([1.3, -0.5, 0.1])
        f = np.linspace(1.0, 450.0, 40)
        for s, rho in ((0.0, 1.0), (10.0, 0.0)):
            d1 = al.ModelLQG(*al.buffer_model(al.theta_from_fir_ar(g, a)), 2.0, rho, s)
            dv = am.MimoModelLQG(*avar.structured_model(g[None, :], a[:, None, None]), [[2.0]], rho, s)
            np.testing.assert_allclose(am.frequency_response(*dv.ctrl, f, 1e-3)[:, 0, 0],
                                       al.frequency_response(*d1.ctrl, f, 1e-3), rtol=1e-7, atol=1e-9)

    def test_implied_plant_is_g_and_state_rebuild(self):
        rng = np.random.default_rng(0)
        G = np.array([[0.0, -0.8, -0.1], [0.05, -0.6, 0.0]])
        A = 0.2 * rng.standard_normal((4, 2, 2))
        As, Bs, Cs, Ks = avar.structured_model(G, A)
        f = np.array([3.0, 60.0, 300.0])
        z = np.exp(2j * np.pi * f * 1e-3)
        P = am.frequency_response(As, Bs, Cs, 0.0, f, 1e-3)
        for zz, Pz in zip(z, P):
            np.testing.assert_allclose(Pz, np.diag([sum(G[i, j] * zz ** -(j + 1) for j in range(3))
                                                    for i in range(2)]), atol=1e-12)
        # the state rebuilt from histories is the state the model propagates
        T = 40
        U, E = rng.standard_normal((T, 2)), rng.standard_normal((T, 2))
        x, Y = np.zeros(As.shape[0]), np.zeros((T, 2))
        for k in range(T):
            Y[k] = Cs @ x + E[k]
            x = As @ x + Bs @ U[k] + Ks @ E[k]
        k = T - 1
        yh, uh = Y[k::-1][:8], U[k::-1][:8]                   # y_k, y_{k-1}, ... = history at k + 1
        # before k = p + n_g the true state has zero pre-history; compare at the end only
        np.testing.assert_allclose(avar.state_from_history(G, A, yh, uh), x, atol=1e-10)

    def test_bias_state_rejects_a_static_aberration(self):
        """The DC states: m more states, identity in C, and the loop cancels a steady
        disturbance that the AR alone (poles < 1) leaves partly uncorrected."""
        rng = np.random.default_rng(3)
        m = 2
        G = np.array([[0.0, -0.7, -0.05], [0.0, -0.6, 0.0]])
        A = np.zeros((3, m, m))
        A[0], A[1] = 0.8 * np.eye(m), 0.1 * np.eye(m)
        pole = float(np.exp(-1e-3 / 2.0))
        n_plain = avar.structured_model(G, A)[0].shape[0]
        As, Bs, Cs, Ks = avar.structured_model(G, A, pole)
        self.assertEqual(As.shape[0], n_plain + m)
        np.testing.assert_allclose(Cs[:, -m:], np.eye(m), atol=0)
        np.testing.assert_allclose(As[-m:, -m:], pole * np.eye(m), atol=0)
        Q = avar.bias_noise(G, A, 0.1 * np.ones(m))
        self.assertEqual(Q.shape, (n_plain + m, n_plain + m))
        taps = [np.diag(G[:, j]) for j in range(G.shape[1])]
        out = {}
        for label, kw in (("no bias", {}), ("bias", dict(Q_extra=Q))):
            model = avar.structured_model(G, A, pole) if kw else avar.structured_model(G, A)
            d = am.tune_mimo_model(*model, np.eye(m), 1e-3, ms_limit_db=6.0,
                                   gain_range=(0.5, 1.5), delay_margin=0.5, **kw)
            Ac, Bc, Cc, Dc = d.ctrl
            x, ul = np.zeros(Ac.shape[0]), [np.zeros(m)] * (G.shape[1] + 1)
            res = np.zeros(8000)
            for k in range(len(res)):
                y = np.array([50.0, 0.0]) + sum(taps[j] @ ul[j] for j in range(G.shape[1]))
                u = Cc @ x + Dc @ y
                x = Ac @ x + Bc @ y
                ul = [u] + ul[:-1]
                res[k] = y[0]
            out[label] = abs(res[-2000:].mean())
        self.assertLess(out["bias"], 0.3 * out["no bias"])

    def test_hard_abort_fires_on_one_frame(self):
        """One sample above hard_abort_ratio aborts the trial at once; the averaged test
        cannot fire before abort_min_frames, which an unstable loop does not leave time for."""
        for y2, expect in ((9.0, False), (25.0, True)):
            c = avar.AdaptiveVarLQG(1e-3, 3, hard_abort_ratio=4.0, abort_min_frames=20)
            c.incumbent_ms = 1.0
            c.trial = dict(prev=None, cand="candidate", start=c.k, info={})
            c.trial_fast = 1.0
            c._watch_trial(y2)
            aborted = [info for _, e, info in c.log if e == "trial aborted"]
            self.assertEqual(c.trial is None, expect)
            self.assertEqual(len(aborted), 1 if expect else 0)
            if expect:
                self.assertTrue(aborted[0]["instantaneous"])
                self.assertEqual(aborted[0]["after"], 0)

    def test_vector_ar_fit(self):
        rng = np.random.default_rng(1)
        At = np.array([[[0.6, 0.2], [-0.1, 0.5]], [[0.2, 0.0], [0.05, 0.3]]])
        D = np.zeros((20000, 2))
        for k in range(2, len(D)):
            D[k] = At[0] @ D[k - 1] + At[1] @ D[k - 2] + rng.standard_normal(2)
        A, S, moved = avar.fit_var(D, 2)
        np.testing.assert_allclose(A, At, atol=0.03)
        self.assertEqual(moved, 0)
        e_var = np.sum(avar.var_prediction_error(D[-3000:], A, 2) ** 2)
        e_ar = np.sum(avar.var_prediction_error(D[-3000:], avar.fit_var(D, 2, diag=True)[0], 2) ** 2)
        self.assertLess(e_var, e_ar)


class TestDataDrivenLqgObject(unittest.TestCase):

    def _loop(self, obj, d, t_step, M=None):
        """Drive the object in a SPECULA-like loop: controller delay 1 inside
        the object, DM layer read one frame later (dm.out_layer:-1), so
        y_k = d_k - M out_comm_{k-1} (M: modal cross-talk, identity by default)."""
        T, n = d.shape
        M = np.eye(n) if M is None else M
        meas = BaseValue(value=np.zeros(n), target_device_idx=-1)
        obj.inputs['delta_comm'].set(meas)
        obj.setup()
        prev_comm = np.zeros(n)
        comm = np.zeros((T, n))
        for k in range(T):
            t = k * t_step
            meas.value = d[k] - M @ prev_comm
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

        # the design output mirrors the event log
        design = cpuArray(obj.outputs['out_design'].value)
        cols = DataDrivenLqg.DESIGN_COLUMNS
        self.assertEqual(design.shape, (1, len(cols) + 3))
        self.assertEqual(design[0, cols.index('active')], 1.0)
        self.assertEqual(design[0, cols.index('n_accepted')], len(accepted))
        np.testing.assert_allclose(design[0, len(cols):], accepted[-1]["g"], atol=1e-5)


    def test_object_leaky_integrator(self):
        """int_ff makes the integrator modes and the warm-up of the LQG modes leaky,
        exactly as the ff of the SPECULA Integrator."""
        simul_params = SimulParams(time_step=0.001)
        T = 1500
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2, 3)])
        ref = Integrator(int_gain=[0.4], ff=[0.98], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))
        for method in ('free_theta', 'var_lqg'):
            # no dither and no design within T: the LQG mode is still in warm-up
            obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[1], int_gain=0.4,
                                int_ff=0.98, method=method, dither_std=0.0,
                                min_samples=10 * T, delay=1, target_device_idx=-1)
            comm = self._loop(obj, d, obj.seconds_to_t(0.001))
            np.testing.assert_allclose(comm, comm_ref, rtol=1e-4, atol=1e-3, err_msg=method)
        with self.assertRaises(ValueError):
            DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[1], int_ff=1.1, target_device_idx=-1)

    def test_object_iir_warmup(self):
        """warmup_num/den make the free-theta warm-up run that IIR, exactly as the
        SPECULA IirFilter; after the first accepted design the LQG takes over."""
        simul_params = SimulParams(time_step=0.001)
        num = [0.19125, -0.65, 0.5]                     # MORFEO tip-tilt IIR
        den = [0.9995, -1.9995, 1.0]
        T = 1500
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2)])
        data = IirFilterData(ordnum=[3, 3], ordden=[3, 3], num=[num, num], den=[den, den],
                             target_device_idx=-1)
        ref = IirFilter(data, delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))
        obj = DataDrivenLqg(simul_params, n_modes=2, lqg_modes=[0, 1], method='free_theta',
                            warmup_num=num, warmup_den=den, dither_std=0.0,
                            min_samples=10 * T, delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001))
        np.testing.assert_allclose(comm, comm_ref, rtol=1e-4, atol=1e-3)

        T = 5000
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2)])
        obj = DataDrivenLqg(simul_params, n_modes=2, lqg_modes=[0, 1], method='free_theta',
                            warmup_num=[num, num], warmup_den=[den, den], n_theta=6,
                            plant_window=2500, min_samples=1500, dither_std=2.0,
                            dither_after=0.5, delay=1, target_device_idx=-1)
        self._loop(obj, d, obj.seconds_to_t(0.001))
        for ctrl in obj.mode_ctrl:
            self.assertTrue(any(ev == "accepted" for _, ev, _ in ctrl.log), ctrl.log)
        with self.assertRaises(ValueError):
            DataDrivenLqg(simul_params, n_modes=2, warmup_num=num, warmup_den=den,
                          target_device_idx=-1)          # structured: no IIR warm-up

    def test_object_free_theta(self):
        simul_params = SimulParams(time_step=0.001)
        T = 5000
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2, 3)])

        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[1], int_gain=0.4,
                            method='free_theta', n_theta=6, plant_window=2500, min_samples=1500,
                            dither_std=2.0, dither_after=0.5, delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001))

        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))
        np.testing.assert_allclose(comm[:, [0, 2]], comm_ref[:, [0, 2]], rtol=1e-4, atol=1e-3)

        ctrl = obj.mode_ctrl[0]
        accepted = [info for _, ev, info in ctrl.log if ev == "accepted"]
        self.assertTrue(accepted, f"no design accepted: {ctrl.log}")
        design = cpuArray(obj.outputs['out_design'].value)
        cols = DataDrivenLqg.DESIGN_COLUMNS
        self.assertEqual(design[0, cols.index('n_accepted')], len(accepted))
        n_turned_down = sum(1 for _, ev, _ in ctrl.log
                            if ev in ("rejected", "trial rejected", "trial aborted"))
        self.assertEqual(design[0, cols.index('n_rejected')], n_turned_down)
        if ctrl.design is not None:
            np.testing.assert_allclose(design[0, len(cols):], ctrl.design.g, atol=1e-5)


    def test_object_mimo_free_theta(self):
        """Two LQG modes jointly on a plant whose commands are rotated by 30 deg
        (misregistration), the third mode on the integrator."""
        simul_params = SimulParams(time_step=0.001)
        T = 6000
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2, 3)])
        a = np.deg2rad(30.0)
        M = np.eye(3)
        M[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]

        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[0, 1], int_gain=0.4,
                            method='mimo_free_theta', n_theta=6, plant_window=2500, min_samples=1500,
                            dither_std=2.0, dither_after=0.5, delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001), M)
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001), M)
        np.testing.assert_allclose(comm[:, 2], comm_ref[:, 2], rtol=1e-4, atol=1e-3)

        self.assertEqual(len(obj.mode_ctrl), 1)
        ctrl = obj.mode_ctrl[0]
        accepted = [info for _, ev, info in ctrl.log if ev == "accepted"]
        self.assertTrue(accepted, f"no design accepted: {ctrl.log}")
        design = cpuArray(obj.outputs['out_design'].value)
        cols = DataDrivenLqg.DESIGN_COLUMNS
        self.assertEqual(design.shape, (2, len(cols) + 3))
        np.testing.assert_array_equal(design[:, cols.index('n_accepted')], len(accepted))
        if ctrl.design is not None:
            np.testing.assert_allclose(design[:, len(cols):], ctrl.design.g, atol=1e-5)

        res = d[1:, :2] - (M @ comm[:-1].T).T[:, :2]
        res_int = d[1:, :2] - (M @ comm_ref[:-1].T).T[:, :2]
        self.assertLess(np.sqrt(np.mean(res[-2000:] ** 2)), np.sqrt(np.mean(res_int[-2000:] ** 2)))


    def test_object_var_lqg(self):
        """Frozen-flow-like pair: mode 1 is mode 0 three frames later, and mode 0 is fast
        (AR(1), 0.8), so only mode 0's past predicts mode 1. var_lqg starts per-mode,
        moves to the vector AR and removes most of mode 1's residual."""
        from scipy.signal import lfilter
        simul_params = SimulParams(time_step=0.001)
        T = 7000
        rng = np.random.default_rng(4)
        d0 = lfilter([1.0], [1.0, -0.8], rng.standard_normal(T))
        d0 = 50.0 * d0 / d0.std()
        d1 = np.r_[np.zeros(3), d0[:-3]] + 5.0 * rng.standard_normal(T)
        d = np.column_stack([d0, d1, ar2_turbulence(T, seed=3, std=50.0)])

        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[0, 1], int_gain=0.4,
                            method='var_lqg', p=4, n_g=3, plant_window=2500, min_samples=1500,
                            dither_std=5.0, delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001))
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))
        np.testing.assert_allclose(comm[:, 2], comm_ref[:, 2], rtol=1e-4, atol=1e-3)

        ctrl = obj.mode_ctrl[0]
        accepted = [info for _, ev, info in ctrl.log if ev == "accepted"]
        self.assertTrue(accepted, f"no design accepted: {ctrl.log}")
        self.assertEqual(accepted[0]["structure"], "per-mode AR")
        self.assertIn("vector AR", [a["structure"] for a in accepted])
        # lag-2 taps: -1 (IV on a 5 nm dither against a fast 50 nm disturbance, 2500 frames)
        np.testing.assert_allclose(np.asarray(accepted[-1]["g"])[:, 1], -1.0, atol=0.3)
        design = cpuArray(obj.outputs['out_design'].value)
        np.testing.assert_allclose(design[:, len(DataDrivenLqg.DESIGN_COLUMNS):], ctrl.design.g, atol=1e-5)
        res1 = d[1:, 1] - comm[:-1, 1]
        res1_int = d[1:, 1] - comm_ref[:-1, 1]
        self.assertLess(np.sqrt(np.mean(res1[-2000:] ** 2)), 0.5 * np.sqrt(np.mean(res1_int[-2000:] ** 2)))


    def test_object_mimo_free_theta_diag_plant(self):
        """The frozen-flow pair of test_object_var_lqg with a free Theta, coupled in the
        measurement part only: mode 1 is predicted from mode 0's past."""
        from scipy.signal import lfilter
        simul_params = SimulParams(time_step=0.001)
        T = 7000
        rng = np.random.default_rng(4)
        d0 = lfilter([1.0], [1.0, -0.8], rng.standard_normal(T))
        d0 = 50.0 * d0 / d0.std()
        d1 = np.r_[np.zeros(3), d0[:-3]] + 5.0 * rng.standard_normal(T)
        d = np.column_stack([d0, d1, ar2_turbulence(T, seed=3, std=50.0)])

        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[0, 1], int_gain=0.4,
                            method='mimo_free_theta', mimo_structure='diag_plant', n_theta=6,
                            plant_window=2500, min_samples=1500, dither_std=5.0, delay=1,
                            target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001))
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))

        ctrl = obj.mode_ctrl[0]
        self.assertTrue([1 for _, ev, _ in ctrl.log if ev == "accepted"], f"no design accepted: {ctrl.log}")
        Theta = ctrl.design.C.T                           # (2Nm, m)
        N, m = 6, 2
        self.assertEqual(np.abs(Theta[N * m + 1::m, 0]).max(), 0.0)   # no u_1 in mode 0
        self.assertEqual(np.abs(Theta[N * m::m, 1]).max(), 0.0)       # no u_0 in mode 1
        res1 = d[1:, 1] - comm[:-1, 1]
        res1_int = d[1:, 1] - comm_ref[:-1, 1]
        self.assertLess(np.sqrt(np.mean(res1[-2000:] ** 2)), 0.5 * np.sqrt(np.mean(res1_int[-2000:] ** 2)))

    def test_object_mimo_free_theta_filters(self):
        """test_object_mimo_free_theta on stable, non-deadbeat filters: the controller
        state is the filter state and the loop beats the integrator."""
        simul_params = SimulParams(time_step=0.001)
        T = 6000
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2, 3)])
        a = np.deg2rad(30.0)
        M = np.eye(3)
        M[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[0, 1], int_gain=0.4,
                            method='mimo_free_theta', theta_poles=[0, 0, 0.5, 0.5, 0.5, 0.5],
                            theta_refit_u=True, plant_window=2500, min_samples=1500,
                            dither_std=2.0, dither_after=0.5, delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001), M)
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001), M)
        ctrl = obj.mode_ctrl[0]
        self.assertEqual(ctrl.N, 6)
        self.assertTrue([1 for _, ev, _ in ctrl.log if ev == "accepted"], f"no design accepted: {ctrl.log}")
        res = d[1:, :2] - (M @ comm[:-1].T).T[:, :2]
        res_int = d[1:, :2] - (M @ comm_ref[:-1].T).T[:, :2]
        self.assertLess(np.sqrt(np.mean(res[-2000:] ** 2)), np.sqrt(np.mean(res_int[-2000:] ** 2)))

    def test_object_var_lqg_blocks(self):
        """var_block_size splits the LQG modes into independent vector-AR controllers:
        with blocks of one mode each, every block runs its own design on its mode only."""
        simul_params = SimulParams(time_step=0.001)
        T = 6000
        d = np.column_stack([ar2_turbulence(T, seed=s, std=50.0) for s in (1, 2, 3)])
        obj = DataDrivenLqg(simul_params, n_modes=3, lqg_modes=[0, 2], int_gain=0.4,
                            method='var_lqg', p=4, n_g=3, plant_window=2500, min_samples=1500,
                            dither_std=5.0, var_block_size=1, delay=1, target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001))
        self.assertEqual(len(obj.mode_ctrl), 2)
        self.assertEqual([c.m for c in obj.mode_ctrl], [1, 1])
        for ctrl in obj.mode_ctrl:
            accepted = [info for _, ev, info in ctrl.log if ev == "accepted"]
            self.assertTrue(accepted, f"no design accepted in {ctrl.name}: {ctrl.log}")
        # mode 1 stays on the integrator
        ref = Integrator(int_gain=[0.4], n_modes=[3], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001))
        np.testing.assert_allclose(comm[:, 1], comm_ref[:, 1], rtol=1e-4, atol=1e-3)

    def test_object_identification_only_modes(self):
        """ident_modes are dithered and identified but stay on the integrator: the taps
        must recover the per-mode gain of the plant, and the dither must perturb the
        applied command without being accumulated by the integrator."""
        simul_params = SimulParams(time_step=0.001)
        T, n = 6000, 5
        gains = np.array([1.0, 0.9, 0.7, 0.5, 0.3])       # the 'optical gain' of each mode
        d = np.column_stack([ar2_turbulence(T, seed=10 + i, std=60.0) for i in range(n)])

        obj = DataDrivenLqg(simul_params, n_modes=n, lqg_modes=[0], int_gain=0.4,
                            method='var_lqg', p=4, n_g=3, plant_window=2500, min_samples=1500,
                            dither_std=5.0, ident_modes=[1, 2, 3, 4], ident_dither=5.0,
                            ident_window=2500, ident_min_samples=1500, delay=1,
                            target_device_idx=-1)
        comm = self._loop(obj, d, obj.seconds_to_t(0.001), M=np.diag(gains))
        ref = Integrator(int_gain=[0.4], n_modes=[n], delay=1, target_device_idx=-1)
        comm_ref = self._loop(ref, d, ref.seconds_to_t(0.001), M=np.diag(gains))

        ident = obj.ident
        self.assertIsNotNone(ident)
        done = [info for _, ev, info in ident.log if ev == "identified"]
        self.assertTrue(done, f"nothing identified: {ident.log}")
        dc = np.asarray(done[-1]["dc"], float)            # |G_i(1)| of modes 1..4
        np.testing.assert_allclose(dc, gains[1:], atol=0.15)
        # the response sits at lag 2 (delay 1 in the object, DM read one frame later),
        # with the plant sign; lags 1 and 3 must be empty
        g = np.asarray(done[-1]["g"], float)
        np.testing.assert_allclose(g[:, 1], -gains[1:], atol=0.15)
        np.testing.assert_allclose(g[:, [0, 2]], 0.0, atol=0.15)

        # the dither perturbs the command, it is not integrated: a random walk would make
        # the identified modes' commands grow without bound
        for i in (1, 2, 3, 4):
            self.assertLess(np.std(comm[-1500:, i]), 1.6 * np.std(comm_ref[-1500:, i]))
            self.assertLess(abs(np.mean(comm[-1500:, i] - comm_ref[-1500:, i])),
                            4.0 * np.std(comm_ref[-1500:, i]))
        # mode 0 is the LQG one and must be untouched by the identifier
        self.assertEqual(len(obj.mode_ctrl), 1)


if __name__ == '__main__':
    unittest.main()
