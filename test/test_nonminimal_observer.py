import unittest

import numpy as np
from scipy.signal import lfilter

from specula.lib.nonminimal_observer import (MimoObserver, ModeObserver, white_noise_var, cascade_filter, default_poles,
                                             fit_theta, implied_taps, luenberger,
                                             realization, run_filters)


def closed_loop(T=10000, og=0.8, gain=0.4, dither=1.0, noise=0.01, seed=0):
    """y = d - og u_{k-2} + n, d = AR(2) resonance at 30 Hz (1 kHz) + slow AR(1),
    integrator u_k = u_{k-1} + gain y_k plus white dither."""
    rng = np.random.default_rng(seed)
    r, w = 0.99, 2 * np.pi * 30e-3
    d1, d2 = np.zeros(T), np.zeros(T)
    for k in range(2, T):
        d1[k] = 2 * r * np.cos(w) * d1[k - 1] - r * r * d1[k - 2] + rng.standard_normal()
        d2[k] = 0.995 * d2[k - 1] + 0.5 * rng.standard_normal()
    d = d1 + d2
    y, u, r = np.zeros(T), np.zeros(T), dither * rng.standard_normal(T)
    u_int = 0.0
    for k in range(T):
        y[k] = d[k] - og * (u[k - 2] if k >= 2 else 0.0) + noise * rng.standard_normal()
        u_int += gain * y[k]
        u[k] = u_int + r[k]
    return y, u, d, r


def colored_loop(d, n, r, og, gain=0.4):
    """The loop of closed_loop on a given disturbance d, sensor error n and dither r."""
    y, u = np.zeros(len(d)), np.zeros(len(d))
    u_int = 0.0
    for k in range(len(d)):
        y[k] = d[k] - og * (u[k - 2] if k >= 2 else 0.0) + n[k]
        u_int += gain * y[k]
        u[k] = u_int + r[k]
    return y, u


class TestNonminimalObserver(unittest.TestCase):

    def test_zero_poles_are_the_command_buffer(self):
        Lam, ell = cascade_filter(default_poles(8, 3, 0.9))
        u = np.random.default_rng(1).standard_normal(50)
        Z = run_filters(Lam, ell, np.zeros(50), u)
        for i in range(3):
            np.testing.assert_allclose(Z[10:, 8 + i], u[9 - i:49 - i], atol=1e-14)

    def test_realization_is_the_luenberger_observer(self):
        """Any theta: theta^T zeta is the one-step predictor of the order-N model it implies,
        with observer poles spec(Lambda)."""
        rng = np.random.default_rng(2)
        poles = np.r_[0.0, 0.0, 0.5, 0.8, 0.9, 0.95]
        Lam, ell = cascade_filter(poles)
        theta = rng.standard_normal(12)
        A, B, C, L = realization(Lam, ell, theta)
        np.testing.assert_allclose(np.poly(A - L @ C), np.poly(poles), atol=1e-9)
        y, u = rng.standard_normal(300), rng.standard_normal(300)
        np.testing.assert_allclose(run_filters(Lam, ell, y, u) @ theta,
                                   luenberger(A, B, C, L, y, u), atol=1e-9)

    def test_known_plant_taps(self):
        """Noise-free, open loop: the fitted theta gives the plant taps exactly."""
        rng = np.random.default_rng(3)
        T = 3000
        u = rng.standard_normal(T)
        y = np.zeros(T)
        y[2:] = -0.7 * u[:-2] + 0.1 * u[1:-1]
        Lam, ell = cascade_filter(default_poles(6, 3, 0.8))
        theta = fit_theta(run_filters(Lam, ell, y, u)[100:], y[100:])
        np.testing.assert_allclose(implied_taps(Lam, ell, theta, 3), [0.1, -0.7, 0.0], atol=1e-6)

    def test_tracks_disturbance(self):
        og = 0.8
        y, u, d, r = closed_loop(og=og)
        test = slice(6500, None)
        for fit, pole in (("ls", 0.0), ("ls", 0.9), ("iv", 0.0), ("iv", 0.9)):
            obs = ModeObserver(N=8, n_g=3, slow_pole=pole, fit_start=200, fit_frames=6000, fit=fit)
            out = np.array([obs.step(*v) for v in zip(y, u, r)])
            np.testing.assert_allclose(obs.g, [0.0, -og, 0.0], atol=0.1)
            self.assertTrue(np.all(np.isnan(out[:6199, 0])))
            err = np.std(out[test, 0] - d[test]) / np.std(d[test])
            self.assertLess(err, 0.12 if pole == 0.0 else 0.25)   # little noise: slow lags
            # beats persistence (d_hat_{k|k-1} = d_hat_{k-1|k-1})
            self.assertLess(err, np.std(out[test, 1][:-1] - d[test][1:]) / np.std(d[test]))

    def test_slow_observer_rejects_measurement_noise(self):
        """Sensor noise dominating the disturbance innovation: the slow observer gives a
        smaller disturbance error (with less noise the deadbeat one wins: speed is a trade-off)."""
        y, u, d, r = closed_loop(noise=10.0, seed=4)
        test = slice(6500, None)
        err = {}
        for pole in (0.0, 0.9):
            obs = ModeObserver(N=8, n_g=3, slow_pole=pole, fit_start=200, fit_frames=6000, fit="iv")
            out = np.array([obs.step(*v) for v in zip(y, u, r)])
            err[pole] = np.std(out[test, 0] - d[test])
        self.assertLess(err[0.9], err[0.0])

    def test_iv_is_unbiased_where_least_squares_is_not(self):
        """Colored sensor error (as the pyramid's aliasing): the closed-loop least squares on
        the free theta scatters and is biased, the dither-instrumented gain stays within 3 of
        its own standard deviations of the truth."""
        og, dev = 0.8, {"ls": [], "iv": []}
        for seed in range(6):
            rng = np.random.default_rng(seed)
            T = 20000
            y, u, d, r = closed_loop(T=T, og=og, dither=0.5, noise=0.0, seed=seed)
            n = 2.0 * lfilter([np.sqrt(1 - 0.95 ** 2)], [1, -0.95], rng.standard_normal(T))
            y, u = colored_loop(d, n, r, og)
            for fit in dev:
                obs = ModeObserver(N=8, n_g=3, slow_pole=0.0, fit_start=200, fit_frames=19000, fit=fit)
                for v in zip(y, u, r):
                    obs.step(*v)
                dev[fit].append(-obs.g.sum() - og)
                if fit == "iv":
                    self.assertLess(abs(dev[fit][-1]), 3 * obs.og_std)
        rms = {fit: np.sqrt(np.mean(np.square(v))) for fit, v in dev.items()}
        self.assertLess(rms["iv"], 0.5 * rms["ls"])

    def test_cold_start_transient_lasts_what_lambda_says(self):
        """Cold start: the error decays with spec(Lambda); after it, the output equals the warm
        observer's (theta is the same, only the filter initial state differs)."""
        y, u, d, r = closed_loop()
        for pole, frames in ((0.0, 8 + 3), (0.9, 200)):
            out = {}
            for cold in (False, True):
                obs = ModeObserver(N=11, n_g=3, slow_pole=pole, fit_start=200, fit_frames=6000,
                                   cold_start=cold)
                out[cold] = np.array([obs.step(*v) for v in zip(y, u, r)])[:, 0]
            k0 = 6200                                       # first output frame
            diff = np.abs(out[True] - out[False]) / np.std(out[False][k0:])
            self.assertGreater(diff[k0:k0 + frames].max(), 0.1)   # starts from zero: off
            self.assertLess(diff[k0 + frames:].max(), 1e-2 if pole else 1e-9)


    def test_mimo_with_one_mode_is_the_siso_observer(self):
        y, u, d, r = closed_loop()
        for pole, cold in ((0.0, False), (0.9, True)):
            siso = ModeObserver(N=11, n_g=3, slow_pole=pole, fit_start=200, fit_frames=6000,
                                cold_start=cold)
            mimo = MimoObserver(1, N=11, n_g=3, slow_pole=pole, fit_start=200, fit_frames=6000,
                                cold_start=cold)
            a = np.array([siso.step(*v) for v in zip(y, u, r)])
            b = np.array([np.r_[mimo.step([yk], [uk], [rk])] for yk, uk, rk in zip(y, u, r)])
            np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-8 * np.nanstd(a))

    def test_mimo_uses_the_other_modes(self):
        """Mode 2's disturbance is mode 1's, 3 frames later (frozen flow): the joint model
        predicts mode 2 better than its own past does."""
        rng = np.random.default_rng(7)
        T = 10000
        y1, u1, d1, r1 = closed_loop(T=T, seed=7)
        d2 = np.r_[np.zeros(3), d1[:-3]] + 0.3 * rng.standard_normal(T)
        y2, u2 = colored_loop(d2, 0.01 * rng.standard_normal(T), 1.0 * rng.standard_normal(T), 0.8)
        r2 = u2 - np.r_[0.0, u2[:-1]] - 0.4 * y2          # dither of the second loop
        Y, U, R = np.c_[y1, y2], np.c_[u1, u2], np.c_[r1, r2]
        test = slice(6500, None)
        mimo = MimoObserver(2, N=8, n_g=3, slow_pole=0.0, fit_start=200, fit_frames=6000)
        out_m = np.array([mimo.step(*v)[0] for v in zip(Y, U, R)])
        siso = ModeObserver(N=8, n_g=3, slow_pole=0.0, fit_start=200, fit_frames=6000)
        out_s = np.array([siso.step(*v)[0] for v in zip(y2, u2, r2)])
        err_m = np.std(out_m[test, 1] - d2[test])
        err_s = np.std(out_s[test] - d2[test])
        self.assertLess(err_m, 0.8 * err_s)


    def test_white_noise_variance_from_autocovariance(self):
        rng = np.random.default_rng(8)
        x = lfilter([1.0], [1.0, -0.99], rng.standard_normal(20000))   # smooth, var ~ 50
        n = 0.5 * rng.standard_normal(20000)                             # white, var 0.25
        self.assertAlmostEqual(white_noise_var(x + n), 0.25, delta=0.05)
        self.assertLess(white_noise_var(x), 0.05)

    def test_kalman_update_and_two_step_prediction(self):
        """Noisy sensor: the Kalman update beats both the one-step prediction and the raw
        reconstruction; the two-step prediction beats persistence over two frames."""
        y, u, d, r = closed_loop(noise=3.0, seed=4)
        test = slice(6500, None)
        for obs in (ModeObserver(N=11, n_g=3, slow_pole=0.0, fit_start=200, fit_frames=6000),
                    MimoObserver(1, N=11, n_g=3, slow_pole=0.0, fit_start=200, fit_frames=6000)):
            if isinstance(obs, MimoObserver):
                out = np.array([np.r_[obs.step([a], [b], [c])] for a, b, c in zip(y, u, r)])
            else:
                out = np.array([obs.step(*v) for v in zip(y, u, r)])
            err = lambda x: np.std(x[test] - d[test])
            e_pred, e_meas, e_kal = err(out[:, 0]), err(out[:, 1]), err(out[:, 3])
            self.assertLess(e_kal, 0.95 * min(e_pred, e_meas))
            pred2 = np.r_[np.full(2, np.nan), out[:-2, 4]]   # output at k predicts d_{k+2}
            pers2 = np.r_[np.full(2, np.nan), out[:-2, 1]]   # last measurement, two frames old
            self.assertLess(err(pred2), err(pers2))
            self.assertLess(e_pred, err(pred2))              # further ahead is harder


if __name__ == "__main__":
    unittest.main()
