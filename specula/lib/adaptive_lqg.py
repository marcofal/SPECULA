"""Data-driven adaptive LQG for one modal channel.

Port of the RAMA prototype (theta_lqr.py + adaptive_theta_lqr.py, LQG branch)
with no dependency outside numpy/scipy. Everything runs on the CPU in float64.

Model, identified online from closed-loop data (no plant model assumed):

    y_k = sum_{i=1..n_g} g_i u_{k-i} + d_k + n_k,     A_d(q) d = e

  * u is the command written by the controller at frame k (dither included),
    y the modal measurement at frame k. The taps g absorb the loop latency
    (fractional delays included), the modal optical gain and its sign.
  * g: sliding-window instrumental variables, the dither r as instrument,
    signals prefiltered with the current A_d (whitens the turbulence).
  * A_d: sliding-window least squares on d_hat = y - G u.

Every `redesign_every` frames the window samples are recomputed with the
current estimates (re-anchor), the plant is gated on its IV covariance, and a
Kalman + LQR design on (g, A_d) is tuned for the smallest predicted residual
under robustness checks done on the identified plant only (peak sensitivity,
gain range, extra fractional delay). A supervisor reverts a design that makes
the residual blow up. Before the first accepted design an integrator runs.

Sign convention: y = d + P u, S = 1 / (1 - P C). In a SPECULA loop with an
interaction matrix calibrated on the same DM, P is about -z^-2 (controller
delay 1 plus the DM layer read at ':-1'), so the warm-up integrator is
u_k = u_{k-1} + g y_k, exactly as the SPECULA Integrator.
"""

from collections import deque

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
from scipy.signal import lfilter


# ------------------------------------------------------------------ #
# Sliding-window recursive least squares / instrumental variables    #
# ------------------------------------------------------------------ #
class SlidingWindowRLS:
    """theta = (delta I + sum z phi^T)^-1 sum z y over the last `window` samples.
    Pass z = None for ordinary least squares (z = phi). The inverse is updated
    with rank-one add/downdate steps and recomputed exactly every
    `refresh_every` samples to stop round-off drift."""

    def __init__(self, n, window, delta=1e-6, refresh_every=None):
        self.n, self.window, self.delta = n, int(window), float(delta)
        self.refresh_every = self.window if refresh_every is None else int(refresh_every)
        self.buf = deque()
        self.reset()

    def reset(self):
        self.P = np.eye(self.n) / self.delta
        self.theta = np.zeros(self.n)
        self.buf.clear()
        self._since_refresh = 0

    def _add(self, phi, z, y):
        Pz = self.P @ z
        K = Pz / (1.0 + phi @ Pz)
        self.theta = self.theta + K * (y - phi @ self.theta)
        self.P = self.P - np.outer(K, phi @ self.P)

    def _remove(self, phi, z, y):
        Pz = self.P @ z
        K = Pz / (1.0 - phi @ Pz)
        self.theta = self.theta - K * (y - phi @ self.theta)
        self.P = self.P + np.outer(K, phi @ self.P)

    def update(self, phi, y, z=None):
        phi = np.asarray(phi, float)
        z = phi if z is None else np.asarray(z, float)
        self.buf.append((phi, z, float(y)))
        self._add(phi, z, y)
        if len(self.buf) > self.window:
            self._remove(*self.buf.popleft())
        self._since_refresh += 1
        if self._since_refresh >= self.refresh_every:
            self.refresh()
        return self.theta

    def load(self, Phi, Y, Z=None):
        """Replace the window with the given samples (last `window` rows)."""
        Z = Phi if Z is None else Z
        self.buf.clear()
        for phi, z, y in zip(Phi[-self.window:], Z[-self.window:], Y[-self.window:]):
            self.buf.append((np.asarray(phi, float), np.asarray(z, float), float(y)))
        self.refresh()

    def refresh(self):
        M = self.delta * np.eye(self.n)
        b = np.zeros(self.n)
        for phi, z, y in self.buf:
            M += np.outer(z, phi)
            b += z * y
        self.P = np.linalg.inv(M)
        self.theta = self.P @ b
        self._since_refresh = 0

    def residual_variance(self, theta=None):
        """Variance of y - phi^T theta over the window (theta: current estimate by default)."""
        if not self.buf:
            return np.nan
        Phi = np.array([phi for phi, _, _ in self.buf])
        Y = np.array([y for _, _, y in self.buf])
        return float(np.var(Y - Phi @ (self.theta if theta is None else theta)))

    @property
    def full(self):
        return len(self.buf) >= self.window


# ------------------------------------------------------------------ #
# Batch identification                                               #
# ------------------------------------------------------------------ #
def lagged(x, n, first=1):
    """Columns x_{k-first}, ..., x_{k-first-n+1} (zeros before the start)."""
    x = np.asarray(x, float)
    T = len(x)
    return np.column_stack([np.r_[np.zeros(min(i, T)), x[:max(T - i, 0)]]
                            for i in range(first, first + n)])


def stabilize_ar(a, max_radius=0.9995):
    """Pull the roots of 1 - a_1 z^-1 - ... inside max_radius."""
    a = np.asarray(a, float)
    roots = np.roots(np.r_[1.0, -a])
    big = np.abs(roots) > max_radius
    if np.any(big):
        roots[big] *= max_radius / np.abs(roots[big])
        return -np.real(np.poly(roots))[1:]
    return a.copy()


def fit_ar(d, p, skip=0, max_radius=0.9995):
    """AR(p) on d[skip:], roots pulled inside max_radius. Returns (a, q)."""
    d = np.asarray(d, float)
    D = lagged(d, p)[skip:]
    a, *_ = np.linalg.lstsq(D, d[skip:], rcond=None)
    a = stabilize_ar(a, max_radius)
    return a, float(np.var(d[skip:] - D @ a))


def fit_plant_iv(y, u, r, n_g, p=4, iters=3, skip=100, a0=None):
    """FIR plant y = sum_{i=1..n_g} g_i u_{k-i} + d with the dither r as
    (square) instrument, signals prefiltered with the AR polynomial of the
    disturbance. Returns (g, covariance of g)."""
    y, u, r = (np.asarray(v, float) for v in (y, u, r))
    prefilter = np.array([1.0]) if a0 is None else np.r_[1.0, -np.asarray(a0, float)]
    for _ in range(iters + 1):
        yf, uf, rf = (lfilter(prefilter, [1.0], v) for v in (y, u, r))
        X, Zi = lagged(uf, n_g)[skip:], lagged(rf, n_g)[skip:]
        g, *_ = np.linalg.lstsq(Zi.T @ X, Zi.T @ yf[skip:], rcond=None)
        a, _ = fit_ar(y - lagged(u, n_g) @ g, p, skip)
        prefilter = np.r_[1.0, -a]
    s2 = float(np.var(yf[skip:] - X @ g))
    ZX_pinv = np.linalg.pinv(Zi.T @ X)
    return g, s2 * ZX_pinv @ (Zi.T @ Zi) @ ZX_pinv.T


# ------------------------------------------------------------------ #
# Plants and robustness checks                                       #
# ------------------------------------------------------------------ #
def fir_plant(g):
    """(A, B, C) of y = sum_{i=1..n} g_i u_{k-i}, state (u_{k-1} .. u_{k-n})."""
    g = np.asarray(g, float)
    n = len(g)
    A = np.eye(n, k=-1)
    B = np.zeros((n, 1))
    B[0, 0] = 1.0
    return A, B, g[None, :].copy()


def with_extra_delay(plant, frac):
    """plant * ((1 - frac) + frac q^-1): an extra fractional input delay."""
    A, B, C = plant
    n = A.shape[0]
    A_e = np.block([[A, frac * B], [np.zeros((1, n)), np.zeros((1, 1))]])
    B_e = np.vstack([(1.0 - frac) * B, [[1.0]]])
    return A_e, B_e, np.hstack([C, np.zeros((1, 1))])


def closed_loop_matrix(plant, ctrl, gain=1.0):
    A, B, C = plant
    Ac, Bc, Cc, Dc = ctrl
    return np.block([[A + gain * B @ Dc @ C, gain * B @ Cc],
                     [Bc @ C, Ac]])


def frequency_response(A, B, C, D, freqs_hz, dt):
    z = np.exp(2j * np.pi * np.asarray(freqs_hz) * dt)
    I = np.eye(A.shape[0])
    D = np.atleast_2d(D)
    return np.array([(C @ np.linalg.solve(zz * I - A, B) + D).item() for zz in z])


def peak_sensitivity_db(plant, ctrl, dt, n=500):
    f = np.linspace(0.05, 0.5 / dt, n)
    P = frequency_response(*plant, 0.0, f, dt)
    C = frequency_response(*ctrl, f, dt)
    return float(20 * np.log10(np.max(np.abs(1.0 / (1.0 - P * C)))))


# ------------------------------------------------------------------ #
# Design                                                             #
# ------------------------------------------------------------------ #
class FirArLQG:
    """Kalman + LQR on y = G(q) u + d + n, A_d(q) d = e.

    State chi = (u_{k-1} .. u_{k-n_g}, d_k .. d_{k-p+1}); the controller state
    is the Kalman predictor chi_{k|k-1}.
      q        innovation variance of the disturbance model
      r        measurement noise variance (used for the predicted residual)
      r_kalman noise variance the Kalman filter is designed for (robustness
               knob: larger means a slower, less aggressive estimator)
      rho      penalty on the command change u_k - u_{k-1}
    """

    def __init__(self, g, a, q, r, rho=0.0, r_kalman=None):
        self.g, self.a = np.asarray(g, float), np.asarray(a, float)
        self.q, self.r, self.rho = float(q), float(r), float(rho)
        self.r_kalman = self.r if r_kalman is None else float(r_kalman)
        n_g, p = len(self.g), len(self.a)
        self.n_g, self.p = n_g, p
        N = n_g + p
        A = np.zeros((N, N))
        A[1:n_g, :n_g - 1] = np.eye(n_g - 1)
        A[n_g, n_g:] = self.a
        A[n_g + 1:, n_g:N - 1] = np.eye(p - 1)
        B = np.zeros((N, 1))
        B[0, 0] = 1.0
        C = np.r_[self.g, 1.0, np.zeros(p - 1)][None, :]
        E = np.zeros((N, 1))
        E[n_g, 0] = 1.0
        self.A, self.B, self.C, self.E, self.N = A, B, C, E, N

        W = self.q * (E @ E.T)
        R = np.array([[max(self.r_kalman, 1e-9)]])
        P = solve_discrete_are(A.T, C.T, W, R)
        self.M = P @ C.T / (C @ P @ C.T + R)
        e1 = np.zeros((N, 1))
        e1[0, 0] = 1.0
        Qx = C.T @ C + self.rho * (e1 @ e1.T)
        Ru = np.array([[max(self.rho, 1e-9)]])
        Nx = -self.rho * e1
        X = solve_discrete_are(A, B, Qx, Ru, None, Nx)
        self.K = np.linalg.solve(Ru + B.T @ X @ B, B.T @ X @ A + Nx.T)

        I_MC = np.eye(N) - self.M @ C
        ABK = A - B @ self.K
        self.Ac, self.Bc = ABK @ I_MC, ABK @ self.M
        self.Cc, self.Dc = -self.K @ I_MC, -self.K @ self.M

    @property
    def ctrl(self):
        return self.Ac, self.Bc, self.Cc, self.Dc

    @property
    def identified_plant(self):
        return fir_plant(self.g)

    def predicted_output_std(self):
        """Stationary std of the residual G u + d predicted in closed loop."""
        A, B, C, E = self.A, self.B, self.C, self.E
        Ac, Bc, Cc, Dc = self.ctrl
        n = self.N
        Acl = np.block([[A + B @ Dc @ C, B @ Cc], [Bc @ C, Ac]])
        if np.abs(np.linalg.eigvals(Acl)).max() >= 1.0:
            return np.inf
        Ge = np.vstack([E, np.zeros((n, 1))])
        Gn = np.vstack([B @ Dc, Bc])
        Sigma = solve_discrete_lyapunov(Acl, self.q * Ge @ Ge.T + self.r * Gn @ Gn.T)
        Ccl = np.hstack([C, np.zeros((1, n))])
        return float(np.sqrt((Ccl @ Sigma @ Ccl.T).item()))

    def predictor_state(self, y_hist, u_hist):
        """chi_{k|k-1} rebuilt from (y_{k-1}, ...) and (u_{k-1}, ...), for a
        bumpless switch: exact command lags, disturbance from d_hat = y - G u."""
        n_g, p = self.n_g, self.p
        d_past = np.array([y_hist[i] - self.g @ u_hist[i + 1:i + 1 + n_g] for i in range(p)])
        return np.r_[u_hist[:n_g], self.a @ d_past, d_past[:p - 1]]


def tune_lqg(g, a, q, r, dt, ms_limit_db=6.0, gain_range=(0.5, 1.5), delay_margin=0.0,
             r_scales=None, rhos=None, r_floor_rel=1e-3):
    """(r_kalman, rho) with the smallest predicted residual, subject to
    Ms <= limit and stability for gain x gain_range and +delay_margin frames,
    all on the identified FIR plant.

    The Kalman noise grid is r_base * r_scales with r_base = max(r, r_floor_rel q).
    The floor matters for bright stars: with r close to zero the grid would not
    reach a useful detuning."""
    r_scales = np.logspace(0, 6, 7) if r_scales is None else r_scales
    rhos = np.r_[0.0, np.logspace(-1, 2, 4)] if rhos is None else rhos
    r_base = max(float(r), r_floor_rel * float(q))
    gains = np.linspace(*gain_range, 11)
    plant = fir_plant(g)
    plants = [plant] + ([with_extra_delay(plant, delay_margin)] if delay_margin > 0 else [])
    best = None
    for rs in r_scales:
        for rho in rhos:
            try:
                d = FirArLQG(g, a, q, r, rho, r_base * rs)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if not all(np.abs(np.linalg.eigvals(closed_loop_matrix(pl, d.ctrl, k))).max() < 1
                       for pl in plants for k in gains):
                continue
            if peak_sensitivity_db(plant, d.ctrl, dt) > ms_limit_db:
                continue
            std = d.predicted_output_std()
            if best is None or std < best[0]:
                best = (std, d)
    if best is None:
        raise RuntimeError("no (r_kalman, rho) in the grid meets the constraints "
                           "on the identified plant")
    return best[1]


# ------------------------------------------------------------------ #
# Adaptive controller for one mode                                   #
# ------------------------------------------------------------------ #
class AdaptiveModeLQG:
    """One modal channel. Call step(y_k) once per frame, it returns u_k
    (dither included), the command whose effect the plant taps describe."""

    def __init__(self, dt, n_g=3, p=8, plant_window=4000, dist_window=2000,
                 min_samples=None, dither_std=5.0, redesign_every=250, ms_limit_db=6.0,
                 gain_range=(0.5, 1.5), delay_margin=0.5, plant_rel_std_max=0.25,
                 warmup_gain=0.4, warmup_ff=1.0, noise_var=0.0, max_radius=0.9995, supervisor=True,
                 blowup_factor=2.0, fast_window=50, holdoff=4, rng=None, name=""):
        self.dt, self.n_g, self.p = float(dt), int(n_g), int(p)
        self.min_samples = int(plant_window if min_samples is None else min_samples)
        self.dither_std, self.redesign_every = float(dither_std), int(redesign_every)
        self.ms_limit_db, self.gain_range = float(ms_limit_db), tuple(gain_range)
        self.delay_margin, self.plant_rel_std_max = float(delay_margin), float(plant_rel_std_max)
        self.warmup_gain, self.noise_var = float(warmup_gain), float(noise_var)
        self.warmup_ff = float(warmup_ff)
        self.max_radius = float(max_radius)
        self.supervisor, self.blowup_factor, self.holdoff = supervisor, float(blowup_factor), int(holdoff)
        self.fast_alpha = 1.0 / fast_window
        self.rng = np.random.default_rng() if rng is None else rng
        self.name = name

        self.iv = SlidingWindowRLS(self.n_g, plant_window)
        self.ar = SlidingWindowRLS(self.p, dist_window)
        self.hist_len = self.n_g + 2 * self.p + 1
        self.y_hist = np.zeros(self.hist_len)           # y_{k-1}, y_{k-2}, ...
        self.u_hist = np.zeros(self.hist_len)           # applied u_{k-1}, ...
        self.r_hist = np.zeros(self.hist_len)
        self.raw = deque(maxlen=max(plant_window, dist_window) + self.hist_len)
        self.g = np.zeros(self.n_g)
        self.a = np.zeros(self.p)
        self.g_used = None
        self.k = 0
        self.u_int = 0.0
        self.design = None
        self.good_design = None
        self.period_sq, self.healthy = 0.0, deque(maxlen=8)
        self.fast_ms = None
        self.hold = 0
        self.xc = None
        self.log = []                                    # (k, event, info)

    # ---------------- identification ---------------- #
    def _prefilter(self, x_now, x_hist, a):
        """A_d(q) at lags 0..n_g: (x^f_k, x^f_{k-1}, ..., x^f_{k-n_g})."""
        seq = np.r_[x_now, x_hist]
        p = len(a)
        return np.array([seq[j] - a @ seq[j + 1:j + 1 + p] for j in range(self.n_g + 1)])

    def _identify(self, y):
        if self.k < self.hist_len:
            return
        yf = self._prefilter(y, self.y_hist, self.a)[0]
        uf = self._prefilter(np.nan, self.u_hist, self.a)[1:]
        rf = self._prefilter(np.nan, self.r_hist, self.a)[1:]
        self.g = self.iv.update(uf, yf, z=rf).copy()
        d_now = y - self.g @ self.u_hist[:self.n_g]
        d_lags = np.array([self.y_hist[i] - self.g @ self.u_hist[i + 1:i + 1 + self.n_g]
                           for i in range(self.p)])
        self.a = self.ar.update(d_lags, d_now).copy()

    def _reanchor(self):
        y, u, r = (np.array(v) for v in zip(*self.raw))
        skip = self.hist_len
        Wg, Wd = self.iv.window + skip, self.ar.window + skip
        yg, ug, rg = y[-Wg:], u[-Wg:], r[-Wg:]
        g, cov = fit_plant_iv(yg, ug, rg, self.n_g, self.p, iters=1, skip=skip,
                              a0=stabilize_ar(self.a, self.max_radius))
        yd, ud = y[-Wd:], u[-Wd:]
        d_hat = yd - lagged(ud, self.n_g) @ g
        a, q = fit_ar(d_hat, self.p, skip, self.max_radius)
        pre = np.r_[1.0, -a]
        yf, uf, rf = (lfilter(pre, [1.0], v) for v in (yg, ug, rg))
        self.iv.load(lagged(uf, self.n_g)[skip:], yf[skip:], lagged(rf, self.n_g)[skip:])
        self.ar.load(lagged(d_hat, self.p)[skip:], d_hat[skip:])
        self.g, self.a = self.iv.theta.copy(), self.ar.theta.copy()
        dc = abs(g.sum())
        rel_std = float(np.sqrt(np.ones(self.n_g) @ cov @ np.ones(self.n_g)) / max(dc, 1e-9))
        return g, a, q, rel_std

    def _redesign(self):
        g, a, q, rel_std = self._reanchor()
        if rel_std <= self.plant_rel_std_max:
            self.g_used = g
        elif self.g_used is None:
            self.log.append((self.k, "waiting", dict(plant_rel_std=rel_std, g=g.tolist())))
            return
        a = stabilize_ar(a, self.max_radius)
        try:
            d = tune_lqg(self.g_used, a, q, self.noise_var, self.dt, self.ms_limit_db,
                         self.gain_range, self.delay_margin)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
            self.log.append((self.k, "rejected", dict(reason=str(exc), g=self.g_used.tolist(),
                                                       plant_rel_std=rel_std)))
            return
        self._activate(d)
        self.log.append((self.k, "accepted", dict(rho=d.rho, r_kalman=d.r_kalman, q=q,
                                                   g=self.g_used.tolist(),
                                                   plant_rel_std=rel_std,
                                                   plant_updated=rel_std <= self.plant_rel_std_max,
                                                   pred_std=d.predicted_output_std())))

    def _activate(self, design):
        self.design = design
        if design is not None:
            self.xc = design.predictor_state(self.y_hist, self.u_hist)

    # ---------------- supervisor ---------------- #
    def _supervise(self, y):
        if not self.supervisor:
            return
        self.fast_ms = y * y if self.fast_ms is None else \
            (1 - self.fast_alpha) * self.fast_ms + self.fast_alpha * y * y
        self.period_sq += y * y
        if self.healthy and self.fast_ms > (self.blowup_factor ** 2) * np.median(self.healthy) \
                and self.design is not None and self.design is not self.good_design:
            self._activate(self.good_design)
            self.hold = self.holdoff
            self.fast_ms = None
            self.log.append((self.k, "reverted", dict(to="previous good design"
                                                       if self.good_design is not None
                                                       else "integrator")))

    def _end_of_period(self):
        ms = self.period_sq / self.redesign_every
        self.period_sq = 0.0
        if not self.supervisor:
            return
        if not self.healthy or ms <= (self.blowup_factor ** 2) * np.median(self.healthy):
            self.healthy.append(ms)
            self.good_design = self.design

    # ---------------- control ---------------- #
    def step(self, y):
        y = float(y)
        self._identify(y)
        self._supervise(y)

        if self.k > 0 and self.k % self.redesign_every == 0:
            self._end_of_period()
            if self.hold > 0:
                self.hold -= 1
            elif len(self.raw) >= self.min_samples + self.hist_len:
                self._redesign()

        r = self.dither_std * self.rng.standard_normal()
        if self.design is None:
            self.u_int = self.warmup_ff * self.u_int + self.warmup_gain * y
            u_ctrl = self.u_int
        else:
            dsg = self.design
            u_ctrl = (dsg.Cc @ self.xc).item() + dsg.Dc.item() * y
            self.u_int = u_ctrl                          # fallback integrator starts bumpless
        u = u_ctrl + r

        self.y_hist = np.r_[y, self.y_hist[:-1]]
        self.u_hist = np.r_[u, self.u_hist[:-1]]
        self.r_hist = np.r_[r, self.r_hist[:-1]]
        if self.design is not None:
            # Kalman predictor update with the command actually applied
            dsg = self.design
            x_f = self.xc + dsg.M[:, 0] * (y - (dsg.C @ self.xc).item())
            self.xc = dsg.A @ x_f + dsg.B[:, 0] * u
        self.raw.append((y, u, r))
        self.k += 1
        return u, r


# ------------------------------------------------------------------ #
# Free theta: output equation on buffer filters, no plant structure  #
# ------------------------------------------------------------------ #
#
# y_k = theta^T zeta_k + e_k,  zeta_k = (y_{k-1} .. y_{k-N}, u_{k-1} .. u_{k-N})
#
# (the nonminimal realization of Bosso et al. with buffer filters, Lambda = 0).
# theta is fitted by sliding-window least squares with all 2N entries free; the
# plant it implies can be far from the true one, because closed-loop data pin it
# down only through the dither and through controller switching. A new design is
# therefore judged on the measured residual (acceptance="residual"): it runs on
# trial for one period, is aborted if the fast residual blows up, and is kept only
# if the period rms does not get worse. Prototype: RAMA ORforRAMA/adaptive_free_theta.py.

def buffer_model(theta):
    """Innovations model of y = theta^T zeta + e on buffer filters:
    zeta+ = A zeta + B u + K e, y = C zeta + e. Returns (A, B, C, K)."""
    theta = np.asarray(theta, float)
    n = theta.size
    N = n // 2
    F = np.zeros((n, n))
    F[1:N, :N - 1] = np.eye(N - 1)
    F[N + 1:, N:n - 1] = np.eye(N - 1)
    g_y, g_u = np.zeros(n), np.zeros(n)
    g_y[0], g_u[N] = 1.0, 1.0
    return F + np.outer(g_y, theta), g_u[:, None], theta[None, :], g_y[:, None]


def stabilize_theta(theta, max_radius=0.9995):
    """Pull the roots of A(q) = 1 - sum theta_y,i q^-i (the denominator shared by
    plant and disturbance) inside max_radius; theta_u is left unchanged. Closed-loop
    data can give an unstable A that the loop itself hides (a direction the
    controller does not excite); no controller then passes the checks on the model.
    Returns (theta, number of roots moved)."""
    theta = np.asarray(theta, float).copy()
    N = theta.size // 2
    roots = np.roots(np.r_[1.0, -theta[:N]])
    big = np.abs(roots) > max_radius
    if np.any(big):
        roots[big] *= max_radius / np.abs(roots[big])
        theta[:N] = -np.real(np.poly(roots))[1:]
    return theta, int(big.sum())


def theta_from_fir_ar(g, a, N=None):
    """theta of the buffer model for y = G(q) u + d, A_d(q) d = e (N >= len(g) + len(a))."""
    g, a = np.asarray(g, float), np.asarray(a, float)
    N = len(g) + len(a) if N is None else int(N)
    c = np.convolve(np.r_[1.0, -a], np.r_[0.0, g])[1:]
    return np.r_[a, np.zeros(N - len(a)), c, np.zeros(N - len(c))]


class ModelLQG:
    """Kalman (measurement noise inflated by s q) + LQR with a rho (u_k - u_{k-1})^2
    penalty on the innovations model x+ = A x + B u + K e, y = C x + e (var e = q).
    The law uses the newest measurement: u_k acts on s_k = (A - L C) x_k + L y_k.
    With s = 0 the predictor gain is K (the buffers are the observer).

    Controller state xbar = (x_hat, u_{k-1}); one frame, with the applied command u:
        u_ctrl = Cc xbar + Dc y,   xbar+ = F_bar xbar + L_bar y + Bd u
    """

    def __init__(self, A, B, C, K, q, rho=0.0, s=0.0):
        n = A.shape[0]
        self.A, self.B, self.C, self.K, self.q = A, B, C, K, float(q)
        self.rho, self.s = float(rho), float(s)
        self.r_kalman = self.s * self.q                  # added measurement-noise variance
        if s > 0:
            W, V, S = q * K @ K.T, np.array([[q * (1.0 + s)]]), q * K
            P = solve_discrete_are(A.T, C.T, W, V, None, S)
            L = (A @ P @ C.T + S) / (C @ P @ C.T + V)
        else:
            L = K
        Ad = np.zeros((n + 1, n + 1))
        Ad[:n, :n] = A
        Bd = np.r_[B[:, 0], 1.0][:, None]
        Q = np.zeros((n + 1, n + 1))
        Q[:n, :n] = C.T @ C
        Q[n, n] = rho
        R = np.array([[max(rho, 1e-9)]])
        Nx = np.zeros((n + 1, 1))
        Nx[n, 0] = -rho
        X = solve_discrete_are(Ad, Bd, Q, R, None, Nx)
        Wg = np.linalg.inv(R + Bd.T @ X @ Bd)
        K_s, K_x = Wg @ Bd.T @ X, Wg @ Nx.T
        self.F_bar = np.zeros((n + 1, n + 1))
        self.F_bar[:n, :n] = A - L @ C
        self.L_bar = np.r_[L[:, 0], 0.0][:, None]
        self.Bd = Bd
        self.Cc = -(K_s @ self.F_bar + K_x)
        self.Dc = -(K_s @ self.L_bar)
        self.Ac = self.F_bar + Bd @ self.Cc
        self.Bc = self.L_bar + Bd @ self.Dc
        self.g = None                                    # implied taps, set by the owner

    @property
    def ctrl(self):
        return self.Ac, self.Bc, self.Cc, self.Dc

    @property
    def identified_plant(self):
        return self.A, self.B, self.C

    def predicted_output_std(self):
        """Stationary std of y predicted by the model in closed loop."""
        A, B, C, K = self.A, self.B, self.C, self.K
        Ac, Bc, Cc, Dc = self.ctrl
        Acl = np.block([[A + B @ Dc @ C, B @ Cc], [Bc @ C, Ac]])
        if np.abs(np.linalg.eigvals(Acl)).max() >= 1.0:
            return np.inf
        Bcl = np.vstack([K + B @ Dc, Bc])
        Ccl = np.hstack([C, np.zeros((1, Ac.shape[0]))])
        Sigma = solve_discrete_lyapunov(Acl, self.q * Bcl @ Bcl.T)
        return float(np.sqrt((Ccl @ Sigma @ Ccl.T).item() + self.q))


def tune_model_lqg(theta, q, dt, ms_limit_db=6.0, gain_range=(0.5, 1.5), delay_margin=0.0,
                   s_grid=(0.0, 0.1, 1.0, 10.0, 100.0, 1e3), rho_grid=(0.0, 1.0, 10.0)):
    """(s, rho) of ModelLQG on buffer_model(theta) with the smallest predicted std,
    subject to Ms <= limit and stability for gain x gain_range (and +delay_margin
    frames), all on the plant implied by theta."""
    A, B, C, K = buffer_model(theta)
    plant = (A, B, C)
    plants = [plant] + ([with_extra_delay(plant, delay_margin)] if delay_margin > 0 else [])
    gains = np.linspace(*gain_range, 11)
    best = None
    for s in s_grid:
        for rho in rho_grid:
            try:
                d = ModelLQG(A, B, C, K, q, rho, s)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if not all(np.abs(np.linalg.eigvals(closed_loop_matrix(pl, d.ctrl, k))).max() < 1
                       for pl in plants for k in gains):
                continue
            if peak_sensitivity_db(plant, d.ctrl, dt) > ms_limit_db:
                continue
            std = d.predicted_output_std()
            if best is None or std < best[0]:
                best = (std, d)
    if best is None:
        raise RuntimeError("no (s, rho) in the grid meets the constraints on the identified plant")
    return best[1]


def implied_taps(theta, n):
    """First n samples of the impulse response of the plant implied by theta."""
    A, B, C, _ = buffer_model(theta)
    x, out = B[:, 0].copy(), []
    for _ in range(n):
        out.append(float(C[0] @ x))
        x = A @ x
    return np.array(out)


class AdaptiveModeFreeTheta:
    """One modal channel with a free theta. Same interface as AdaptiveModeLQG:
    step(y_k) -> (u_k, r_k), `log`, `design`, `name`.

    Every frame: sliding-window least squares on (zeta_k, y_k); control with the
    active ModelLQG; dither r. Every `redesign_every` frames (once `min_samples`
    are in the window): tune_model_lqg on the current theta, then
      acceptance="model"     switch to the new design at once
      acceptance="residual"  run it on trial for one period; abort as soon as the
                             fast residual rms exceeds abort_ratio times the rms of
                             the last clean period of the current controller; keep
                             it at the end of the period only if its rms is at most
                             accept_ratio times that reference. After an abort or a
                             rejection the current controller runs one clean period.
    Before each design the shared denominator of theta is stabilized
    (stabilize_theta, max_radius), as the AR model of the structured method.
    Dither: dither_std until `switch_designs` designs are accepted, then
    dither_after (back to dither_std after a supervisor revert).
    Before the first accepted design an integrator runs, or the IIR filter
    (warmup_num, warmup_den) when given, in the IirFilterData convention (oldest
    coefficient first: num[-1] multiplies y_k, den[-1] multiplies u_k). Its state
    follows the applied commands (dither excluded) while a design runs, so a
    revert to it is bumpless, as for the integrator.
    """

    def __init__(self, dt, N=11, window=4000, min_samples=None, dither_std=5.0,
                 dither_after=None, switch_designs=2, redesign_every=250, ms_limit_db=6.0,
                 gain_range=(0.5, 1.5), delay_margin=0.5, warmup_gain=0.4, warmup_ff=1.0,
                 warmup_num=None, warmup_den=None,
                 acceptance="residual", accept_ratio=1.05, abort_ratio=1.5, hard_abort_ratio=4.0,
                 abort_min_frames=20, n_taps_log=3, max_radius=0.9995, supervisor=True, blowup_factor=2.0,
                 fast_window=50, holdoff=4, rng=None, name=""):
        if acceptance not in ("model", "residual"):
            raise ValueError(f"acceptance must be 'model' or 'residual', got {acceptance!r}")
        self.dt, self.N = float(dt), int(N)
        self.rls = SlidingWindowRLS(2 * self.N, window)
        self.min_samples = int(window if min_samples is None else min_samples)
        self.dither_std = float(dither_std)
        self.dither_after = self.dither_std if dither_after is None else float(dither_after)
        self.switch_designs = int(switch_designs)
        self.dither_level = self.dither_std
        self.redesign_every = int(redesign_every)
        self.ms_limit_db, self.gain_range = float(ms_limit_db), tuple(gain_range)
        self.delay_margin, self.warmup_gain = float(delay_margin), float(warmup_gain)
        self.warmup_ff = float(warmup_ff)
        self.iir = None
        if (warmup_num is None) != (warmup_den is None):
            raise ValueError("warmup_num and warmup_den must be given together")
        if warmup_num is not None:
            num = np.atleast_1d(np.asarray(warmup_num, dtype=float))
            den = np.atleast_1d(np.asarray(warmup_den, dtype=float))
            if den[-1] == 0:
                raise ValueError("warmup_den[-1] (the u_k coefficient) must be nonzero")
            self.iir = dict(b=num / den[-1], a=den[:-1] / den[-1],
                            y=np.zeros(num.size - 1), u=np.zeros(den.size - 1))  # oldest first
        self.acceptance = acceptance
        self.accept_ratio, self.abort_ratio = float(accept_ratio), float(abort_ratio)
        self.hard_abort_ratio = float(hard_abort_ratio)
        self.abort_min_frames, self.n_taps_log = int(abort_min_frames), int(n_taps_log)
        self.max_radius = float(max_radius)
        self.supervisor, self.blowup_factor, self.holdoff = supervisor, float(blowup_factor), int(holdoff)
        self.fast_alpha = 1.0 / fast_window
        self.rng = np.random.default_rng() if rng is None else rng
        self.name = name

        self.y_hist = np.zeros(self.N)                   # y_{k-1}, y_{k-2}, ...
        self.u_hist = np.zeros(self.N)                   # applied u_{k-1}, ...
        self.k = 0
        self.u_int = 0.0
        self.design = None
        self.good_design = None
        self.n_accepted = 0
        self.xc = None
        self.period_sq, self.healthy = 0.0, deque(maxlen=8)
        self.fast_ms = None
        self.hold = 0
        self.log = []                                    # (k, event, info)
        self.trial = None
        self.incumbent_ms = None
        self.clean = True
        self.p_sq = 0.0
        self.trial_fast = 0.0

    # ---------------- design ---------------- #
    def _redesign(self):
        theta, moved = stabilize_theta(self.rls.theta, self.max_radius)
        q = self.rls.residual_variance(theta)
        taps = implied_taps(theta, self.n_taps_log)
        try:
            d = tune_model_lqg(theta, q, self.dt, self.ms_limit_db, self.gain_range, self.delay_margin)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
            self.log.append((self.k, "rejected", dict(reason=str(exc), g=taps.tolist())))
            return
        d.g = taps
        info = dict(rho=d.rho, r_kalman=d.r_kalman, s=d.s, q=q, g=taps.tolist(),
                    pred_std=d.predicted_output_std(), dither=self.dither_level,
                    roots_stabilized=moved)
        if self.acceptance == "residual":
            if self.incumbent_ms is None:
                self.log.append((self.k, "no reference", info))
                return
            self.trial = dict(prev=self.design, cand=d, start=self.k, info=info)
            self._activate(d)
            self.clean = False
            self.trial_fast = self.incumbent_ms
            self.log.append((self.k, "trial", info))
            return
        self._accept(d, info)

    def _accept(self, d, info):
        if self.design is not d:
            self._activate(d)
        self.n_accepted += 1
        if self.n_accepted >= self.switch_designs:
            self.dither_level = self.dither_after
        self.log.append((self.k, "accepted", info))

    def _activate(self, design):
        self.design = design
        if design is not None:
            self.xc = np.r_[self.y_hist, self.u_hist, self.u_hist[0]]

    def _finish_trial(self, ms):
        t, self.trial = self.trial, None
        ratio = float(np.sqrt(ms / self.incumbent_ms))
        if ratio <= self.accept_ratio:
            self.incumbent_ms = ms
            self._accept(t["cand"], dict(t["info"], trial_rms_ratio=ratio))
        else:
            self._activate(t["prev"])
            self.hold = max(self.hold, 1)
            self.log.append((self.k, "trial rejected", dict(t["info"], trial_rms_ratio=ratio)))

    def _watch_trial(self, y):
        self.trial_fast = (1 - self.fast_alpha) * self.trial_fast + self.fast_alpha * y * y
        # the averaged criterion needs abort_min_frames to ignore single noisy samples, but an
        # unstable loop grows by ~7x in 5 frames at 1 kHz, so one sample far above the reference
        # aborts at once: a false positive only costs one period on the previous design
        hard = y * y > self.hard_abort_ratio ** 2 * self.incumbent_ms
        if hard or (self.k - self.trial["start"] >= self.abort_min_frames
                    and self.trial_fast > self.abort_ratio ** 2 * self.incumbent_ms):
            t, self.trial = self.trial, None
            self._activate(t["prev"])
            self.hold = max(self.hold, 1)
            self.clean = False
            self.log.append((self.k, "trial aborted",
                             dict(t["info"], after=self.k - t["start"], instantaneous=bool(hard))))

    # ---------------- supervisor ---------------- #
    def _supervise(self, y):
        if not self.supervisor:
            return
        self.fast_ms = y * y if self.fast_ms is None else \
            (1 - self.fast_alpha) * self.fast_ms + self.fast_alpha * y * y
        self.period_sq += y * y
        if self.trial is None and self.healthy \
                and self.fast_ms > (self.blowup_factor ** 2) * np.median(self.healthy) \
                and self.design is not None and self.design is not self.good_design:
            self._activate(self.good_design)
            self.hold = self.holdoff
            self.fast_ms = None
            self.clean = False
            self.dither_level = self.dither_std
            self.log.append((self.k, "reverted", dict(to="previous good design"
                                                       if self.good_design is not None
                                                       else "integrator")))

    def _end_of_period(self):
        ms = self.period_sq / self.redesign_every
        self.period_sq = 0.0
        if not self.supervisor:
            return
        if not self.healthy or ms <= (self.blowup_factor ** 2) * np.median(self.healthy):
            self.healthy.append(ms)
            self.good_design = self.design

    # ---------------- control ---------------- #
    def step(self, y):
        y = float(y)
        if self.k >= self.N:
            self.rls.update(np.r_[self.y_hist, self.u_hist], y)
        self._supervise(y)
        if self.trial is not None:
            self._watch_trial(y)

        if self.k > 0 and self.k % self.redesign_every == 0:
            ms, self.p_sq = self.p_sq / self.redesign_every, 0.0
            if self.trial is not None:
                self._finish_trial(ms)
            elif self.clean:
                self.incumbent_ms = ms
            self.clean = True
            self._end_of_period()                        # after the trial decision
            if self.hold > 0:
                self.hold -= 1
            elif len(self.rls.buf) >= self.min_samples:
                self._redesign()

        r = self.dither_level * self.rng.standard_normal()
        if self.design is None and self.iir is not None:
            f = self.iir
            u_ctrl = f["b"][:-1] @ f["y"] + f["b"][-1] * y - f["a"] @ f["u"]
        elif self.design is None:
            self.u_int = self.warmup_ff * self.u_int + self.warmup_gain * y
            u_ctrl = self.u_int
        else:
            dsg = self.design
            u_ctrl = (dsg.Cc @ self.xc).item() + dsg.Dc.item() * y
            self.u_int = u_ctrl                          # fallback integrator starts bumpless
        if self.iir is not None:                         # and so does the fallback IIR
            f = self.iir
            if f["y"].size:
                f["y"] = np.r_[f["y"][1:], y]
            if f["u"].size:
                f["u"] = np.r_[f["u"][1:], u_ctrl]
        u = u_ctrl + r

        if self.design is not None:
            dsg = self.design
            self.xc = dsg.F_bar @ self.xc + dsg.L_bar[:, 0] * y + dsg.Bd[:, 0] * u
        self.y_hist = np.r_[y, self.y_hist[:-1]]
        self.u_hist = np.r_[u, self.u_hist[:-1]]
        self.p_sq += y * y
        self.k += 1
        return u, r
