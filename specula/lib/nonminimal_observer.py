"""Plant + disturbance observer in nonminimal (Bosso-Borghesi) form, free theta.

One modal channel, the loop convention of specula.lib.adaptive_lqg:

    y_k = sum_{i=1..n_g} g_i u_{k-i} + d_k + n_k        (RAMA: g ~ (0, -OG, 0))

Filters. (Lambda, l) of order N, controllable, Schur, chosen by the user:

    xi+    = Lambda xi    + l y
    omega+ = Lambda omega + l u,        zeta = (xi, omega) in R^{2N}

Output equation. For any system of order N and any observer gain L with
char(A - L C) = char(Lambda) there are Pi_y, Pi_u (A_o Pi = Pi Lambda, Pi l = L
resp. B) with chi = Pi zeta + A_o^k e_0, so

    y_k = theta^T zeta_k + eps_k,     theta^T = C Pi,

linear in theta. theta is fitted by least squares on closed-loop data (a dither
on u makes the plant part identifiable), all 2N entries free. The observer is
the Luenberger predictor with poles spec(Lambda), whatever theta is:

    y_hat_{k|k-1} = theta^T zeta_k

Speed. spec(Lambda) is the observer spectrum: poles near 0 follow the
measurement (deadbeat at 0: the buffers of AdaptiveModeFreeTheta), poles near 1
make a slow observer that rejects the measurement noise and relies on the
model. Default: n_g poles at 0 for the plant part (the command history is
known exactly, nothing to reject) and N - n_g at `slow_pole`.

Realization. From theta, n_y / lambda = theta_y^T (zI - Lambda)^-1 l and
n_u / lambda = theta_u^T (zI - Lambda)^-1 l, with lambda = char(Lambda); the
model is A(z) y = B(z) u + lambda(z) eps with A = lambda - n_y, B = n_u. In
observer canonical form (C = e1^T) the gain is L = coefficients of n_y and
A - L C has characteristic polynomial lambda: `realization` returns it, and its
Luenberger observer reproduces theta^T zeta exactly.

Split. Plant taps g = the first n_g samples of the impulse response of B/A.
The first n_g states of omega are the command buffer x_k = (u_{k-1} .. u_{k-n_g})
(poles at 0 first in the cascade), so

    d_hat_{k|k-1} = y_hat_{k|k-1} - g^T x_k      observer output (disturbance)
    d_hat_{k|k}   = y_k - g^T x_k                 reconstruction, no model

Fit with the dither as instrument (fit="iv", default). Under one fixed controller the
free theta is known only up to theta + c v (v^T zeta = 0 is the control law); least
squares does not pin c down, so the plant gain -sum(g) it implies wanders with the
data window. Instrumenting all 2N entries with the dither fails (the dither is weakly
correlated with the past-y filter states). The structure of the observer gain
L = (0, L_w) is used instead: chi = (x, w), x the command buffer (deadbeat, known), w
the disturbance, observed through its own filters (Lambda_d, l_d) = the slow part of
spec(Lambda), driven by d_meas = y - g^T x:

    theta = (g, theta_d),    y_k = g^T x_k + theta_d^T xi_d,k + eps_k
  1. g: instrumental variables, the dither r as instrument, on the equation prefiltered
     by the innovation filter of a deadbeat disturbance observer (AR(p) whitening of the
     turbulence; fit_plant_iv). Independent of the observer speed: with a slow
     prefilter the equation error is not white and the estimate degrades.
  2. theta_d: least squares on d_meas_k = theta_d^T xi_d,k + eps_k, xi_d from (Lambda_d, l_d).
Observer output: d_hat_{k|k-1} = theta_d^T xi_d,k, y_hat_{k|k-1} = g^T x_k + d_hat_{k|k-1}.

Estimates at frame k (step returns them in this order):
    d_hat_{k|k-1}   one-step prediction, data up to k-1 (the observer output)
    d_meas_k        reconstruction y_k - g^T x_k (measurement, no disturbance model)
    y_hat_{k|k-1}   predicted measurement
    d_hat_{k|k}     Kalman update: d_hat_{k|k-1} + kappa (d_meas_k - d_hat_{k|k-1}). With
                    d_meas = d + n, n white of variance R, and innovation variance q (fit
                    window), the steady-state gain is kappa = 1 - R/q (MIMO: K = I - R Q^-1).
                    R is not visible in the innovations; it is estimated as the lag-0 excess
                    of the autocovariance of d_meas over its extrapolation from lags 1-4
                    (white_noise_var; the turbulence is smooth, white noise lives at lag 0).
    d_hat_{k+2|k}   two-step prediction, data up to k: the one-step predictor iterated with
                    the next measurement replaced by its prediction (zero-mean innovation),
                    exact for the model theta implies. The horizon of a 2-frame loop delay.
                    iv / MIMO only (fit="ls" would need the future command): NaN otherwise.

Units: d is the disturbance as the sensor sees it (OG x the open-loop coefficient,
OG = -sum(g)).
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.signal import ss2tf

from specula.lib.adaptive_lqg import fit_plant_iv, lagged


# ------------------------------------------------------------------ #
# Filters                                                            #
# ------------------------------------------------------------------ #
def _conj_pairs(poles, tol=1e-9):
    """Real poles (sorted) and one representative (Im > 0) per complex pair."""
    poles = np.asarray(poles, complex)
    real = np.sort(poles[np.abs(poles.imag) <= tol].real)
    cplx = poles[poles.imag > tol]
    if np.sum(poles.imag < -tol) != cplx.size:
        raise ValueError("complex poles must come in conjugate pairs")
    return real, cplx[np.argsort(np.abs(cplx))]


def cascade_filter(poles):
    """(Lambda, l): chain of first-order sections (gain 1 - p, unit DC gain; p = 0 is
    a pure shift) and rotation sections, each driven by the first state of the previous
    one. Real poles in increasing order, so the zeros come first: with n zeros the first
    n states are the last n inputs."""
    real, cplx = _conj_pairs(poles)
    if np.any(np.abs(np.r_[real, np.abs(cplx)]) >= 1.0):
        raise ValueError("filter poles must be inside the unit circle")
    blocks, gains = [], []
    for p in real:
        blocks.append(np.array([[p]]))
        gains.append(1.0 - p)
    for p in cplx:
        blocks.append(np.array([[p.real, -p.imag], [p.imag, p.real]]))
        gains.append(abs(1.0 - p))
    Lam = block_diag(*blocks)
    first = np.cumsum([0] + [b.shape[0] for b in blocks[:-1]])
    ell = np.zeros(Lam.shape[0])
    ell[0] = gains[0]
    for i in range(1, len(blocks)):
        Lam[first[i], first[i - 1]] = gains[i]
    return Lam, ell


def default_poles(N, n_g, slow_pole):
    """n_g poles at 0 (command history), N - n_g at slow_pole (disturbance)."""
    return np.r_[np.zeros(n_g), np.full(N - n_g, float(slow_pole))]


def run_filters(Lam, ell, y, u):
    """zeta_k = (xi_k, omega_k) for every k (row k uses data up to k-1)."""
    N = Lam.shape[0]
    T = len(y)
    xi, om = np.zeros(N), np.zeros(N)
    Z = np.empty((T, 2 * N))
    for k in range(T):
        Z[k, :N], Z[k, N:] = xi, om
        xi = Lam @ xi + ell * y[k]
        om = Lam @ om + ell * u[k]
    return Z


def solve_scaled(ZZ, Zy, Zs, n, ridge=1e-9):
    """Least squares from the sums Z^T Z, Z^T y, sum Z over n samples, columns scaled
    to unit std (the states of a slow filter are nearly collinear)."""
    s = np.maximum(np.sqrt(np.maximum(np.diag(ZZ) / n - (Zs / n) ** 2, 0.0)), 1e-12)
    th = np.linalg.solve(ZZ / np.outer(s, s) + ridge * n * np.eye(len(s)), Zy / s)
    return th / s


def fit_theta(Z, y, ridge=1e-9):
    """Least squares y = Z theta (batch form of ModeObserver's fit)."""
    y = np.asarray(y, float)
    return solve_scaled(Z.T @ Z, Z.T @ y, Z.sum(axis=0), len(y), ridge)


def white_noise_var(x, lags=(1, 2, 3, 4)):
    """Variance of the white part of x: autocovariance at lag 0 minus its quadratic
    extrapolation from `lags` (the classic AO noise estimate from temporal correlation).
    Clipped to [0, var(x)]."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    c0 = x @ x / T
    c = np.array([x[l:] @ x[:T - l] / T for l in lags])
    coef = np.polyfit(np.asarray(lags, float), c, 2)
    return float(np.clip(c0 - np.polyval(coef, 0.0), 0.0, c0))


def kalman_gain(innov_var, noise_var):
    """Steady-state update gain 1 - R/q for d_meas = d + white noise (clipped to [0, 1])."""
    return float(np.clip(1.0 - noise_var / innov_var, 0.0, 1.0))


# ------------------------------------------------------------------ #
# Model from theta                                                   #
# ------------------------------------------------------------------ #
def realization(Lam, ell, theta):
    """(A, B, C, L) of the order-N model implied by theta, observer canonical form:
    A(z) = lambda(z) - n_y(z), B(z) = n_u(z), L = coefficients of n_y,
    char(A - L C) = char(Lambda)."""
    N = Lam.shape[0]
    theta = np.asarray(theta, float)
    lam = np.real(np.poly(Lam))                            # [1, l1 .. lN]

    def num(c):
        n, _ = ss2tf(Lam, ell[:, None], c[None, :], np.zeros((1, 1)))
        return np.real(n[0])[1:]                           # strictly proper: z^(N-1) .. z^0

    n_y, n_u = num(theta[:N]), num(theta[N:])
    A = np.eye(N, k=1)
    A[:, 0] = -(lam[1:] - n_y)                             # A(z) = z^N + a1 z^(N-1) + ...
    C = np.zeros((1, N))
    C[0, 0] = 1.0
    return A, n_u[:, None], C, n_y[:, None]


def implied_taps(Lam, ell, theta, n):
    """First n samples (lags 1..n) of the impulse response of the plant B/A."""
    A, B, C, _ = realization(Lam, ell, theta)
    x, out = B[:, 0].copy(), []
    for _ in range(n):
        out.append(float(C[0] @ x))
        x = A @ x
    return np.array(out)


def luenberger(A, B, C, L, y, u):
    """One-step predictor chi+ = A chi + B u + L (y - C chi); returns C chi_k for every k."""
    chi = np.zeros(A.shape[0])
    out = np.empty(len(y))
    for k in range(len(y)):
        out[k] = C[0] @ chi
        chi = A @ chi + B[:, 0] * u[k] + L[:, 0] * (y[k] - out[k])
    return out


# ------------------------------------------------------------------ #
# Online observer, one mode                                          #
# ------------------------------------------------------------------ #
class ModeObserver:
    """Filters run from the start; theta fitted on frames [fit_start, fit_start + fit_frames),
    then frozen. fit="iv": plant taps by instrumental variables on the dither, disturbance
    filters (Lambda_d, l_d) = the non-zero poles of spec(Lambda); fit="ls": free theta,
    least squares, filters (Lambda, l) on y and u.

    step(y_k, u_k, r_k) -> (d_hat_{k|k-1}, d_hat_{k|k}, y_hat_{k|k-1}); NaN before theta.
    u_k is the command applied at frame k (dither included), r_k the dither in it, y_k the
    measurement.

    cold_start: the filters (xi, omega, xi_d) are reset to zero when theta is frozen, so the
    output shows the observer transient e_k = A_o^k e_0, spec(A_o) = spec(Lambda): n_g frames
    for the command buffer, then the disturbance filters (deadbeat: p frames; pole 0.9: about
    150 frames to 1 %). Otherwise (default) the filters have run since frame 0 and the output
    is converged when it appears. theta never enters the filter dynamics, so a new theta
    needs no restart.
    """

    def __init__(self, N=11, n_g=3, slow_pole=0.9, poles=None, fit_start=500,
                 fit_frames=6000, fit="iv", iv_iters=3, cold_start=False):
        if fit not in ("iv", "ls"):
            raise ValueError(f"fit must be 'iv' or 'ls', got {fit!r}")
        self.fit = fit
        self.N, self.n_g = int(N), int(n_g)
        self.poles = default_poles(self.N, self.n_g, slow_pole) if poles is None \
            else np.asarray(poles)
        if self.poles.size != self.N:
            raise ValueError(f"need {self.N} filter poles, got {self.poles.size}")
        zero = np.abs(self.poles) < 1e-12
        if np.sum(zero) < self.n_g:
            raise ValueError("need at least n_g poles at 0: the first n_g states of omega "
                             "are then the command buffer")
        self.Lam, self.ell = cascade_filter(self.poles)
        # disturbance filters: spec(Lambda) without n_g of its zeros
        self.p = self.N - self.n_g
        self.Lam_d, self.ell_d = cascade_filter(np.r_[np.zeros(np.sum(zero) - self.n_g),
                                                      self.poles[~zero]])
        self.iv_iters = int(iv_iters)
        self.cold_start = bool(cold_start)
        self.fit_start, self.fit_frames = int(fit_start), int(fit_frames)
        self.xi, self.om = np.zeros(self.N), np.zeros(self.N)
        self.xi_d = np.zeros(self.p)
        self.hist = []                                   # (y, u, r) of the fit window
        self.theta = None                                # ls: 2N free; iv: (g, theta_d)
        self.g = None
        self.og_std = np.nan
        self.q = np.nan
        self.noise_var = np.nan
        self.kappa = np.nan
        self.k = 0
        self.log = []

    def _build(self):
        Y, U, R = (np.array(c) for c in zip(*self.hist))
        # the filters are rerun over the window from zero: skip their transient. It matters:
        # along the closed-loop ambiguity the least-squares cost is so flat that the 11
        # transient rows of a deadbeat filter move the tip gain from 0.80 to 0.56 on RAMA
        skip = min(len(Y) // 10, 500)
        if self.fit == "ls":
            Z = run_filters(self.Lam, self.ell, Y, U)
            self.theta = fit_theta(Z[skip:], Y[skip:])
            self.q = float(np.var(Y[skip:] - Z[skip:] @ self.theta))
            self.g = implied_taps(self.Lam, self.ell, self.theta, self.n_g)
            d_meas = Y - lagged(U, self.n_g) @ self.g
        else:
            self.g, cov = fit_plant_iv(Y, U, R, self.n_g, p=self.p, iters=self.iv_iters)
            self.og_std = float(np.sqrt(np.sum(cov)))
            d_meas = Y - lagged(U, self.n_g) @ self.g
            Zd = run_filters(self.Lam_d, self.ell_d, d_meas, np.zeros_like(d_meas))[:, :self.p]
            theta_d = fit_theta(Zd[skip:], d_meas[skip:])
            self.q = float(np.var(d_meas[skip:] - Zd[skip:] @ theta_d))
            self.theta = np.r_[self.g, theta_d]
            # disturbance filter state after the window (row k of Zd is before frame k)
            self.xi_d = self.Lam_d @ Zd[-1] + self.ell_d * d_meas[-1]
        self.noise_var = white_noise_var(d_meas[skip:])
        self.kappa = kalman_gain(self.q, self.noise_var)
        if self.cold_start:
            self.xi[:], self.om[:], self.xi_d[:] = 0.0, 0.0, 0.0
        self.log.append((self.k, "theta", dict(fit=self.fit, g=self.g.tolist(), og=float(-self.g.sum()),
                                               og_std=self.og_std, q=self.q, noise_var=self.noise_var,
                                               kappa=self.kappa,
                                               poles=self.poles.tolist(), theta=self.theta.tolist())))

    def step(self, y, u, r=0.0):
        y, u, r = float(y), float(u), float(r)
        d_pred = d_filt = y_pred = d_kal = d_pred2 = np.nan
        if self.theta is not None:
            gx = float(self.g @ self.om[:self.n_g])      # x_k = (u_{k-1} .. u_{k-n_g})
            d_filt = y - gx
            if self.fit == "ls":
                y_pred = float(self.theta @ np.r_[self.xi, self.om])
                d_pred = y_pred - gx
            else:
                th_d = self.theta[self.n_g:]
                d_pred = float(th_d @ self.xi_d)
                y_pred = gx + d_pred
                self.xi_d = self.Lam_d @ self.xi_d + self.ell_d * d_filt
                d_next = float(th_d @ self.xi_d)          # d_hat_{k+1|k}
                d_pred2 = float(th_d @ (self.Lam_d @ self.xi_d + self.ell_d * d_next))
            d_kal = d_pred + self.kappa * (d_filt - d_pred)
        if self.fit_start <= self.k < self.fit_start + self.fit_frames:
            self.hist.append((y, u, r))
            if len(self.hist) == self.fit_frames:
                self._build()
                self.hist = []
        self.xi = self.Lam @ self.xi + self.ell * y
        self.om = self.Lam @ self.om + self.ell * u
        self.k += 1
        return d_pred, d_filt, y_pred, d_kal, d_pred2


# ------------------------------------------------------------------ #
# Online observer, several modes jointly (vector disturbance model)  #
# ------------------------------------------------------------------ #
class MimoObserver:
    """m modes jointly: diagonal plant, vector disturbance model (the structure of
    specula.lib.adaptive_lqg_var: modal turbulence is correlated across modes at nonzero lags
    by frozen flow, so the other modes' past helps predict a mode).

      plant         per-mode FIR taps g_i by instrumental variables on the dither r_i
                    (fit_plant_diag), as ModeObserver with fit="iv"
      disturbance   per-mode filters (Lambda_d, l_d) driven by d_meas_i = y_i - g_i^T x_i;
                    xi_d = (xi_d,1 .. xi_d,m) in R^{m p}; one row of Theta_d (m, m p) per mode,
                    least squares: d_meas_i,k = Theta_d[i] xi_d,k + eps_i,k
      output        d_hat_{k|k-1} = Theta_d xi_d,k

    With m = 1 this is ModeObserver(fit="iv"). Same arguments as ModeObserver, plus m.
    step(y_k, u_k, r_k) with vectors of length m -> three vectors of length m (NaN before theta).
    """

    def __init__(self, m, N=11, n_g=3, slow_pole=0.9, poles=None, fit_start=500,
                 fit_frames=6000, iv_iters=3, cold_start=False):
        self.m, self.N, self.n_g = int(m), int(N), int(n_g)
        self.poles = default_poles(self.N, self.n_g, slow_pole) if poles is None \
            else np.asarray(poles)
        zero = np.abs(self.poles) < 1e-12
        if self.poles.size != self.N or np.sum(zero) < self.n_g:
            raise ValueError("need N filter poles, at least n_g of them at 0")
        self.p = self.N - self.n_g
        self.Lam_d, self.ell_d = cascade_filter(np.r_[np.zeros(np.sum(zero) - self.n_g),
                                                      self.poles[~zero]])
        self.iv_iters, self.cold_start = int(iv_iters), bool(cold_start)
        self.fit_start, self.fit_frames = int(fit_start), int(fit_frames)
        self.u_buf = np.zeros((self.m, self.n_g))         # (u_{k-1} .. u_{k-n_g}) per mode
        self.xi_d = np.zeros((self.m, self.p))
        self.hist = []
        self.G = None                                    # (m, n_g)
        self.Theta_d = None                              # (m, m p)
        self.og_std = np.full(self.m, np.nan)
        self.q = np.full(self.m, np.nan)
        self.noise_var = np.full(self.m, np.nan)
        self.K = None                                    # (m, m) Kalman update gain
        self.k = 0
        self.log = []

    def _build(self):
        H = np.array(self.hist)                          # (T, 3, m)
        Y, U, R = H[:, 0], H[:, 1], H[:, 2]
        self.G = np.empty((self.m, self.n_g))
        for i in range(self.m):
            self.G[i], cov = fit_plant_iv(Y[:, i], U[:, i], R[:, i], self.n_g, p=self.p,
                                          iters=self.iv_iters)
            self.og_std[i] = float(np.sqrt(np.sum(cov)))
        D = Y - np.column_stack([lagged(U[:, i], self.n_g) @ self.G[i] for i in range(self.m)])
        Zs = [run_filters(self.Lam_d, self.ell_d, D[:, i], np.zeros(len(D)))[:, :self.p]
              for i in range(self.m)]
        Z = np.hstack(Zs)                                # (T, m p), mode-major
        skip = min(len(Y) // 10, 500)
        self.Theta_d = np.array([fit_theta(Z[skip:], D[skip:, i]) for i in range(self.m)])
        innov = D[skip:] - Z[skip:] @ self.Theta_d.T
        self.q = np.var(innov, axis=0)
        Q = np.cov(innov.T, bias=True).reshape(self.m, self.m)
        self.noise_var = np.array([white_noise_var(D[skip:, i]) for i in range(self.m)])
        Rn = np.diag(np.minimum(self.noise_var, 0.999 * np.linalg.eigvalsh(Q).min()))
        self.K = np.eye(self.m) - Rn @ np.linalg.inv(Q)   # I - R Q^-1, R shrunk so Q - R > 0
        self.xi_d = np.array([self.Lam_d @ Zs[i][-1] + self.ell_d * D[-1, i] for i in range(self.m)])
        if self.cold_start:
            self.u_buf[:], self.xi_d[:] = 0.0, 0.0
        self.log.append((self.k, "theta", dict(fit="iv", mimo=True, g=self.G.tolist(),
                                               og=(-self.G.sum(axis=1)).tolist(),
                                               og_std=self.og_std.tolist(), q=self.q.tolist(),
                                               noise_var=self.noise_var.tolist(), K=self.K.tolist(),
                                               poles=self.poles.tolist(),
                                               theta_d=self.Theta_d.tolist())))

    def step(self, y, u, r=None):
        y, u = np.asarray(y, float), np.asarray(u, float)
        r = np.zeros(self.m) if r is None else np.asarray(r, float)
        nan = np.full(self.m, np.nan)
        d_pred, d_filt, y_pred, d_kal, d_pred2 = (nan.copy() for _ in range(5))
        if self.Theta_d is not None:
            gx = np.sum(self.G * self.u_buf, axis=1)
            d_filt = y - gx
            d_pred = self.Theta_d @ self.xi_d.ravel()
            y_pred = gx + d_pred
            d_kal = d_pred + self.K @ (d_filt - d_pred)
            self.xi_d = self.xi_d @ self.Lam_d.T + np.outer(d_filt, self.ell_d)
            d_next = self.Theta_d @ self.xi_d.ravel()     # d_hat_{k+1|k}
            d_pred2 = self.Theta_d @ (self.xi_d @ self.Lam_d.T + np.outer(d_next, self.ell_d)).ravel()
        if self.fit_start <= self.k < self.fit_start + self.fit_frames:
            self.hist.append((y, u, r))
            if len(self.hist) == self.fit_frames:
                self._build()
                self.hist = []
        self.u_buf = np.column_stack([u, self.u_buf[:, :-1]])
        self.k += 1
        return d_pred, d_filt, y_pred, d_kal, d_pred2
