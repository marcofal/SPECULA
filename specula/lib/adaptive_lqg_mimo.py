"""MIMO adaptive LQG on a free Theta: m modes identified and controlled jointly.

The MIMO version of the free-theta method of adaptive_lqg (AdaptiveModeFreeTheta).
With misregistration the command of one mode also moves others (the modal
interaction seen by the WFS is not the identity any more); every SISO loop then
contains the others, and checks made on one mode at a time do not cover the coupled
loop. Here the same algorithm runs on the vector y in R^m.

Buffer filters, lag-major:

    zeta_k = (y_{k-1}, ..., y_{k-N}, u_{k-1}, ..., u_{k-N}) in R^{2Nm}

Output equation, Theta in R^{2Nm x m}:

    y_k = Theta^T zeta_k + e_k,   cov(e) = Sigma

i.e. A(q) y = B(q) u + e: the plant implied by Theta is A^-1 B, a full m x m
transfer matrix, cross-talk included. Innovations model:

    zeta+ = A_Theta zeta + G_u u + G_y e,   A_Theta = F + G_y Theta^T,   y = Theta^T zeta + e

Structure of Theta (`structure`):
  'full'     every entry free, 2Nm parameters per mode
  'diag_ar'  A(q) diagonal (each mode's own AR), B(q) full: mode i regresses on its
             own past and on all the commands, N (m + 1) parameters. Under
             decentralized integrators the other modes' integrator laws are then out of
             its regressor, and the closed-loop ambiguity is one direction per mode as
             in SISO. Fits nearly uncorrelated modal disturbances.

Design (MimoModelLQG): Kalman with W = G_y Sigma G_y^T, V = (1 + s) Sigma,
S = G_y Sigma (s = 0: the buffers are the observer); LQR on sum |y|^2 +
rho |u_k - u_{k-1}|^2 using the newest measurement. Checks on the implied plant:
stable for input gains k I (k in gain_range) and for per-mode gains diag(k_i) at the
corners of the range, stable with +delay_margin frames on every input, and
max_f sigma_max((I - P C)^-1) <= ms_limit_db. (s, rho) common to all modes: the
smallest predicted trace cov(y) that passes.

The rest is the SISO scheme with y^2 -> |y|^2: denominator stabilization (only the
eigenvalues outside max_radius move), residual-based acceptance, supervisor, dither
(independent per mode) until switch_designs designs are accepted.
For m = 1 this is AdaptiveModeFreeTheta. Prototype and tests:
RAMA ORforRAMA/mimo_free_theta.py, test_mimo_free_theta.py.
"""

from collections import deque

import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov


# ------------------------------------------------------------------ #
# Model and identification                                           #
# ------------------------------------------------------------------ #
def buffer_matrices(N, m):
    """(F, G_y, G_u) of the lag-major buffers of N past y and N past u (m each)."""
    Nm = N * m
    S = np.eye(Nm, k=-m)
    Z = np.zeros((Nm, Nm))
    E1 = np.zeros((Nm, m))
    E1[:m] = np.eye(m)
    F = np.block([[S, Z], [Z, S]])
    return F, np.vstack([E1, np.zeros((Nm, m))]), np.vstack([np.zeros((Nm, m)), E1])


def buffer_model(Theta, N, m):
    """Innovations model (A, B, C, K) of y = Theta^T zeta + e."""
    F, G_y, G_u = buffer_matrices(N, m)
    Theta = np.asarray(Theta, float)
    return F + G_y @ Theta.T, G_u, Theta.T.copy(), G_y


def own_ar_masks(N, m):
    """Regressor of mode i for structure 'diag_ar': its own past and all the commands."""
    u = np.arange(N * m, 2 * N * m)
    return [np.r_[np.arange(i, N * m, m), u] for i in range(m)]


class MimoSlidingRLS:
    """Theta = (delta I + sum phi phi^T)^-1 sum phi y^T over the last `window` samples,
    y in R^m. `masks` (one index array per output) restrict each output's regressor;
    outputs with the same regressor share one inverse. Exact recomputation from the
    window every `refresh_every` updates."""

    def __init__(self, n, m, window, delta=1e-6, refresh_every=None, masks=None):
        self.n, self.m, self.window, self.delta = n, m, int(window), float(delta)
        self.refresh_every = self.window if refresh_every is None else int(refresh_every)
        masks = [np.arange(n)] * m if masks is None else [np.asarray(ix) for ix in masks]
        groups = {}
        for j, ix in enumerate(masks):
            groups.setdefault(tuple(ix.tolist()), []).append(j)
        self.groups = [(np.array(ix), np.array(cols)) for ix, cols in groups.items()]
        self.Phi = np.zeros((self.window, n))
        self.Y = np.zeros((self.window, m))
        self.count = 0
        self.P = [np.eye(len(ix)) / self.delta for ix, _ in self.groups]
        self.th = [np.zeros((len(ix), len(cols))) for ix, cols in self.groups]
        self._since = 0

    @property
    def n_samples(self):
        return min(self.count, self.window)

    @property
    def theta(self):
        Theta = np.zeros((self.n, self.m))
        for (ix, cols), th in zip(self.groups, self.th):
            Theta[np.ix_(ix, cols)] = th
        return Theta

    def _rank1(self, phi, y, sign):
        for g, (ix, cols) in enumerate(self.groups):
            p, P = phi[ix], self.P[g]
            Pp = P @ p
            K = Pp / (1.0 + sign * (p @ Pp))
            self.th[g] += sign * np.outer(K, y[cols] - self.th[g].T @ p)
            self.P[g] = P - sign * np.outer(K, Pp)

    def update(self, phi, y):
        phi, y = np.asarray(phi, float), np.asarray(y, float)
        i = self.count % self.window
        if self.count >= self.window:
            self._rank1(self.Phi[i], self.Y[i], -1.0)
        self.Phi[i], self.Y[i] = phi, y
        self._rank1(phi, y, 1.0)
        self.count += 1
        self._since += 1
        if self._since >= self.refresh_every:
            self.refresh()

    def refresh(self):
        k = self.n_samples
        Phi, Y = self.Phi[:k], self.Y[:k]
        for g, (ix, cols) in enumerate(self.groups):
            X = Phi[:, ix]
            self.P[g] = np.linalg.inv(self.delta * np.eye(len(ix)) + X.T @ X)
            self.th[g] = self.P[g] @ (X.T @ Y[:, cols])
        self._since = 0

    def residual_cov(self, theta=None):
        k = self.n_samples
        R = self.Y[:k] - self.Phi[:k] @ (self.theta if theta is None else theta)
        return R.T @ R / max(k, 1)


def stabilize_theta(Theta, m, max_radius=0.9995, max_cond=1e10):
    """Move the eigenvalues of the block companion of A(q) = I - sum A_i q^-i that lie
    outside max_radius radially onto it, keeping the other eigenvalues and all
    eigenvectors (only A_1 .. A_N change). For m = 1: adaptive_lqg.stabilize_theta.
    Falls back to the contraction A_i -> c^i A_i if the eigenvectors are
    ill-conditioned. Returns (Theta, number of eigenvalues moved)."""
    Theta = np.asarray(Theta, float).copy()
    Nm = Theta.shape[0] // 2
    N = Nm // m
    R = Theta[:Nm].T
    comp = np.zeros((Nm, Nm))
    comp[:m] = R
    comp[m:, :-m] = np.eye(Nm - m)
    lam, V = np.linalg.eig(comp)
    big = np.abs(lam) > max_radius
    if not np.any(big):
        return Theta, 0
    if np.linalg.cond(V) < max_cond:
        target = np.zeros((m, Nm), complex)
        cols = V.copy()
        for i in np.flatnonzero(big):
            mu = lam[i] * max_radius / abs(lam[i])
            x = V[-m:, i]
            v = np.concatenate([mu ** (N - 1 - j) * x for j in range(N)])
            cols[:, i] = v
            target[:, i] = mu ** N * x - R @ v
        Theta[:Nm] = (R + np.real(target @ np.linalg.inv(cols))).T
    else:
        c = max_radius / np.abs(lam).max()
        for i in range(N):
            Theta[i * m:(i + 1) * m] *= c ** (i + 1)
    return Theta, int(big.sum())


def stabilize_theta_diag(Theta, m, max_radius=0.9995):
    """stabilize_theta for a diagonal A(q): the SISO root move on each mode's AR."""
    Theta = np.asarray(Theta, float).copy()
    N = Theta.shape[0] // (2 * m)
    moved = 0
    for i in range(m):
        idx = np.arange(i, N * m, m)
        roots = np.roots(np.r_[1.0, -Theta[idx, i]])
        big = np.abs(roots) > max_radius
        if np.any(big):
            roots[big] *= max_radius / np.abs(roots[big])
            Theta[idx, i] = -np.real(np.poly(roots))[1:]
            moved += int(big.sum())
    return Theta, moved


def implied_diag_taps(Theta, N, m, n):
    """First n samples of the impulse response of each mode's own implied plant
    (diagonal of A^-1 B): an (m, n) array."""
    A, B, C, _ = buffer_model(Theta, N, m)
    X, out = B.copy(), []
    for _ in range(n):
        out.append(np.diag(C @ X))
        X = A @ X
    return np.array(out).T


# ------------------------------------------------------------------ #
# Design                                                             #
# ------------------------------------------------------------------ #
def kalman_gain(A, C, K, Sigma, s, Q_extra=None):
    """Predictor gain for x+ = A x + B u + K e + xi, y = C x + e, cov(e) = Sigma,
    cov(xi) = Q_extra (process noise independent of e, None for the pure innovations
    form), measurement noise inflated to (1 + s) Sigma; s = 0 and no Q_extra give K."""
    if s <= 0 and Q_extra is None:
        return K
    W = K @ Sigma @ K.T if Q_extra is None else K @ Sigma @ K.T + Q_extra
    P = solve_discrete_are(A.T, C.T, W, (1.0 + s) * Sigma, None, K @ Sigma)
    return np.linalg.solve((C @ P @ C.T + (1.0 + s) * Sigma).T, (A @ P @ C.T + K @ Sigma).T).T


def lqr_state(A, B, eps_pole=None):
    """(A_d, B_d) of the LQR design state: (x, u_{k-1}) and, with eps_pole = lambda,
    the running mean ubar_k = lambda ubar_{k-1} + (1 - lambda) u_{k-1} as m more states."""
    n, m = B.shape
    nz = n + m + (0 if eps_pole is None else m)
    Ad = np.zeros((nz, nz))
    Ad[:n, :n] = A
    Bd = np.zeros((nz, m))
    Bd[:n] = B
    Bd[n:n + m] = np.eye(m)
    if eps_pole is not None:
        Ad[n + m:, n:n + m] = (1.0 - eps_pole) * np.eye(m)
        Ad[n + m:, n + m:] = eps_pole * np.eye(m)
    return Ad, Bd


def lqr_gains(A, B, C, rho, W_e=None, eps=0.0, eps_pole=None):
    """(K_s, K_x) of the LQR on the state of lqr_state, cost
    y^T W_e y + rho |u_k - u_{k-1}|^2 + eps |u_k - ubar_k|^2: u_k = -K_s s_k - K_x xbar_k
    with s_k the part of xbar_{k+1} fixed by y_k. eps penalizes the command itself and
    buys sensitivity margin; eps_pole = lambda < 1 penalizes it around its own running
    mean, so the DC is free (an absolute penalty leaves eps / (G(1)^2 + eps) of any
    static aberration uncorrected)."""
    n, m = B.shape
    W_e = np.eye(m) if W_e is None else W_e
    Ad, Bd = lqr_state(A, B, eps_pole)
    nz = Ad.shape[0]
    Q = np.zeros((nz, nz))
    Q[:n, :n] = C.T @ W_e @ C
    Q[n:n + m, n:n + m] = rho * np.eye(m)
    Nx = np.zeros((nz, m))
    Nx[n:n + m] = -rho * np.eye(m)
    if eps:
        E = np.zeros((m, nz))
        if eps_pole is not None:
            E[:, n:n + m] = (1.0 - eps_pole) * np.eye(m)
            E[:, n + m:] = eps_pole * np.eye(m)
        Q += eps * E.T @ E
        Nx -= eps * E.T
    R = (max(rho, 1e-9) + eps) * np.eye(m)
    X = solve_discrete_are(Ad, Bd, Q, R, None, Nx)
    Wg = np.linalg.inv(R + Bd.T @ X @ Bd)
    return Wg @ Bd.T @ X, Wg @ Nx.T


class MimoModelLQG:
    """Kalman (measurement noise (1 + s) Sigma) + LQR with rho |u_k - u_{k-1}|^2 on
    x+ = A x + B u + K e, y = C x + e, cov(e) = Sigma: adaptive_lqg.ModelLQG with m
    inputs and outputs. Controller state xbar = (x_hat, u_{k-1}); one frame, with the
    applied command u:  u_ctrl = Cc xbar + Dc y,  xbar+ = F_bar xbar + L_bar y + Bd u."""

    def __init__(self, A, B, C, K, Sigma, rho=0.0, s=0.0, L=None, lqr=None,
                 Q_extra=None, eps=0.0, eps_pole=None):
        n, m = B.shape
        self.A, self.B, self.C, self.K = A, B, C, K
        self.Sigma = np.atleast_2d(np.asarray(Sigma, float))
        self.rho, self.s, self.eps = float(rho), float(s), float(eps)
        self.eps_pole = eps_pole
        self.Q_extra = None if Q_extra is None else np.asarray(Q_extra, float)
        self.r_kalman = self.s * float(np.trace(self.Sigma)) / m     # mean added noise variance
        self.n, self.m = n, m
        self.L = kalman_gain(A, C, K, self.Sigma, s, self.Q_extra) if L is None else L
        Ad_z, Bd = lqr_state(A, B, eps_pole if eps else None)
        nz = Ad_z.shape[0]
        K_s, K_x = lqr_gains(A, B, C, rho, None, eps, eps_pole if eps else None) if lqr is None else lqr
        self.F_bar = np.zeros((nz, nz))
        self.F_bar[:n, :n] = A - self.L @ C
        self.F_bar[n + m:, n:] = Ad_z[n + m:, n:]        # the ubar update, if there is one
        self.L_bar = np.vstack([self.L, np.zeros((nz - n, m))])
        self.Bd = Bd
        self.Cc = -(K_s @ self.F_bar + K_x)
        self.Dc = -(K_s @ self.L_bar)
        self.Ac = self.F_bar + Bd @ self.Cc
        self.Bc = self.L_bar + Bd @ self.Dc
        self.g = None                                    # implied diagonal taps, set by the owner
        self._pred = None

    @property
    def ctrl(self):
        return self.Ac, self.Bc, self.Cc, self.Dc

    @property
    def identified_plant(self):
        return self.A, self.B, self.C

    def predicted_output_std(self):
        """sqrt(trace cov(y)) predicted by the model in closed loop: total rms."""
        if self._pred is None:
            A, B, C, K = self.A, self.B, self.C, self.K
            Ac, Bc, Cc, Dc = self.ctrl
            Acl = np.block([[A + B @ Dc @ C, B @ Cc], [Bc @ C, Ac]])
            if np.abs(np.linalg.eigvals(Acl)).max() >= 1.0:
                self._pred = np.inf
            else:
                Bcl = np.vstack([K + B @ Dc, Bc])
                Ccl = np.hstack([C, np.zeros((self.m, Ac.shape[0]))])
                Qcl = Bcl @ self.Sigma @ Bcl.T
                if self.Q_extra is not None:
                    Qcl[:self.n, :self.n] += self.Q_extra
                Sig = solve_discrete_lyapunov(Acl, Qcl)
                self._pred = float(np.sqrt(np.trace(Ccl @ Sig @ Ccl.T + self.Sigma)))
        return self._pred


def closed_loop_matrix(plant, ctrl, G=None):
    """Plant (A, B, C) with input gain matrix G, controller (Ac, Bc, Cc, Dc)."""
    A, B, C = plant
    Ac, Bc, Cc, Dc = ctrl
    BG = B if G is None else B @ G
    return np.block([[A + BG @ Dc @ C, BG @ Cc], [Bc @ C, Ac]])


def with_extra_delay(plant, frac):
    """plant * ((1 - frac) + frac q^-1) on every input."""
    A, B, C = plant
    n, m = B.shape
    A_e = np.block([[A, frac * B], [np.zeros((m, n)), np.zeros((m, m))]])
    return A_e, np.vstack([(1.0 - frac) * B, np.eye(m)]), np.hstack([C, np.zeros((C.shape[0], m))])


def frequency_response(A, B, C, D, freqs_hz, dt):
    """(nf, p, m) transfer matrices C (zI - A)^-1 B + D."""
    z = np.exp(2j * np.pi * np.asarray(freqs_hz) * dt)
    M = z[:, None, None] * np.eye(A.shape[0]) - A
    return C @ np.linalg.solve(M, np.broadcast_to(B, (len(z),) + B.shape)) + D


def peak_sensitivity_db(plant, ctrl, dt, n=300, P=None):
    """max over frequency of sigma_max((I - P C)^-1). P: the plant response on the same
    grid (sensitivity_grid), if already computed."""
    f = sensitivity_grid(dt, n)
    A, B, C = plant
    m = B.shape[1]
    P = frequency_response(A, B, C, 0.0, f, dt) if P is None else P
    Kc = frequency_response(*ctrl, f, dt)
    S = np.linalg.inv(np.eye(m) - P @ Kc)
    return float(20 * np.log10(np.linalg.svd(S, compute_uv=False)[:, 0].max()))


def sensitivity_grid(dt, n=300):
    return np.linspace(0.5, 0.5 / dt, n)


def gain_matrices(m, gain_range, n_uniform=11, max_corners=16, seed=0):
    """Input gains to check: k I on a grid and the corners of diag(k_1 .. k_m) (all
    2^m if 2^m <= max_corners, else a fixed random subset)."""
    lo, hi = gain_range
    Gs = [k * np.eye(m) for k in np.linspace(lo, hi, n_uniform)]
    if m > 1:
        if 2 ** m <= max_corners:
            corners = [[(hi if (c >> i) & 1 else lo) for i in range(m)] for c in range(2 ** m)]
        else:
            corners = np.random.default_rng(seed).choice([lo, hi], size=(max_corners, m)).tolist()
        Gs += [np.diag(c) for c in corners if len(set(c)) > 1]
    return Gs


def tune_mimo_lqg(Theta, N, m, Sigma, dt, ms_limit_db=6.0, gain_range=(0.5, 1.5), delay_margin=0.0,
                  s_grid=(0.0, 0.1, 1.0, 10.0, 100.0, 1e3), rho_grid=(0.0, 1.0, 10.0), corners=True):
    """tune_mimo_model on buffer_model(Theta)."""
    return tune_mimo_model(*buffer_model(Theta, N, m), Sigma, dt, ms_limit_db, gain_range, delay_margin,
                           s_grid, rho_grid, corners)


def tune_mimo_model(A, B, C, K, Sigma, dt, ms_limit_db=6.0, gain_range=(0.5, 1.5), delay_margin=0.0,
                    s_grid=(0.0, 0.1, 1.0, 10.0, 100.0, 1e3), rho_grid=(0.0, 1.0, 10.0), corners=True,
                    eps_grid=(0.0,), eps_pole=None, Q_extra=None):
    """(s, rho) of MimoModelLQG on the innovations model (A, B, C, K) with the smallest
    predicted total residual that passes the checks on its plant (A, B, C) (module
    docstring). Candidates are ranked by the prediction, the checks run from the best down."""
    m = B.shape[1]
    plant = (A, B, C)
    Gs = gain_matrices(m if corners else 1, gain_range)
    if not corners:
        Gs = [G[0, 0] * np.eye(m) for G in Gs]
    uniform = [G for G in Gs if np.allclose(G, G[0, 0] * np.eye(m))]
    delayed = with_extra_delay(plant, delay_margin) if delay_margin > 0 else None
    Ls, lqrs = {}, {}                                   # Kalman depends on s only, LQR on (rho, eps)
    for s in s_grid:
        try:
            Ls[s] = kalman_gain(A, C, K, Sigma, s, Q_extra)
        except (np.linalg.LinAlgError, ValueError):
            pass
    for rho in rho_grid:
        for eps in eps_grid:
            try:
                lqrs[(rho, eps)] = lqr_gains(A, B, C, rho, None, eps, eps_pole if eps else None)
            except (np.linalg.LinAlgError, ValueError):
                pass
    cands = []
    for s in Ls:
        for (rho, eps), lqr in lqrs.items():
            try:
                d = MimoModelLQG(A, B, C, K, Sigma, rho, s, L=Ls[s], lqr=lqr,
                                 Q_extra=Q_extra, eps=eps, eps_pole=eps_pole)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(d.predicted_output_std()):
                cands.append(d)
    # cheapest checks first (nominal stability, Ms), then the gain and delay sweeps;
    # a candidate passes only if all pass, so the order changes the time, not the result
    P_f = frequency_response(A, B, C, 0.0, sensitivity_grid(dt), dt)
    others = [G for G in Gs if not np.allclose(G, np.eye(m))]
    for d in sorted(cands, key=lambda c: c.predicted_output_std()):
        if np.abs(np.linalg.eigvals(closed_loop_matrix(plant, d.ctrl))).max() >= 1:
            continue
        if peak_sensitivity_db(plant, d.ctrl, dt, P=P_f) > ms_limit_db:
            continue
        if not all(np.abs(np.linalg.eigvals(closed_loop_matrix(plant, d.ctrl, G))).max() < 1 for G in others):
            continue
        if delayed is not None and not all(
                np.abs(np.linalg.eigvals(closed_loop_matrix(delayed, d.ctrl, G))).max() < 1 for G in uniform):
            continue
        return d
    raise RuntimeError("no (s, rho) in the grid meets the constraints on the identified plant")


# ------------------------------------------------------------------ #
# Adaptive controller                                                #
# ------------------------------------------------------------------ #
class AdaptiveMimoFreeTheta:
    """m modal channels with one free Theta. Same interface as AdaptiveModeFreeTheta,
    vectors instead of scalars: step(y_k) -> (u_k, r_k), `log`, `design`, `name`.

    Every frame: sliding-window least squares on (zeta_k, y_k); control with the active
    MimoModelLQG (integrator warmup_gain per mode before the first design); dither r.
    Every `redesign_every` frames, once `min_samples` are in the window:
    tune_mimo_lqg on the stabilized Theta, then
      acceptance='model'     switch to it at once
      acceptance='residual'  trial of one period on |y|^2: abort as soon as the fast
                             |y|^2 exceeds abort_ratio^2 times the mean |y|^2 of the
                             last clean period, keep it if the trial period is at most
                             accept_ratio^2 times that.
    """

    def __init__(self, dt, m, N=11, window=4000, min_samples=None, structure="full", dither_std=5.0,
                 dither_after=None, switch_designs=2, redesign_every=250, ms_limit_db=6.0,
                 gain_range=(0.5, 1.5), delay_margin=0.5, warmup_gain=0.4, acceptance="residual",
                 accept_ratio=1.05, abort_ratio=1.5, hard_abort_ratio=4.0, abort_min_frames=20, n_taps_log=3, max_radius=0.9995,
                 corners=True, supervisor=True, blowup_factor=2.0, fast_window=50, holdoff=4,
                 rng=None, name=""):
        if acceptance not in ("model", "residual"):
            raise ValueError(f"acceptance must be 'model' or 'residual', got {acceptance!r}")
        if structure not in ("full", "diag_ar"):
            raise ValueError(f"structure must be 'full' or 'diag_ar', got {structure!r}")
        self.dt, self.m, self.N = float(dt), int(m), int(N)
        self.structure = structure
        masks = own_ar_masks(self.N, self.m) if structure == "diag_ar" else None
        self.rls = MimoSlidingRLS(2 * self.N * self.m, self.m, window, masks=masks)
        self.min_samples = int(window if min_samples is None else min_samples)
        self.dither_std = np.broadcast_to(np.asarray(dither_std, float), (self.m,)).copy()
        self.dither_after = self.dither_std.copy() if dither_after is None else \
            np.broadcast_to(np.asarray(dither_after, float), (self.m,)).copy()
        self.switch_designs = int(switch_designs)
        self.dither_level = self.dither_std.copy()
        self.redesign_every = int(redesign_every)
        self.ms_limit_db, self.gain_range = float(ms_limit_db), tuple(gain_range)
        self.delay_margin = float(delay_margin)
        self.warmup_gain = np.broadcast_to(np.asarray(warmup_gain, float), (self.m,)).copy()
        self.acceptance = acceptance
        self.accept_ratio, self.abort_ratio = float(accept_ratio), float(abort_ratio)
        self.hard_abort_ratio = float(hard_abort_ratio)
        self.abort_min_frames, self.n_taps_log = int(abort_min_frames), int(n_taps_log)
        self.max_radius, self.corners = float(max_radius), bool(corners)
        self.supervisor, self.blowup_factor, self.holdoff = supervisor, float(blowup_factor), int(holdoff)
        self.fast_alpha = 1.0 / fast_window
        self.rng = np.random.default_rng() if rng is None else rng
        self.name = name

        self.y_hist = np.zeros((self.N, self.m))         # row i: y_{k-1-i}
        self.u_hist = np.zeros((self.N, self.m))
        self.k = 0
        self.u_int = np.zeros(self.m)
        self.design = self.good_design = None
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

    def _zeta(self):
        return np.r_[self.y_hist.ravel(), self.u_hist.ravel()]

    # ---------------- design ---------------- #
    def _redesign(self):
        stab = stabilize_theta_diag if self.structure == "diag_ar" else stabilize_theta
        Theta, moved = stab(self.rls.theta, self.m, self.max_radius)
        Sigma = self.rls.residual_cov(Theta)
        taps = implied_diag_taps(Theta, self.N, self.m, self.n_taps_log)
        try:
            d = tune_mimo_lqg(Theta, self.N, self.m, Sigma, self.dt, self.ms_limit_db, self.gain_range,
                              self.delay_margin, corners=self.corners)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
            self.log.append((self.k, "rejected", dict(reason=str(exc), g=taps.tolist())))
            return
        d.g = taps
        info = dict(rho=d.rho, r_kalman=d.r_kalman, s=d.s, innovation_rms=np.sqrt(np.diag(Sigma)).tolist(),
                    g=taps.tolist(), pred_std=d.predicted_output_std(), dither=self.dither_level.tolist(),
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
            self.dither_level = self.dither_after.copy()
        self.log.append((self.k, "accepted", info))

    def _activate(self, design):
        self.design = design
        if design is not None:
            self.xc = np.r_[self._zeta(), self.u_hist[0]]

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

    def _watch_trial(self, y2):
        self.trial_fast = (1 - self.fast_alpha) * self.trial_fast + self.fast_alpha * y2
        # the averaged criterion needs abort_min_frames to ignore single noisy samples, but an
        # unstable loop grows by ~7x in 5 frames at 1 kHz, so one sample far above the reference
        # aborts at once: a false positive only costs one period on the previous design
        hard = y2 > self.hard_abort_ratio ** 2 * self.incumbent_ms
        if hard or (self.k - self.trial["start"] >= self.abort_min_frames
                    and self.trial_fast > self.abort_ratio ** 2 * self.incumbent_ms):
            t, self.trial = self.trial, None
            self._activate(t["prev"])
            self.hold = max(self.hold, 1)
            self.clean = False
            self.log.append((self.k, "trial aborted",
                             dict(t["info"], after=self.k - t["start"], instantaneous=bool(hard))))

    # ---------------- supervisor ---------------- #
    def _supervise(self, y2):
        if not self.supervisor:
            return
        self.fast_ms = y2 if self.fast_ms is None else (1 - self.fast_alpha) * self.fast_ms + self.fast_alpha * y2
        self.period_sq += y2
        if self.trial is None and self.healthy \
                and self.fast_ms > (self.blowup_factor ** 2) * np.median(self.healthy) \
                and self.design is not None and self.design is not self.good_design:
            self._activate(self.good_design)
            self.hold = self.holdoff
            self.fast_ms = None
            self.clean = False
            self.dither_level = self.dither_std.copy()
            self.log.append((self.k, "reverted", dict(to="previous good design"
                                                       if self.good_design is not None else "integrator")))

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
        y = np.asarray(y, float)
        y2 = float(y @ y)
        if self.k >= self.N:
            self.rls.update(self._zeta(), y)
        self._supervise(y2)
        if self.trial is not None:
            self._watch_trial(y2)

        if self.k > 0 and self.k % self.redesign_every == 0:
            ms, self.p_sq = self.p_sq / self.redesign_every, 0.0
            if self.trial is not None:
                self._finish_trial(ms)
            elif self.clean:
                self.incumbent_ms = ms
            self.clean = True
            self._end_of_period()
            if self.hold > 0:
                self.hold -= 1
            elif self.rls.n_samples >= self.min_samples:
                self._redesign()

        r = self.dither_level * self.rng.standard_normal(self.m)
        if self.design is None:
            self.u_int = self.u_int + self.warmup_gain * y
            u_ctrl = self.u_int
        else:
            dsg = self.design
            u_ctrl = dsg.Cc @ self.xc + dsg.Dc @ y
            self.u_int = u_ctrl                          # fallback integrator starts bumpless
        u = u_ctrl + r

        if self.design is not None:
            dsg = self.design
            self.xc = dsg.F_bar @ self.xc + dsg.L_bar @ y + dsg.Bd @ u
        self.y_hist = np.vstack([y, self.y_hist[:-1]])
        self.u_hist = np.vstack([u, self.u_hist[:-1]])
        self.p_sq += y2
        self.k += 1
        return u, r
