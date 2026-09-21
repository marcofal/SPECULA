"""Adaptive MIMO LQG on vector-AR turbulence + structured plant, starting from SISO.

Model, m modes:

    y_k = G(q) u_k + d_k,   G(q) = diag(g_1(q) .. g_m(q)), g_i(q) = sum_{j=1..n_g} g_ij q^-j
    A(q) d_k = e_k,         A(q) = I - sum_{l=1..p} A_l q^-l (full: vector AR),  cov(e) = Sigma

Why: modal turbulence is correlated across modes at nonzero lags (frozen flow), so the
other modes' past predicts a mode better (on RAMA LEO tracking: 2-step error -16 %,
tip/tilt about -20 %). That coupling belongs to A(q), and A is learned from the whole
measured signal, so it is well conditioned. The plant cross-talk of a realistic
misregistration (a few %) is below what a small dither identifies in seconds, and a
free Theta that leaves the turbulence coupling out of A shows it as fake plant cross-talk.
So: dense A, diagonal G by instrumental variables on the dither (as the structured SISO
method), both refitted in batch on the last `window` frames at every redesign.

Design: the innovations model with the minimal state x_k = (u_{k-1} .. u_{k-n_g},
d_{k-1} .. d_{k-p}) (structured_model); MIMO Kalman + LQR, checks and tuning of
adaptive_lqg_mimo (tune_mimo_model). Starting from SISO: the per-mode AR is used until
`siso_first` designs are accepted, then the vector AR whenever it beats the per-mode AR
on held-out data (2-step prediction error, the last `holdout` of the window) by more
than `var_margin`. Acceptance, supervisor and dither as in adaptive_lqg_mimo.
Prototype and tests: RAMA ORforRAMA/var_lqg.py, adaptive_var_lqg.py, compare_var_lqg.py.
"""
from collections import deque

import numpy as np

from specula.lib.adaptive_lqg import fit_plant_iv
from specula.lib.adaptive_lqg_mimo import stabilize_theta, stabilize_theta_diag, tune_mimo_model


def fit_plant_diag(Y, U, R, n_g=3, p=8, skip=100):
    """Per-mode FIR taps (m, n_g) of u_i -> y_i, instruments = dither r_i."""
    return np.array([fit_plant_iv(Y[:, i], U[:, i], R[:, i], n_g, p, skip=skip)[0] for i in range(Y.shape[1])])


def apply_plant_diag(G, U):
    """G(q) u for diagonal FIR taps G (m, n_g): row k uses u_{k-1} .. u_{k-n_g}."""
    out = np.zeros_like(U, dtype=float)
    for j in range(G.shape[1]):
        out[j + 1:] += U[:len(U) - j - 1] * G[:, j]
    return out


def var_regressors(D, p):
    T, m = D.shape
    X = np.zeros((T, p * m))
    for l in range(1, p + 1):
        X[l:, (l - 1) * m:l * m] = D[:T - l]
    return X


def fit_var(D, p, skip=0, diag=False, max_radius=0.9995):
    """Vector AR(p) D_k = sum_l A_l D_{k-l} + e_k (diag=True: one AR per mode), block
    companion eigenvalues pulled inside max_radius. Returns (A (p, m, m), Sigma, moved)."""
    T, m = D.shape
    X, Y = var_regressors(D, p)[skip:], D[skip:]
    A = np.zeros((p, m, m))
    if diag:
        for i in range(m):
            cols = np.arange(i, p * m, m)
            A[:, i, i] = np.linalg.lstsq(X[:, cols], Y[:, i], rcond=None)[0]
    else:
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        for l in range(p):
            A[l] = W[l * m:(l + 1) * m].T
    Theta_y = np.vstack([A[l].T for l in range(p)])
    stab = stabilize_theta_diag if diag else stabilize_theta
    Th, moved = stab(np.vstack([Theta_y, np.zeros_like(Theta_y)]), m, max_radius)
    for l in range(p):
        A[l] = Th[l * m:(l + 1) * m].T
    E = Y - X @ np.vstack([A[l].T for l in range(p)])
    return A, E.T @ E / len(E), moved


def var_prediction_error(D, A, horizon=1):
    """rms per mode of the h-step prediction error of the vector AR on D."""
    p = A.shape[0]
    errs = []
    for k in range(p + horizon - 1, len(D)):
        hist = [D[k - horizon - l] for l in range(p)]
        for _ in range(horizon):
            hist = [sum(A[l] @ hist[l] for l in range(p))] + hist[:-1]
        errs.append(D[k] - hist[0])
    return np.sqrt(np.mean(np.array(errs) ** 2, axis=0))


def structured_model(G, A, bias_pole=None):
    """Innovations model (A_s, B_s, C_s, K_s) of y = G(q) u + d, A(q) d = e, state
    x_k = (u_{k-1} .. u_{k-n_g}, d_{k-1} .. d_{k-p}): y_k = C x_k + e_k with
    C = [G_1 .. G_n_g, A_1 .. A_p]; d_k = [A_1 .. A_p] d-lags + e_k enters the state.

    bias_pole (lambda_b < 1): m extra states b with b+ = lambda_b b + eta and d = b + v,
    A(q) v = e, so C = [G_1 .. G_n_g, A_1 .. A_p, I] and the d-lags hold the fluctuation.
    Without them the AR poles pull the prediction to zero and a steady aberration is left
    uncorrected; the leak keeps the uncontrollable bias mode inside the unit circle, which
    the LQR Riccati needs. cov(eta) goes to the Kalman filter as Q_extra (see `design`)."""
    G, A = np.asarray(G, float), np.asarray(A, float)
    m, n_g = G.shape
    p = A.shape[0]
    nu, nd = n_g * m, p * m
    nb = 0 if bias_pole is None else m
    n = nu + nd + nb
    As, Bs, Cs, Ks = np.zeros((n, n)), np.zeros((n, m)), np.zeros((m, n)), np.zeros((n, m))
    As[m:nu, :nu - m] = np.eye(nu - m)
    Bs[:m] = np.eye(m)
    AR = np.hstack([A[l] for l in range(p)])
    for j in range(n_g):
        Cs[:, j * m:(j + 1) * m] = np.diag(G[:, j])
    Cs[:, nu:nu + nd] = AR
    As[nu:nu + m, nu:nu + nd] = AR
    Ks[nu:nu + m] = np.eye(m)
    As[nu + m:nu + nd, nu:nu + nd - m] = np.eye(nd - m)
    if nb:
        As[nu + nd:, nu + nd:] = float(bias_pole) * np.eye(m)
        Cs[:, nu + nd:] = np.eye(m)
    return As, Bs, Cs, Ks


def bias_noise(G, A, bias_std):
    """Q_extra of structured_model(.., bias_pole): cov(eta), on the bias states only."""
    m, n_g = np.asarray(G, float).shape
    n = (n_g + A.shape[0]) * m
    Q = np.zeros((n + m, n + m))
    Q[n:, n:] = np.diag(np.broadcast_to(np.asarray(bias_std, float), (m,)) ** 2)
    return Q


def state_from_history(G, A, y_hist, u_hist, mu=None):
    """x_k of structured_model from y_hist[i] = y_{k-1-i}, u_hist[i] = u_{k-1-i}
    (depth >= p + n_g): exact command lags, d_{k-l} = y_{k-l} - G u. With mu (the
    estimated DC of d) the state holds v = d - mu and the bias state starts at mu."""
    n_g = G.shape[1]
    p = A.shape[0]
    d = [y_hist[l] - sum(G[:, j] * u_hist[l + j + 1] for j in range(n_g)) for l in range(p)]
    if mu is None:
        return np.r_[np.ravel(u_hist[:n_g]), np.ravel(d)]
    mu = np.asarray(mu, float)
    return np.r_[np.ravel(u_hist[:n_g]), np.ravel([x - mu for x in d]), mu]


class AdaptiveVarLQG:
    """m modal channels, vector-AR turbulence + diagonal plant. Same interface as
    adaptive_lqg_mimo.AdaptiveMimoFreeTheta: step(y_k) -> (u_k, r_k), `log`, `design`,
    `name`; the design carries `g` (m, n_g), the identified taps."""

    def __init__(self, dt, m, n_g=3, p=8, window=4000, min_samples=None, dither_std=5.0,
                 dither_after=None, switch_designs=2, redesign_every=250, ms_limit_db=6.0,
                 gain_range=(0.5, 1.5), delay_margin=0.5, warmup_gain=0.4, var_margin=0.02,
                 holdout=0.25, siso_first=1, bias_tau=2.0, bias_rel=0.1,
                 rho_grid=(0.0, 0.1, 0.3, 1.0, 10.0), eps_grid=(0.0, 0.1), eps_tau=5e-3,
                 plant_min_dither=1.0, allow_var=True, acceptance="residual",
                 accept_ratio=1.05, abort_ratio=1.5, hard_abort_ratio=4.0, abort_min_frames=20, max_radius=0.9995,
                 corners=True, supervisor=True, blowup_factor=2.0, fast_window=50, holdoff=4,
                 rng=None, name=""):
        if acceptance not in ("model", "residual"):
            raise ValueError(f"acceptance must be 'model' or 'residual', got {acceptance!r}")
        self.dt, self.m, self.n_g, self.p = float(dt), int(m), int(n_g), int(p)
        self.depth = self.p + self.n_g + 1
        self.window = int(window)
        self.min_samples = self.window if min_samples is None else int(min_samples)
        self.dither_std = np.broadcast_to(np.asarray(dither_std, float), (self.m,)).copy()
        self.dither_after = self.dither_std.copy() if dither_after is None else \
            np.broadcast_to(np.asarray(dither_after, float), (self.m,)).copy()
        self.switch_designs = int(switch_designs)
        self.dither_level = self.dither_std.copy()
        self.redesign_every = int(redesign_every)
        self.tune_kw = dict(ms_limit_db=float(ms_limit_db), gain_range=tuple(gain_range),
                            delay_margin=float(delay_margin), corners=bool(corners),
                            rho_grid=tuple(rho_grid), eps_grid=tuple(eps_grid),
                            eps_pole=None if eps_tau is None else float(np.exp(-self.dt / float(eps_tau))))
        self.bias_pole = None if bias_tau is None else float(np.exp(-self.dt / float(bias_tau)))
        self.bias_rel = float(bias_rel)
        self.warmup_gain = np.broadcast_to(np.asarray(warmup_gain, float), (self.m,)).copy()
        self.var_margin, self.holdout, self.siso_first = float(var_margin), float(holdout), int(siso_first)
        self.plant_min_dither, self.allow_var = float(plant_min_dither), bool(allow_var)
        self.acceptance = acceptance
        self.accept_ratio, self.abort_ratio = float(accept_ratio), float(abort_ratio)
        self.hard_abort_ratio = float(hard_abort_ratio)
        self.abort_min_frames, self.max_radius = int(abort_min_frames), float(max_radius)
        self.supervisor, self.blowup_factor, self.holdoff = supervisor, float(blowup_factor), int(holdoff)
        self.fast_alpha = 1.0 / fast_window
        self.rng = np.random.default_rng() if rng is None else rng
        self.name = name

        self.Y, self.U, self.R = (deque(maxlen=self.window) for _ in range(3))
        self.y_hist = np.zeros((self.depth, self.m))
        self.u_hist = np.zeros((self.depth, self.m))
        self.G = None
        self.k = 0
        self.u_int = np.zeros(self.m)
        self.u_bar = np.zeros(self.m)                    # running mean of the applied command (eps term)
        self.design = self.good_design = None
        self.n_accepted = 0
        self.xc = None
        self.period_sq, self.healthy = 0.0, deque(maxlen=8)
        self.fast_ms = None
        self.hold = 0
        self.log = []
        self.trial = None
        self.incumbent_ms = None
        self.clean = True
        self.p_sq = 0.0
        self.trial_fast = 0.0

    # ---------------- identification and design ---------------- #
    def _identify(self):
        Y, U, R = np.array(self.Y), np.array(self.U), np.array(self.R)
        G = fit_plant_diag(Y, U, R, self.n_g, p=8, skip=100)
        if self.G is not None:
            weak = R[100:].std(axis=0) < self.plant_min_dither
            G[weak] = self.G[weak]
        self.G = G
        D = (Y - apply_plant_diag(G, U))[100:]
        mu = D.mean(axis=0) if self.bias_pole is not None else np.zeros(self.m)
        D = D - mu                                       # the DC goes to the bias state, not to the AR
        n_fit = int(len(D) * (1 - self.holdout))
        err = {}
        for label, diag in (("per-mode AR", True), ("vector AR", False)):
            A, _, _ = fit_var(D[:n_fit], self.p, skip=self.p, diag=diag, max_radius=self.max_radius)
            err[label] = float(np.sqrt(np.sum(var_prediction_error(D[n_fit:], A, horizon=2) ** 2)))
        use_var = self.allow_var and self.n_accepted >= self.siso_first \
            and err["vector AR"] <= (1 - self.var_margin) * err["per-mode AR"]
        A, S, moved = fit_var(D, self.p, skip=self.p, diag=not use_var, max_radius=self.max_radius)
        if not use_var:
            S = np.diag(np.diag(S))
        info = dict(structure="vector AR" if use_var else "per-mode AR",
                    holdout_err_ar=err["per-mode AR"], holdout_err_var=err["vector AR"],
                    roots_stabilized=moved, mu=np.round(mu, 1).tolist())
        return G, A, S, mu, info

    def _redesign(self):
        try:
            G, A, S, mu, info = self._identify()
            if self.bias_pole is None:
                d = tune_mimo_model(*structured_model(G, A), S, self.dt, **self.tune_kw)
            else:
                d = tune_mimo_model(*structured_model(G, A, self.bias_pole), S, self.dt,
                                    Q_extra=bias_noise(G, A, self.bias_rel * np.sqrt(np.diag(S))),
                                    **self.tune_kw)
        except (RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
            self.log.append((self.k, "rejected", dict(reason=str(exc))))
            return
        d.g, d.var_model = G, (G, A)
        d.bias_mu = None if self.bias_pole is None else mu
        info.update(rho=d.rho, r_kalman=d.r_kalman, s=d.s, eps=d.eps, g=G.tolist(),
                    pred_std=d.predicted_output_std(), dither=self.dither_level.tolist())
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
            G, A = design.var_model
            x = state_from_history(G, A, self.y_hist, self.u_hist, design.bias_mu)
            self.xc = np.r_[x, self.u_hist[0]]
            if design.Ac.shape[0] == len(self.xc) + self.m:   # the eps penalty carries ubar
                self.xc = np.r_[self.xc, self.u_bar]

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
            elif len(self.Y) >= self.min_samples:
                self._redesign()

        r = self.dither_level * self.rng.standard_normal(self.m)
        if self.design is None:
            self.u_int = self.u_int + self.warmup_gain * y
            u_ctrl = self.u_int
        else:
            dsg = self.design
            u_ctrl = dsg.Cc @ self.xc + dsg.Dc @ y
            self.u_int = u_ctrl
        u = u_ctrl + r

        if self.design is not None:
            dsg = self.design
            self.xc = dsg.F_bar @ self.xc + dsg.L_bar @ y + dsg.Bd @ u
        self.Y.append(y)
        self.U.append(u)
        self.R.append(r)
        pole = self.tune_kw["eps_pole"]
        if pole is not None:
            self.u_bar = pole * self.u_bar + (1.0 - pole) * self.u_hist[0]
        self.y_hist = np.vstack([y, self.y_hist[:-1]])
        self.u_hist = np.vstack([u, self.u_hist[:-1]])
        self.p_sq += y2
        self.k += 1
        return u, r
