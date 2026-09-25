import json
import os

import numpy as np

from specula import cpuArray
from specula.base_processing_obj import OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.adaptive_lqg import AdaptiveModeFreeTheta, AdaptiveModeLQG
from specula.lib.adaptive_lqg_mimo import AdaptiveMimoFreeTheta
from specula.lib.adaptive_lqg_var import AdaptiveVarLQG, PlantIdentifier
from specula.processing_objects.base_filter import BaseFilter


class DataDrivenLqg(BaseFilter):
    """
    Data-driven adaptive LQG modal controller.

    The modes listed in `lqg_modes` are controlled by an adaptive LQG that
    identifies, from closed-loop data only, a short FIR plant (loop latency,
    fractional delay and modal optical gain included) and an AR model of the
    turbulence, and redesigns a Kalman + LQR controller every
    `redesign_every` frames. All other modes run a plain integrator.
    See specula.lib.adaptive_lqg for the algorithm.

    Each LQG mode adds a white dither of std `dither_std` to its command: it is
    the instrument that makes the plant identifiable in closed loop. Before the
    first accepted design (warm-up, at least `min_samples` frames) the LQG
    modes also run an integrator with gain `warmup_gain`.

    Four identification methods (`method`):
      'structured'  FIR plant (instrumental variables on the dither) + AR turbulence
      'free_theta'  output equation y_k = theta^T zeta_k + e_k on buffer filters
                    (zeta_k = past y and past u, N of each), all 2N entries free,
                    one sliding-window least-squares fit; new designs accepted on
                    the measured residual (trial period), so the plant implied by
                    theta does not need to be right; the dither can be lowered to
                    `dither_after` once `switch_designs` designs are accepted.
      'mimo_free_theta'  the same on all the LQG modes jointly (vector y, matrix
                    Theta, specula.lib.adaptive_lqg_mimo): the cross-talk between
                    them (misregistration) is identified and designed for, and the
                    robustness checks are made on the coupled loop. One controller,
                    one event log; the out_design rows share its design.
      'var_lqg'     the LQG modes jointly on a vector-AR turbulence model (the other
                    modes' past predicts a mode: frozen flow) and a diagonal FIR plant
                    by instrumental variables on the dither (specula.lib.adaptive_lqg_var),
                    refitted in batch on the last plant_window frames at every redesign.
                    Starts from per-mode AR models; the vector AR is used once
                    `siso_first` designs are accepted and it beats the per-mode AR on
                    held-out data by `var_margin`. Keep the dither on (dither_after
                    unset or > 0): it identifies the plant.

    Timing: the command written at frame k is u_k (out_comm_no_delay);
    out_comm is delayed by `delay` frames like any BaseFilter. The identified
    plant describes u_k -> y, so it includes this delay and the DM layer delay.

    Parameters
    ----------
    simul_params : SimulParams
    n_modes : int
        Total number of controlled modes (size of delta_comm).
    lqg_modes : list[int]
        Indices of the modes controlled by the adaptive LQG.
    int_gain : float or list[float]
        Integrator gain for the other modes (scalar or one per mode).
    warmup_gain : float or list[float], optional
        Integrator gain of the LQG modes during warm-up (default: int_gain of
        that mode). Scalar or one per LQG mode.
    int_ff : float or list[float]
        Forgetting factor of the integrator of the other modes, as `ff` of
        Integrator: u_k = ff u_{k-1} + g y_k. 1 (default) is the pure integrator,
        ff < 1 a leaky one. Scalar or one per mode.
    warmup_ff : float or list[float], optional
        Forgetting factor of the warm-up integrator of the LQG modes (default:
        int_ff of that mode). Scalar or one per LQG mode.
    warmup_num, warmup_den : list[float] or list[list[float]], optional
        free_theta: run this IIR filter instead of the warm-up integrator, and fall
        back to it on a revert to no design. Coefficients as IirFilterData (oldest
        first, num[-1] multiplies y_k, den[-1] multiplies u_k). One row for all the
        LQG modes or one row per LQG mode.
    dither_std, noise_var : float or list[float]
        Dither std and measurement-noise variance, in the units of delta_comm
        (scalar or one per LQG mode).
    n_g, p : int
        Number of plant FIR taps (lags 1..n_g frames) and AR order.
    plant_window, dist_window : int
        Sliding windows [frames] of the plant and disturbance fits.
    min_samples : int, optional
        Frames of data before the first design attempt (default plant_window).
    redesign_every : int
        Frames between redesigns.
    ms_limit_db, gain_range, delay_margin
        Robustness checks on the identified plant: peak sensitivity, stability
        for a plant gain in gain_range, and with delay_margin extra frames.
    plant_rel_std_max : float
        Plant gate: relative std of the identified DC gain.
    supervisor, blowup_factor, fast_window, holdoff
        Revert a design whose fast residual rms exceeds blowup_factor times the
        healthy level; then wait holdoff periods before redesigning.
    method : str
        'structured' (default), 'free_theta', 'mimo_free_theta' or 'var_lqg'.
    n_theta : int
        free_theta: N, number of past y and past u in zeta (2N parameters).
    theta_window : int, optional
        free_theta: least-squares window [frames] (default plant_window).
    dither_after : float, optional
        free_theta: dither std once `switch_designs` designs are accepted
        (default dither_std).
    switch_designs : int
        free_theta: accepted designs before the dither goes to dither_after.
    acceptance : str
        free_theta: 'residual' (trial period, default) or 'model'.
    accept_ratio, abort_ratio, hard_abort_ratio : float
        free_theta: keep a trial design if its period rms <= accept_ratio times the
        reference rms; abort it as soon as the fast rms > abort_ratio times it, or at
        once if a single frame exceeds hard_abort_ratio times it (an unstable loop grows
        by orders of magnitude within the 20 frames the averaged test needs).
    mimo_structure : str
        mimo_free_theta: 'full' (every entry of Theta free, default), 'diag_ar'
        (each mode's own AR, full cross-talk in the command part) or 'diag_plant'
        (full coupling in the measurement part, each mode's own commands only).
    theta_poles : list, optional
        mimo_free_theta: poles of the nonminimal filters (Lambda, ell) of
        nonminimal_observer.cascade_filter, one list for both halves of zeta (n_theta is
        then its length). Default: the buffers (all poles at 0).
    theta_refit_u : bool
        mimo_free_theta: after moving the model poles inside the unit circle, refit the
        command part of Theta by least squares with the measurement part fixed.
    corner_gains : bool
        mimo_free_theta, var_lqg: also check stability for per-mode gains at the
        corners of gain_range (default True).
    var_margin, holdout, siso_first
        var_lqg: relative margin by which the vector AR must beat the per-mode AR on
        the last `holdout` of the window (2-step prediction error), and number of
        accepted per-mode designs before the vector AR is allowed.
    bias_tau, bias_rel : float
        var_lqg: the disturbance model carries a DC state per mode, b+ = exp(-dt/bias_tau) b
        + eta with std(eta) = bias_rel sqrt(diag(Sigma)). Without it the AR poles pull the
        prediction to zero and a steady aberration is left partly uncorrected. bias_tau
        None disables it.
    eps_grid, eps_tau : list of float, float
        var_lqg, mimo_free_theta: LQR penalty eps |u_k - ubar_k|^2 on top of rho |u_k - u_{k-1}|^2, with
        ubar the running command mean of time constant eps_tau (eps_tau None: the absolute
        command, which leaves a static error). It buys sensitivity margin, so the tuner can
        keep rho low.
    ident_modes : list of int, optional
        Modes to identify but not control: they stay on the integrator, a dither of
        `ident_dither` nm rms is added to their command and their plant taps are refitted
        every `redesign_every` frames. |G_i(1)| is then the optical gain of that mode
        (times the diagonal of the misregistration). Identification costs ~1.5 ms per mode
        against 16-31 ms to design one, so it can cover the whole basis while the LQG
        stays on the low orders; the cost is the injected dither, sqrt(n_modes) times its
        per-mode amplitude.
    ident_dither, ident_window, ident_min_samples
        Dither amplitude [nm rms], sliding window and the samples needed before the first
        fit, for `ident_modes`.
    seed : int
        Dither seed.
    log_file : str, optional
        JSON file with the design events of every LQG mode, written at the end.

    Outputs
    -------
    out_comm, out_comm_no_delay : as BaseFilter
    out_dither : dither added to each mode (zero for integrator modes)
    out_design : (len(lqg_modes), 8 + n_g) design state of each LQG mode, updated
        every frame, one row per LQG mode; columns in DESIGN_COLUMNS:
        active (1 LQG, 0 integrator), predicted residual std, Kalman noise
        r_kalman, penalty rho, plant relative std at the last redesign,
        accepted / rejected / reverted counters, then the taps g_1..g_n_g of
        the active design (NaN where there is none). With free_theta: r_kalman is
        the added noise variance s q, plant_rel_std is NaN, n_rejected also counts
        rejected and aborted trials, and the taps are the first n_g samples of the
        impulse response of the plant implied by theta. With mimo_free_theta every row
        shows the joint design (pred_std is the total over the LQG modes, r_kalman the
        mean added noise variance) and each row the taps of its own mode's implied
        plant (diagonal of the transfer matrix). Store it with DataStore to
        keep the design history in the run folder.
    """

    DESIGN_COLUMNS = ['active', 'pred_std', 'r_kalman', 'rho', 'plant_rel_std',
                      'n_accepted', 'n_rejected', 'n_reverted']  # then g_1 .. g_n_g

    def __init__(self,
                 simul_params: SimulParams,
                 n_modes: int,
                 lqg_modes: list = (0, 1),
                 int_gain=0.4,
                 warmup_gain=None,
                 int_ff=1.0,
                 warmup_ff=None,
                 warmup_num: list = None,
                 warmup_den: list = None,
                 delay: float = 1,
                 n_g: int = 3,
                 p: int = 8,
                 plant_window: int = 4000,
                 dist_window: int = 2000,
                 min_samples: int = None,
                 dither_std=5.0,
                 noise_var=0.0,
                 redesign_every: int = 250,
                 ms_limit_db: float = 6.0,
                 gain_range: list = (0.5, 1.5),
                 delay_margin: float = 0.5,
                 plant_rel_std_max: float = 0.25,
                 max_radius: float = 0.9995,
                 supervisor: bool = True,
                 blowup_factor: float = 2.0,
                 fast_window: int = 50,
                 holdoff: int = 4,
                 method: str = 'structured',
                 n_theta: int = 11,
                 theta_window: int = None,
                 dither_after=None,
                 switch_designs: int = 2,
                 acceptance: str = 'residual',
                 accept_ratio: float = 1.05,
                 abort_ratio: float = 1.5,
                 hard_abort_ratio: float = 4.0,
                 mimo_structure: str = 'full',
                 theta_poles: list = None,
                 theta_refit_u: bool = False,
                 var_block_size: int = None,
                 plant_min_dither: float = 1.0,
                 corner_gains: bool = True,
                 var_margin: float = 0.02,
                 holdout: float = 0.25,
                 siso_first: int = 1,
                 bias_tau: float = 2.0,
                 bias_rel: float = 0.1,
                 rho_grid: list = None,
                 eps_grid: list = None,
                 eps_tau: float = 5e-3,
                 ident_modes: list = None,
                 ident_dither: float = 1.0,
                 ident_window: int = 4000,
                 ident_min_samples: int = None,
                 seed: int = 0,
                 log_file: str = None,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(nfilter=n_modes, delay=delay,
                         target_device_idx=target_device_idx, precision=precision)

        self.n_modes = int(n_modes)
        self.lqg_modes = [int(m) for m in lqg_modes]
        if any(m < 0 or m >= self.n_modes for m in self.lqg_modes):
            raise ValueError(f"lqg_modes {self.lqg_modes} out of range for n_modes {self.n_modes}")
        if len(set(self.lqg_modes)) != len(self.lqg_modes):
            raise ValueError("lqg_modes contains duplicates")

        self.int_gain = self._per_mode(int_gain, self.n_modes, 'int_gain')
        n_lqg = len(self.lqg_modes)
        warmup = (self.int_gain[self.lqg_modes] if warmup_gain is None
                  else self._per_mode(warmup_gain, n_lqg, 'warmup_gain'))
        self.int_ff = self._per_mode(int_ff, self.n_modes, 'int_ff')
        warmup_ff = (self.int_ff[self.lqg_modes] if warmup_ff is None
                     else self._per_mode(warmup_ff, n_lqg, 'warmup_ff'))
        if np.any(self.int_ff > 1) or np.any(self.int_ff <= 0) or np.any(warmup_ff > 1) or np.any(warmup_ff <= 0):
            raise ValueError("int_ff and warmup_ff must be in (0, 1]")
        if (warmup_num is not None or warmup_den is not None) and method != 'free_theta':
            raise ValueError("warmup_num / warmup_den are only supported with method 'free_theta'")
        warmup_iir = [(self._per_mode_rows(warmup_num, n_lqg, 'warmup_num')[i],
                       self._per_mode_rows(warmup_den, n_lqg, 'warmup_den')[i])
                      if warmup_num is not None else (None, None) for i in range(n_lqg)]
        dither = self._per_mode(dither_std, n_lqg, 'dither_std')
        noise = self._per_mode(noise_var, n_lqg, 'noise_var')
        self.log_file = log_file
        self.ident_modes = [] if ident_modes is None else [int(m) for m in ident_modes]
        if any(m < 0 or m >= self.n_modes for m in self.ident_modes):
            raise ValueError(f"ident_modes out of range for n_modes {self.n_modes}")
        if set(self.ident_modes) & set(self.lqg_modes):
            raise ValueError("ident_modes and lqg_modes overlap: a mode is either identified "
                             "and controlled by the LQG, or identified only")

        if method not in ('structured', 'free_theta', 'mimo_free_theta', 'var_lqg'):
            raise ValueError("method must be 'structured', 'free_theta', 'mimo_free_theta' or 'var_lqg', "
                             f"got {method!r}")
        self.method = method
        after = (dither if dither_after is None
                 else self._per_mode(dither_after, n_lqg, 'dither_after'))

        rng = np.random.default_rng(seed)
        self.mode_ctrl = []
        self._ctrl_rows = []                             # out_design rows of each controller
        if method == 'mimo_free_theta':
            self.mode_ctrl.append(AdaptiveMimoFreeTheta(
                dt=simul_params.time_step, m=n_lqg, N=n_theta,
                window=plant_window if theta_window is None else theta_window,
                min_samples=min_samples, structure=mimo_structure, dither_std=dither,
                dither_after=after, switch_designs=switch_designs, redesign_every=redesign_every,
                ms_limit_db=ms_limit_db, gain_range=gain_range, delay_margin=delay_margin,
                warmup_gain=warmup, warmup_ff=warmup_ff, acceptance=acceptance, accept_ratio=accept_ratio,
                abort_ratio=abort_ratio, hard_abort_ratio=hard_abort_ratio, n_taps_log=n_g, max_radius=max_radius, corners=corner_gains,
                supervisor=supervisor, blowup_factor=blowup_factor, fast_window=fast_window,
                holdoff=holdoff, poles=theta_poles, refit_u=theta_refit_u,
                rho_grid=tuple(rho_grid) if rho_grid else (0.0, 1.0, 10.0),
                eps_grid=tuple(eps_grid) if eps_grid else (0.0,),
                eps_tau=eps_tau if eps_grid else None,
                rng=np.random.default_rng(rng.integers(2**32)),
                name=f"modes {self.lqg_modes}"))
            self._ctrl_rows.append(list(range(n_lqg)))
        elif method == 'var_lqg':
            # one vector-AR controller on all the LQG modes, or one per block of
            # var_block_size consecutive LQG modes (identified and designed separately:
            # the design cost grows with the cube of the block, not of all the modes)
            bs = n_lqg if not var_block_size else int(var_block_size)
            self._blocks = [list(range(i0, min(i0 + bs, n_lqg))) for i0 in range(0, n_lqg, bs)]
            for blk in self._blocks:
                modes_b = [self.lqg_modes[q] for q in blk]
                self.mode_ctrl.append(AdaptiveVarLQG(
                    dt=simul_params.time_step, m=len(blk), n_g=n_g, p=p, window=plant_window,
                    min_samples=min_samples, dither_std=dither[blk], dither_after=after[blk],
                    switch_designs=switch_designs, redesign_every=redesign_every, ms_limit_db=ms_limit_db,
                    gain_range=gain_range, delay_margin=delay_margin, warmup_gain=warmup[blk],
                    warmup_ff=warmup_ff[blk], plant_min_dither=plant_min_dither,
                    var_margin=var_margin, holdout=holdout, siso_first=siso_first, acceptance=acceptance,
                    bias_tau=bias_tau, bias_rel=bias_rel, eps_tau=eps_tau,
                    rho_grid=tuple(rho_grid) if rho_grid else (0.0, 0.1, 0.3, 1.0, 10.0),
                    eps_grid=tuple(eps_grid) if eps_grid else (0.0, 0.1),
                    accept_ratio=accept_ratio, abort_ratio=abort_ratio, hard_abort_ratio=hard_abort_ratio, max_radius=max_radius,
                    corners=corner_gains, supervisor=supervisor, blowup_factor=blowup_factor,
                    fast_window=fast_window, holdoff=holdoff, rng=np.random.default_rng(rng.integers(2**32)),
                    name=f"modes {modes_b}"))
                self._ctrl_rows.append(list(blk))
        joint = method in ('mimo_free_theta', 'var_lqg')
        for i, m in enumerate(self.lqg_modes if not joint else []):
            mode_rng = np.random.default_rng(rng.integers(2**32))
            if method == 'structured':
                ctrl = AdaptiveModeLQG(dt=simul_params.time_step, n_g=n_g, p=p,
                                       plant_window=plant_window, dist_window=dist_window,
                                       min_samples=min_samples, dither_std=dither[i],
                                       redesign_every=redesign_every, ms_limit_db=ms_limit_db,
                                       gain_range=gain_range, delay_margin=delay_margin,
                                       plant_rel_std_max=plant_rel_std_max, warmup_gain=warmup[i], warmup_ff=warmup_ff[i],
                                       noise_var=noise[i], max_radius=max_radius,
                                       supervisor=supervisor, blowup_factor=blowup_factor,
                                       fast_window=fast_window, holdoff=holdoff,
                                       rng=mode_rng, name=f"mode {m}")
            else:
                ctrl = AdaptiveModeFreeTheta(dt=simul_params.time_step, N=n_theta,
                                             window=plant_window if theta_window is None else theta_window,
                                             min_samples=min_samples, dither_std=dither[i],
                                             dither_after=after[i], switch_designs=switch_designs,
                                             redesign_every=redesign_every, ms_limit_db=ms_limit_db,
                                             gain_range=gain_range, delay_margin=delay_margin,
                                             warmup_gain=warmup[i], warmup_ff=warmup_ff[i],
                                             warmup_num=warmup_iir[i][0], warmup_den=warmup_iir[i][1],
                                             acceptance=acceptance,
                                             accept_ratio=accept_ratio, abort_ratio=abort_ratio, hard_abort_ratio=hard_abort_ratio,
                                             n_taps_log=n_g, max_radius=max_radius,
                                             supervisor=supervisor,
                                             blowup_factor=blowup_factor, fast_window=fast_window,
                                             holdoff=holdoff, rng=mode_rng, name=f"mode {m}")
            self.mode_ctrl.append(ctrl)
            self._ctrl_rows.append([i])
        self._n_events = [0] * len(self.mode_ctrl)

        self.ident = None
        if self.ident_modes:
            self.ident = PlantIdentifier(
                len(self.ident_modes), n_g=n_g, p=p, window=int(ident_window),
                min_samples=ident_min_samples, period=redesign_every,
                dither_std=self._per_mode(ident_dither, len(self.ident_modes), 'ident_dither'),
                rng=np.random.default_rng(rng.integers(2 ** 32)),
                name=f"ident modes {self.ident_modes[0]}-{self.ident_modes[-1]}")

        self._int_mask = np.ones(self.n_modes, dtype=bool)
        self._int_mask[self.lqg_modes] = False
        self._u = np.zeros(self.n_modes)                 # CPU float64 command state

        self.out_dither = BaseValue(value=self.xp.zeros(self.n_modes, dtype=self.dtype),
                                    target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_dither'] = self.out_dither

        self._design = np.full((n_lqg, len(self.DESIGN_COLUMNS) + int(n_g)), np.nan)
        self._design[:, [0, 5, 6, 7]] = 0.0
        self.out_design = BaseValue(value=self.xp.asarray(self._design, dtype=self.dtype),
                                    target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_design'] = self.out_design

    @staticmethod
    def _per_mode(value, n, name):
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size == 1:
            return np.full(n, arr[0])
        if arr.size != n:
            raise ValueError(f"{name} must be a scalar or have {n} elements, got {arr.size}")
        return arr

    @staticmethod
    def _per_mode_rows(value, n, name):
        if value is None:
            return [None] * n
        rows = [np.asarray(r, dtype=float) for r in np.atleast_2d(np.asarray(value, dtype=float))]
        if len(rows) == 1:
            return rows * n
        if len(rows) != n:
            raise ValueError(f"{name} must be one row or {n} rows, got {len(rows)}")
        return rows

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({'out_dither': OutputDesc(BaseValue, 'Dither added to the commands of the LQG modes'),
                       'out_design': OutputDesc(BaseValue, 'Design state per LQG mode (see DESIGN_COLUMNS)')})
        return result

    def trigger_code(self):
        y = cpuArray(self.delta_comm).astype(float)
        gain_mod = cpuArray(self._gain_mod).astype(float)

        # integrator modes
        mask = self._int_mask
        self._u[mask] = self.int_ff[mask] * self._u[mask] + self.int_gain[mask] * gain_mod[mask] * y[mask]

        # adaptive LQG modes
        dither = np.zeros(self.n_modes)
        if self.method == 'var_lqg':
            for i, blk in enumerate(self._blocks):
                idx = [self.lqg_modes[q] for q in blk]
                self._u[idx], dither[idx] = self.mode_ctrl[i].step(y[idx])
                self._log_new_events(i)
        elif self.method == 'mimo_free_theta':
            lqg = self.lqg_modes
            self._u[lqg], dither[lqg] = self.mode_ctrl[0].step(y[lqg])
            self._log_new_events(0)
        else:
            for i, (m, ctrl) in enumerate(zip(self.lqg_modes, self.mode_ctrl)):
                self._u[m], dither[m] = ctrl.step(y[m])
                self._log_new_events(i)

        applied = self._u
        if self.ident is not None:
            # the dither perturbs the command that reaches the DM, not the integrator
            # state, so it is not accumulated by the integrator
            r = self.ident.dither()
            applied = self._u.copy()
            applied[self.ident_modes] += r
            dither[self.ident_modes] = r
            self.ident.record(y[self.ident_modes], applied[self.ident_modes], r)

        self.output_buffer[:, 0] = self.to_xp(applied, dtype=self.dtype)
        self.out_dither.value[:] = self.to_xp(dither, dtype=self.dtype)
        self.out_dither.generation_time = self.current_time
        self.out_design.value = self.to_xp(self._design, dtype=self.dtype)
        self.out_design.generation_time = self.current_time

    def _log_new_events(self, i):
        ctrl = self.mode_ctrl[i]
        new = ctrl.log[self._n_events[i]:]
        if new:
            self._update_design_row(i, new)
        for k, event, info in new:
            brief = {key: (round(v, 4) if isinstance(v, float) else v) for key, v in info.items()
                     if key in ('plant_rel_std', 'pred_std', 'rho', 'r_kalman', 'to', 'reason',
                                'trial_rms_ratio', 'after', 'structure', 'holdout_err_ar', 'holdout_err_var',
                                'eps', 'mu')}
            g = info.get('g')
            if g is not None:
                brief['g'] = np.round(np.asarray(g, float), 3).tolist()
            self.logger.info(f"DataDrivenLqg {ctrl.name} frame {k}: {event} {brief}")
        self._n_events[i] = len(ctrl.log)

    def _update_design_row(self, i, new_events):
        ctrl = self.mode_ctrl[i]
        n_cols = len(self.DESIGN_COLUMNS)
        for j, r in enumerate(self._ctrl_rows[i]):
            row = self._design[r]
            for _, event, info in new_events:
                if 'plant_rel_std' in info:
                    row[4] = info['plant_rel_std']
                col = {'accepted': 5, 'rejected': 6, 'trial rejected': 6, 'trial aborted': 6,
                       'reverted': 7}.get(event)
                if col is not None:
                    row[col] += 1
            d = ctrl.design
            if d is None:
                row[0] = 0.0
                row[1:4] = np.nan
                row[n_cols:] = np.nan
            else:
                row[0] = 1.0
                row[1:4] = d.predicted_output_std(), d.r_kalman, d.rho
                row[n_cols:] = d.g if np.ndim(d.g) == 1 else d.g[j]

    def reset_states(self):
        super().reset_states()
        self._u[:] = 0

    def finalize(self):
        super().finalize()
        if self.log_file is None:
            return
        events = {ctrl.name: [dict(frame=k, event=e, **info) for k, e, info in ctrl.log]
                  for ctrl in self.mode_ctrl}
        if self.ident is not None:
            events[self.ident.name] = [dict(frame=k, event=e, **info)
                                       for k, e, info in self.ident.log]
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(events, f, indent=1, default=float)
