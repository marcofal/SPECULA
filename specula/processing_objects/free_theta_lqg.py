"""FreeThetaLqg: MIMO LQG on a free Theta, nonminimal (Bosso-Borghesi) output equation.

    control:
      class:        'FreeThetaLqg'
      simul_params_ref: 'main'
      n_modes:      83
      lqg_modes:    [0, 1, 2, 3, 4]     # the others stay on the (leaky) integrator
      int_gain:     0.6
      int_ff:       0.9                 # leaky integrator, also the warm-up
      train_s:      20.0                # data before the first design [s]
      redesign_s:   2.0                 # period of the redesigns and of the trials [s]
      dither_std:   5.0                 # [units of delta_comm], always on
      inputs:
        delta_comm: 'rec.out_modes'

Model
-----
The m LQG modes jointly. Filters of the measurements and of the commands, per mode,

    xi+ = Lambda xi + l y,   omega+ = Lambda omega + l u,   zeta = (xi, omega) in R^{2Nm}

(Lambda, l) = the time-shift register by default (deadbeat: zeta holds the last N
measurements and commands), or the stable cascade of nonminimal_observer.cascade_filter
with the given `poles`. Output equation, linear in Theta:

    y_k = Theta^T zeta_k + e_k,   cov(e) = Sigma

With structure 'full' (default) every entry of Theta is free: A(q) y = B(q) u + e with
A and B full, the plant A^-1 B and the turbulence A^-1 e share the denominator, the
optical gains and any cross-talk of the plant are inside Theta, and no plant/turbulence
split is imposed. 'diag_plant' keeps each mode's own commands only (coupling in the
turbulence part), 'diag_ar' each mode's own measurements only (time-shift register only).

Theta is fitted by sliding-window least squares on (zeta_k, y_k) over window_s. The dither
makes the command part identifiable in closed loop and stays on. Before each design the
poles of the model outside the unit circle are moved onto max_radius, and the command
part is refitted with the measurement part fixed (refit_u).

Design
------
Innovations model zeta+ = (F + G_y Theta^T) zeta + G_u u + G_y e, y = Theta^T zeta + e;
Kalman with the measurement noise inflated to (1 + s) Sigma (s = 0: zeta itself is the
state estimate); LQR on sum |y|^2 + rho |u_k - u_{k-1}|^2 + eps |u_k - ubar_k|^2 with the
newest measurement. (s, rho, eps) on a grid: the smallest predicted residual among the
designs that pass the checks on the model plant (Ms, gain range with per-mode corners,
extra delay). Every new design runs on trial for one redesign period and is kept if its
residual is not worse than accept_ratio times the previous one; a supervisor reverts to
the last good controller if the residual blows up.

Before the first accepted design the LQG modes run the integrator of the other modes
(int_gain, int_ff); gain_mod scales the integrator modes only.

Validated on the RAMA twin (LEO tracking, pyramid tip defect, modes 0-4, 40 s runs):
time-shift register, full Theta, 20 s window, 5 nm dither, the defaults below: 106 nm rms
on modes 0-4 once running, against 108 for DdLqg 'mimo' (vector AR) and 148 for the leaky
integrator.

Algorithm: specula.lib.adaptive_lqg_mimo.AdaptiveMimoFreeTheta.
"""
import json
import os

import numpy as np

from specula import cpuArray
from specula.base_processing_obj import OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.adaptive_lqg_mimo import AdaptiveMimoFreeTheta
from specula.processing_objects.base_filter import BaseFilter


class FreeThetaLqg(BaseFilter):
    """MIMO LQG on a free Theta, y_k = Theta^T zeta_k + e_k (see the module docstring).

    Parameters
    ----------
    simul_params : SimulParams
    n_modes : int
        Number of modes of delta_comm.
    lqg_modes : list[int]
        Modes controlled jointly by the LQG; the others run the integrator.
    structure : {'full', 'diag_plant', 'diag_ar'}
        Free entries of Theta: all, each mode's own commands only, each mode's own
        measurements only.
    n_theta : int
        N, filter length per mode (2 N m entries in zeta). Ignored when `poles` is given.
    poles : list[float], optional
        Poles of the cascade filters (Lambda, l), real, inside the unit circle; N = len.
        Default: the time-shift register (all poles at 0).
    int_gain, int_ff : float or list[float]
        Integrator u_k = int_ff u_{k-1} + int_gain y_k of the other modes and of the
        warm-up (scalar or one value per mode of delta_comm); int_ff < 1 is leaky.
    train_s : float
        Seconds of data before the first design.
    window_s : float, optional
        Least-squares window [s], at least train_s (default train_s).
    redesign_s : float
        Redesign period [s], also the length of the trial of a new design.
    dither_std : float or list[float]
        Dither std per LQG mode, units of delta_comm.
    dither_after : float or list[float], optional
        Dither after `switch_designs` accepted designs (default: unchanged).
    switch_designs : int
    refit_u : bool
        Refit the command part of Theta after moving poles of the model.
    rho_grid, eps_grid : list[float]
        Penalties on the command increment and on the command around its running mean
        (time constant eps_tau [s]); the Kalman inflation s is searched on a fixed grid.
    ms_limit_db, gain_range, delay_margin, corner_gains
        Checks of every design on the model plant.
    max_radius : float
        Poles of the model beyond this radius are moved onto it.
    accept_ratio, abort_ratio, hard_abort_ratio : float
        Acceptance of a trial, abort on the fast residual, abort on a single frame, as
        multiples of the residual of the last clean period.
    n_taps : int
        Samples of the implied plant impulse response in out_design and in the log.
    log_file : str, optional
        JSON file with the design events, written at the end.
    seed : int
    delay : float
        Delay of the output command [frames], as for Integrator.
    """

    DESIGN_COLUMNS = ['active', 'pred_std', 's', 'rho', 'eps', 'n_accepted', 'n_rejected', 'n_reverted']
    _EVENT_COLUMN = {'accepted': 5, 'rejected': 6, 'trial rejected': 6, 'trial aborted': 6, 'reverted': 7}

    def __init__(self,
                 simul_params: SimulParams,
                 n_modes: int,
                 lqg_modes: list = (0, 1, 2, 3, 4),
                 structure: str = 'full',
                 n_theta: int = 11,
                 poles: list = None,
                 int_gain=0.6,
                 int_ff=1.0,
                 train_s: float = 20.0,
                 window_s: float = None,
                 redesign_s: float = 2.0,
                 dither_std=5.0,
                 dither_after=None,
                 switch_designs: int = 2,
                 refit_u: bool = True,
                 rho_grid: list = (0.0, 0.1, 0.3, 1.0, 10.0),
                 eps_grid: list = (0.0, 0.1),
                 eps_tau: float = 5e-3,
                 ms_limit_db: float = 6.0,
                 gain_range: list = (0.5, 1.5),
                 delay_margin: float = 0.5,
                 corner_gains: bool = True,
                 max_radius: float = 0.9995,
                 accept_ratio: float = 1.05,
                 abort_ratio: float = 1.5,
                 hard_abort_ratio: float = 4.0,
                 n_taps: int = 3,
                 log_file: str = None,
                 seed: int = 0,
                 delay: float = 1,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(nfilter=n_modes, delay=delay,
                         target_device_idx=target_device_idx, precision=precision)

        self.n_modes = int(n_modes)
        self.lqg_modes = [int(m) for m in lqg_modes]
        if not self.lqg_modes:
            raise ValueError("lqg_modes is empty")
        if any(m < 0 or m >= self.n_modes for m in self.lqg_modes):
            raise ValueError(f"lqg_modes {self.lqg_modes} out of range for n_modes {self.n_modes}")
        if len(set(self.lqg_modes)) != len(self.lqg_modes):
            raise ValueError("lqg_modes contains duplicates")
        self.structure = str(structure).lower()
        if self.structure not in ('full', 'diag_plant', 'diag_ar'):
            raise ValueError(f"structure must be 'full', 'diag_plant' or 'diag_ar', got {structure!r}")
        if poles is not None:
            poles = [float(p) for p in poles]
            if self.structure == 'diag_ar':
                raise ValueError("structure 'diag_ar' needs the time-shift register (poles=None)")
        self.int_gain = self._per_mode(int_gain, self.n_modes, 'int_gain')
        self.int_ff = self._per_mode(int_ff, self.n_modes, 'int_ff')
        if np.any(self.int_ff > 1) or np.any(self.int_ff <= 0):
            raise ValueError("int_ff must be in (0, 1]")

        dt = simul_params.time_step
        frames = lambda s: max(int(round(float(s) / dt)), 1)
        train, redesign = frames(train_s), frames(redesign_s)
        window = frames(window_s) if window_s is not None else train
        if window < train:
            raise ValueError("window_s must be >= train_s: the first design needs a full window")

        m = len(self.lqg_modes)
        dither = self._per_mode(dither_std, m, 'dither_std')
        after = dither if dither_after is None else self._per_mode(dither_after, m, 'dither_after')
        lqg = self.lqg_modes
        rng = np.random.default_rng(seed)
        self.ctrl = AdaptiveMimoFreeTheta(
            dt=dt, m=m, N=int(n_theta), window=window, min_samples=train, structure=self.structure,
            dither_std=dither, dither_after=after, switch_designs=switch_designs,
            redesign_every=redesign, ms_limit_db=ms_limit_db, gain_range=tuple(gain_range),
            delay_margin=delay_margin, warmup_gain=self.int_gain[lqg], warmup_ff=self.int_ff[lqg],
            acceptance='residual', accept_ratio=accept_ratio, abort_ratio=abort_ratio,
            hard_abort_ratio=hard_abort_ratio, n_taps_log=n_taps, max_radius=max_radius,
            corners=corner_gains, poles=poles, refit_u=refit_u, rho_grid=tuple(rho_grid),
            eps_grid=tuple(eps_grid), eps_tau=eps_tau if any(eps_grid) else None,
            rng=np.random.default_rng(rng.integers(2**32)), name=f"modes {lqg}")
        self.log_file = log_file
        self._n_events = 0

        self._int_mask = np.ones(self.n_modes, dtype=bool)
        self._int_mask[lqg] = False
        self._u = np.zeros(self.n_modes)                 # CPU float64 command state

        self.out_dither = BaseValue(value=self.xp.zeros(self.n_modes, dtype=self.dtype),
                                    target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_dither'] = self.out_dither
        self._design = np.zeros((m, len(self.DESIGN_COLUMNS) + int(n_taps)))
        self._design[:, 1:5] = np.nan
        self._design[:, len(self.DESIGN_COLUMNS):] = np.nan
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

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({'out_dither': OutputDesc(BaseValue, 'Dither added to the commands of the LQG modes'),
                       'out_design': OutputDesc(BaseValue, 'Design state per LQG mode: DESIGN_COLUMNS, '
                                                           'then the implied plant taps')})
        return result

    def trigger_code(self):
        y = cpuArray(self.delta_comm).astype(float)
        gain_mod = cpuArray(self._gain_mod).astype(float)
        mask = self._int_mask
        self._u[mask] = self.int_ff[mask] * self._u[mask] + self.int_gain[mask] * gain_mod[mask] * y[mask]
        dither = np.zeros(self.n_modes)
        self._u[self.lqg_modes], dither[self.lqg_modes] = self.ctrl.step(y[self.lqg_modes])
        self._new_events()

        self.output_buffer[:, 0] = self.to_xp(self._u, dtype=self.dtype)
        self.out_dither.value[:] = self.to_xp(dither, dtype=self.dtype)
        self.out_dither.generation_time = self.current_time
        self.out_design.value = self.to_xp(self._design, dtype=self.dtype)
        self.out_design.generation_time = self.current_time

    def _new_events(self):
        new = self.ctrl.log[self._n_events:]
        if not new:
            return
        self._n_events = len(self.ctrl.log)
        for k, event, info in new:
            col = self._EVENT_COLUMN.get(event)
            if col is not None:
                self._design[:, col] += 1
            brief = {key: (round(v, 4) if isinstance(v, float) else v) for key, v in info.items()
                     if key in ('pred_std', 's', 'rho', 'eps', 'reason', 'trial_rms_ratio', 'after',
                                'roots_stabilized', 'to')}
            self.logger.info(f"FreeThetaLqg {self.ctrl.name} frame {k}: {event} {brief}")
        d = self.ctrl.design
        n_cols = len(self.DESIGN_COLUMNS)
        if d is None:
            self._design[:, 0] = 0.0
            self._design[:, 1:5] = np.nan
            self._design[:, n_cols:] = np.nan
        else:
            self._design[:, 0] = 1.0
            self._design[:, 1:5] = d.predicted_output_std(), d.s, d.rho, d.eps
            self._design[:, n_cols:] = d.g

    def reset_states(self):
        super().reset_states()
        self._u[:] = 0

    def finalize(self):
        super().finalize()
        if self.log_file is None:
            return
        events = {self.ctrl.name: [dict(frame=k, event=e, **info) for k, e, info in self.ctrl.log]}
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(events, f, indent=1, default=float)
