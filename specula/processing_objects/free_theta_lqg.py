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
turbulence part), 'diag_ar' each mode's own measurements only (deadbeat register only).

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
residual is not worse than accept_ratio times the previous one.

Validated on the RAMA twin (LEO tracking, pyramid tip defect, modes 0-4, 40 s runs):
deadbeat register, full Theta, 20 s window, 5 nm dither, the defaults below:
106 nm rms on modes 0-4 once running, against 108 for DdLqg 'mimo' (vector AR) and 148
for the leaky integrator. Poles of Lambda at 0.5 gave the same result; slower poles make
the low-frequency plant ill-determined. With a 10 s window or the grid rho in {0, 1, 10}
it was clearly worse.

A thin layer on DataDrivenLqg (method 'mimo_free_theta'): same inputs, outputs and event
log. Library: specula.lib.adaptive_lqg_mimo.
"""
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.data_driven_lqg import DataDrivenLqg


class FreeThetaLqg(DataDrivenLqg):
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
        warm-up; int_ff < 1 is a leaky integrator.
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
    log_file : str, optional
        JSON file with the design events, written at the end.
    seed : int
    delay : float
        Delay of the output command [frames], as for Integrator.
    """

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
                 log_file: str = None,
                 seed: int = 0,
                 delay: float = 1,
                 target_device_idx: int = None,
                 precision: int = None):
        structure = str(structure).lower()
        if structure not in ('full', 'diag_plant', 'diag_ar'):
            raise ValueError(f"structure must be 'full', 'diag_plant' or 'diag_ar', got {structure!r}")
        if poles is not None:
            poles = [float(p) for p in poles]
            if structure == 'diag_ar':
                raise ValueError("structure 'diag_ar' needs the time-shift register (poles=None)")
            n_theta = len(poles)
        dt = simul_params.time_step
        frames = lambda s: max(int(round(float(s) / dt)), 1)
        train, redesign = frames(train_s), frames(redesign_s)
        window = frames(window_s) if window_s is not None else train
        if window < train:
            raise ValueError("window_s must be >= train_s: the first design needs a full window")
        super().__init__(simul_params=simul_params, n_modes=n_modes, lqg_modes=list(lqg_modes),
                         int_gain=int_gain, int_ff=int_ff, delay=delay,
                         method='mimo_free_theta', mimo_structure=structure, n_theta=int(n_theta),
                         theta_poles=poles, theta_refit_u=refit_u, plant_window=window,
                         theta_window=window, min_samples=train, dither_std=dither_std,
                         dither_after=dither_std if dither_after is None else dither_after,
                         switch_designs=switch_designs, redesign_every=redesign,
                         rho_grid=list(rho_grid), eps_grid=list(eps_grid), eps_tau=eps_tau,
                         ms_limit_db=ms_limit_db, gain_range=list(gain_range),
                         delay_margin=delay_margin, corner_gains=corner_gains,
                         max_radius=max_radius, acceptance='residual',
                         accept_ratio=accept_ratio, abort_ratio=abort_ratio,
                         hard_abort_ratio=hard_abort_ratio, seed=seed, log_file=log_file,
                         target_device_idx=target_device_idx, precision=precision)
        self.structure = structure
