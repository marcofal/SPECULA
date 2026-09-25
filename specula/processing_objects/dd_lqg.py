"""DdLqg: the data-driven LQG with a compact interface for its two methods.

    control:
      class:        'DdLqg'
      simul_params_ref: 'main'
      method:       'mimo'              # or 'siso'
      n_modes:      83
      lqg_modes:    [0, 1, 2, 3, 4]     # the others stay on the (leaky) integrator
      int_gain:     0.6
      int_ff:       0.9                 # leaky integrator, also the warm-up
      train_s:      10.0                # data before the first design [s]
      redesign_s:   1.0                 # period of the redesigns and of the trials [s]
      dither_std:   3.0                 # [units of delta_comm], see dither_design
      inputs:
        delta_comm: 'rec.out_modes'

A thin layer on DataDrivenLqg (same inputs, outputs and event log), which stays available
with all its parameters; DdLqg only picks the two methods validated on the RAMA twin,
expresses the times in seconds and sets the defaults found in that campaign.

Methods
-------
'siso'  one controller per mode, output equation y_k = theta^T zeta_k + e_k on the
        time-shift register zeta_k = (y_{k-1..k-N}, u_{k-1..k-N}), theta free, fitted by
        sliding-window least squares (DataDrivenLqg method 'free_theta'). The dither is
        on until `switch_designs` designs are accepted, then `dither_after` (0 by default).
'mimo'  all the LQG modes jointly, or in blocks of `block_size` consecutive modes:
        diagonal FIR plant Gamma(q) identified by instrumental variables on the dither,
        vector AR of order p on the pseudo open loop y - Gamma(q) u, Kalman + LQR on the
        minimal state (DataDrivenLqg method 'var_lqg'). The dither stays on by default,
        since the plant is identified through it at every redesign.

Scheduling
----------
The warm-up integrator (int_gain, int_ff) runs until the first design. Designs happen at
the end of each redesign period once train_s seconds of data are in the window: the first
design is at the first multiple of redesign_s that is >= train_s. A new design runs on
trial for one period and is kept if the residual of that period is at most accept_ratio
times the one of the last clean period, and dropped at once if a single frame exceeds
hard_abort_ratio times it. The identification window is window_s (train_s by default).

Dither
------
The plant is identified through the dither, and its accuracy scales as sigma_e /
(dither sqrt(window)): specula.lib.dither_design.suggest_dither_std returns the smallest
dither per mode for a target accuracy, from a stretch of closed-loop data. On the RAMA
twin with the pyramid tip defect: LEO tracking 4-7 nm on modes 0-4 and ~2 nm on the higher
ones for a 6 s window; star observation ~1 nm per mode.
"""
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.data_driven_lqg import DataDrivenLqg


class DdLqg(DataDrivenLqg):
    """Data-driven LQG, method 'siso' or 'mimo' (see the module docstring).

    Parameters
    ----------
    simul_params : SimulParams
    n_modes : int
        Number of modes of delta_comm.
    method : {'mimo', 'siso'}
    lqg_modes : list[int]
        Modes controlled by the data-driven LQG; the others run the integrator.
    int_gain, int_ff : float or list[float]
        Gain and forgetting factor of the integrator of the other modes and of the warm-up
        (u_k = int_ff u_{k-1} + int_gain y_k). int_ff < 1 is a leaky integrator.
    train_s : float
        Seconds of data before the first design.
    window_s : float, optional
        Identification window [s] (default train_s).
    redesign_s : float
        Redesign period [s], also the length of the trial of a new design.
    dither_std : float or list[float]
        Dither std per LQG mode, in the units of delta_comm.
    dither_after : float or list[float], optional
        Dither after `switch_designs` accepted designs. Default: unchanged for 'mimo',
        0 for 'siso'.
    switch_designs : int
        Accepted designs before the dither goes to dither_after.
    n_g, p : int
        'mimo': plant FIR taps and order of the vector AR.
    block_size : int, optional
        'mimo': identify and design in blocks of this many consecutive LQG modes (the
        design cost grows with the cube of the block). Default: one block.
    rho_grid, eps_grid : list[float]
        'mimo': grid of the penalties on the command increment and on the command around
        its running mean (time constant eps_tau [s]).
    bias_tau : float
        'mimo': time constant [s] of the DC state of the disturbance model.
    n_theta : int
        'siso': length N of each half of the time-shift register (2N parameters).
    ms_limit_db, gain_range, delay_margin
        Checks of every design on the identified plant: peak sensitivity, stability for
        plant gains scaled by gain_range and with delay_margin frames of extra delay.
    plant_min_dither : float
        'mimo': a mode whose dither std in the window is below this value keeps its
        previous plant estimate (units of delta_comm). Lower it with a dither below 1.
    corner_gains : bool
        'mimo': check also per-mode gain corners (random subset beyond 4 modes); set
        False for very large blocks.
    accept_ratio, abort_ratio, hard_abort_ratio : float
        Acceptance of a trial (end of period), abort on the fast residual, abort on a
        single frame, as multiples of the residual of the last clean period.
    log_file : str, optional
        JSON file with the design events of every controller, written at the end.
    seed : int
    delay : float
        Delay of the output command [frames], as for Integrator.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 n_modes: int,
                 method: str = 'mimo',
                 lqg_modes: list = (0, 1, 2, 3, 4),
                 int_gain=0.6,
                 int_ff=1.0,
                 train_s: float = 10.0,
                 window_s: float = None,
                 redesign_s: float = 1.0,
                 dither_std=3.0,
                 dither_after=None,
                 switch_designs: int = 2,
                 n_g: int = 3,
                 p: int = 8,
                 block_size: int = None,
                 rho_grid: list = (0.0, 0.1, 0.3, 1.0, 10.0),
                 eps_grid: list = (0.0, 0.1),
                 eps_tau: float = 5e-3,
                 bias_tau: float = 2.0,
                 n_theta: int = 11,
                 ms_limit_db: float = 6.0,
                 gain_range: list = (0.5, 1.5),
                 delay_margin: float = 0.5,
                 corner_gains: bool = True,
                 plant_min_dither: float = 1.0,
                 accept_ratio: float = 1.05,
                 abort_ratio: float = 1.5,
                 hard_abort_ratio: float = 4.0,
                 log_file: str = None,
                 seed: int = 0,
                 delay: float = 1,
                 target_device_idx: int = None,
                 precision: int = None):
        method = str(method).lower()
        if method not in ('siso', 'mimo'):
            raise ValueError(f"method must be 'siso' or 'mimo', got {method!r}")
        dt = simul_params.time_step
        frames = lambda s: max(int(round(float(s) / dt)), 1)
        train, redesign = frames(train_s), frames(redesign_s)
        window = frames(window_s) if window_s is not None else train
        if window < train and method == 'mimo':
            raise ValueError("window_s must be >= train_s: the first design needs a full window")
        if dither_after is None:
            dither_after = dither_std if method == 'mimo' else 0.0
        common = dict(simul_params=simul_params, n_modes=n_modes, lqg_modes=list(lqg_modes),
                      int_gain=int_gain, int_ff=int_ff, delay=delay, n_g=n_g,
                      min_samples=train, dither_std=dither_std, dither_after=dither_after,
                      switch_designs=switch_designs, redesign_every=redesign,
                      ms_limit_db=ms_limit_db, gain_range=list(gain_range),
                      delay_margin=delay_margin, acceptance='residual',
                      accept_ratio=accept_ratio, abort_ratio=abort_ratio,
                      hard_abort_ratio=hard_abort_ratio, corner_gains=corner_gains,
                      seed=seed, log_file=log_file, target_device_idx=target_device_idx,
                      precision=precision)
        if method == 'mimo':
            super().__init__(method='var_lqg', p=p, plant_window=window, dist_window=window,
                             var_block_size=block_size, rho_grid=list(rho_grid),
                             eps_grid=list(eps_grid), eps_tau=eps_tau, bias_tau=bias_tau,
                             plant_min_dither=plant_min_dither,
                             **common)
        else:
            super().__init__(method='free_theta', n_theta=n_theta, plant_window=window,
                             theta_window=window, **common)
        self.dd_method = method
