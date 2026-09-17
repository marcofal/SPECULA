import json

import numpy as np

from specula import cpuArray
from specula.base_processing_obj import OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.adaptive_lqg import AdaptiveModeLQG
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
        the active design (NaN where there is none). Store it with DataStore to
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
        dither = self._per_mode(dither_std, n_lqg, 'dither_std')
        noise = self._per_mode(noise_var, n_lqg, 'noise_var')
        self.log_file = log_file

        rng = np.random.default_rng(seed)
        self.mode_ctrl = [
            AdaptiveModeLQG(dt=simul_params.time_step, n_g=n_g, p=p,
                            plant_window=plant_window, dist_window=dist_window,
                            min_samples=min_samples, dither_std=dither[i],
                            redesign_every=redesign_every, ms_limit_db=ms_limit_db,
                            gain_range=gain_range, delay_margin=delay_margin,
                            plant_rel_std_max=plant_rel_std_max, warmup_gain=warmup[i],
                            noise_var=noise[i], max_radius=max_radius,
                            supervisor=supervisor, blowup_factor=blowup_factor,
                            fast_window=fast_window, holdoff=holdoff,
                            rng=np.random.default_rng(rng.integers(2**32)),
                            name=f"mode {m}")
            for i, m in enumerate(self.lqg_modes)]
        self._n_events = [0] * n_lqg

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
        self._u[mask] += self.int_gain[mask] * gain_mod[mask] * y[mask]

        # adaptive LQG modes
        dither = np.zeros(self.n_modes)
        for i, (m, ctrl) in enumerate(zip(self.lqg_modes, self.mode_ctrl)):
            self._u[m], dither[m] = ctrl.step(y[m])
            self._log_new_events(i)

        self.output_buffer[:, 0] = self.to_xp(self._u, dtype=self.dtype)
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
                     if key in ('plant_rel_std', 'pred_std', 'rho', 'r_kalman', 'to', 'reason')}
            g = info.get('g')
            if g is not None:
                brief['g'] = [round(x, 3) for x in g]
            self.logger.info(f"DataDrivenLqg {ctrl.name} frame {k}: {event} {brief}")
        self._n_events[i] = len(ctrl.log)

    def _update_design_row(self, i, new_events):
        row, ctrl = self._design[i], self.mode_ctrl[i]
        n_cols = len(self.DESIGN_COLUMNS)
        for _, event, info in new_events:
            if 'plant_rel_std' in info:
                row[4] = info['plant_rel_std']
            col = {'accepted': 5, 'rejected': 6, 'reverted': 7}.get(event)
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
            row[n_cols:] = d.g

    def reset_states(self):
        super().reset_states()
        self._u[:] = 0

    def finalize(self):
        super().finalize()
        if self.log_file is None:
            return
        events = {ctrl.name: [dict(frame=k, event=e, **info) for k, e, info in ctrl.log]
                  for ctrl in self.mode_ctrl}
        with open(self.log_file, 'w') as f:
            json.dump(events, f, indent=1, default=float)
