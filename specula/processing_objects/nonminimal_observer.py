import json
import os

import numpy as np

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.lib.nonminimal_observer import MimoObserver, ModeObserver


class NonminimalObserver(BaseProcessingObj):
    """
    Plant + disturbance observer in nonminimal (Bosso-Borghesi) form on selected modes,
    from a free theta fitted in the loop (specula.lib.nonminimal_observer).

    It only observes: it reads the modal measurement and the applied command and
    does not act on the loop. For each mode in `modes`:

      1. filters xi+ = Lambda xi + l y, omega+ = Lambda omega + l u of order n_theta,
         spec(Lambda) = filter_poles (default: n_g zeros for the command history,
         n_theta - n_g at slow_pole). spec(Lambda) is the observer spectrum: slow poles
         make an observer that does not follow the measurement noise.
      2. theta fitted on frames [fit_start, fit_start + fit_frames), then frozen:
           fit='iv' (default): theta = (g, theta_d). Plant taps g by instrumental variables,
             the dither (in_r) as instrument; theta_d by least squares on
             d_meas = y - g^T x = theta_d^T xi_d + eps, xi_d the disturbance filters (the
             non-zero poles of Lambda) driven by d_meas. Needs in_r.
           fit='ls': least squares on y_k = theta^T zeta_k + eps_k, all 2 n_theta entries
             free. Under a fixed controller the plant gain it implies is not pinned down.
      3. d_hat_{k|k-1}: theta_d^T xi_d (iv) or theta^T zeta_k - g^T x_k (ls),
         x_k = (u_{k-1} .. u_{k-n_g})

    mimo=True (fit='iv' only): the modes are observed jointly (MimoObserver): diagonal plant
    as above, but each mode's disturbance is predicted from the filter states of all the
    modes (vector disturbance model: frozen flow couples the modes at nonzero lags).
    cold_start=True resets the filters to zero when theta is frozen, so the output shows the
    observer transient (otherwise the filters have run since the start and the output is
    converged when it appears).
    The plant is identifiable only if the command carries a dither (DataDrivenLqg on the
    same modes provides one).

    Inputs
    ------
    in_y : modal measurement y_k (rec.out_modes)
    in_u : command applied at frame k, dither included (control.out_comm_no_delay)
    in_r : dither in that command (control.out_dither); required for fit='iv'

    Outputs (one entry per mode in `modes`, NaN before theta)
    -------
    out_d_pred : d_hat_{k|k-1}, observer output (disturbance, data up to k-1)
    out_d_filt : d_hat_{k|k} = y_k - g^T x_k (reconstruction, no model)
    out_y_pred : y_hat_{k|k-1} = theta^T zeta_k
    out_d_kal  : d_hat_{k|k}, Kalman update of d_hat_{k|k-1} with d_meas_k (gain from the fit
                 window: 1 - R/q, R the white sensor noise from the autocovariance of d_meas)
    out_d_pred2: d_hat_{k+2|k}, two-step prediction (fit='iv' / mimo; NaN for 'ls')
    out_theta  : fitted theta, (len(modes), n_theta) = (g, theta_d) for 'iv',
                 (len(modes), 2 n_theta) for 'ls', (len(modes), n_g + len(modes) p) = (g, row
                 of Theta_d) for mimo, p = n_theta - n_g
    out_g      : (len(modes), n_g) plant taps implied by theta

    d is in the units of in_y: the disturbance as the sensor sees it, i.e. the
    open-loop modal coefficient times the modal optical gain -sum(g).
    """

    def __init__(self,
                 modes: list = (0, 1),
                 n_theta: int = 11,
                 n_g: int = 3,
                 slow_pole: float = 0.9,
                 filter_poles: list = None,
                 fit_start: int = 500,
                 fit_frames: int = 6000,
                 fit: str = 'iv',
                 cold_start: bool = False,
                 mimo: bool = False,
                 log_file: str = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.modes = [int(m) for m in modes]
        self.n_theta, self.n_g = int(n_theta), int(n_g)
        self.log_file = log_file
        if mimo and fit != 'iv':
            raise ValueError("mimo requires fit='iv'")
        self.mimo = bool(mimo)
        if self.mimo:
            self.obs = MimoObserver(len(self.modes), N=n_theta, n_g=n_g, slow_pole=slow_pole,
                                    poles=filter_poles, fit_start=fit_start, fit_frames=fit_frames,
                                    cold_start=cold_start)
        else:
            self.obs = [ModeObserver(N=n_theta, n_g=n_g, slow_pole=slow_pole, poles=filter_poles,
                                     fit_start=fit_start, fit_frames=fit_frames, fit=fit,
                                     cold_start=cold_start)
                        for _ in self.modes]

        self.inputs['in_y'] = InputValue(type=BaseValue)
        self.inputs['in_u'] = InputValue(type=BaseValue)
        self.inputs['in_r'] = InputValue(type=BaseValue, optional=fit != 'iv')
        n_par = (self.n_g + len(self.modes) * (self.n_theta - self.n_g) if mimo
                 else self.n_theta if fit == 'iv' else 2 * self.n_theta)
        n = len(self.modes)
        self.out_d_pred = self._value(np.full(n, np.nan))
        self.out_d_filt = self._value(np.full(n, np.nan))
        self.out_y_pred = self._value(np.full(n, np.nan))
        self.out_d_kal = self._value(np.full(n, np.nan))
        self.out_d_pred2 = self._value(np.full(n, np.nan))
        self.out_theta = self._value(np.full((n, n_par), np.nan))
        self.out_g = self._value(np.full((n, self.n_g), np.nan))
        for name in ('out_d_pred', 'out_d_filt', 'out_y_pred', 'out_d_kal', 'out_d_pred2',
                     'out_theta', 'out_g'):
            self.outputs[name] = getattr(self, name)

    def _value(self, arr):
        return BaseValue(value=self.xp.asarray(arr, dtype=self.dtype),
                         target_device_idx=self.target_device_idx, precision=self.precision)

    @classmethod
    def input_names(cls):
        return {'in_y': InputDesc(BaseValue, 'Modal measurement'),
                'in_u': InputDesc(BaseValue, 'Command applied at this frame (dither included)'),
                'in_r': InputDesc(BaseValue, 'Dither in that command (instrument of the iv fit)')}

    @classmethod
    def output_names(cls):
        return {'out_d_pred': OutputDesc(BaseValue, 'One-step prediction of the disturbance'),
                'out_d_filt': OutputDesc(BaseValue, 'Filtered disturbance estimate'),
                'out_y_pred': OutputDesc(BaseValue, 'One-step prediction of the measurement'),
                'out_d_kal': OutputDesc(BaseValue, 'Kalman-updated disturbance estimate d_hat_{k|k}'),
                'out_d_pred2': OutputDesc(BaseValue, 'Two-step prediction d_hat_{k+2|k}'),
                'out_theta': OutputDesc(BaseValue, 'Fitted free theta per mode'),
                'out_g': OutputDesc(BaseValue, 'Plant FIR taps per mode')}

    def trigger_code(self):
        y = cpuArray(self.local_inputs['in_y'].value).astype(float).ravel()
        u = cpuArray(self.local_inputs['in_u'].value).astype(float).ravel()
        r_in = self.local_inputs['in_r']
        r = cpuArray(r_in.value).astype(float).ravel() if r_in is not None else np.zeros_like(u)
        if self.mimo:
            o = self.obs
            est = np.column_stack(o.step(y[self.modes], u[self.modes], r[self.modes]))
            theta = (np.hstack([o.G, o.Theta_d]) if o.Theta_d is not None
                     else np.full(self.out_theta.value.shape, np.nan))
            g = o.G if o.G is not None else np.full((len(self.modes), self.n_g), np.nan)
        else:
            est = np.array([o.step(y[m], u[m], r[m]) for m, o in zip(self.modes, self.obs)])
            theta = np.array([o.theta if o.theta is not None
                              else np.full(self.out_theta.value.shape[1], np.nan) for o in self.obs])
            g = np.array([o.g if o.g is not None else np.full(self.n_g, np.nan) for o in self.obs])
        for out, val in ((self.out_d_pred, est[:, 0]), (self.out_d_filt, est[:, 1]),
                         (self.out_y_pred, est[:, 2]), (self.out_d_kal, est[:, 3]),
                         (self.out_d_pred2, est[:, 4]), (self.out_theta, theta), (self.out_g, g)):
            out.value = self.xp.asarray(val, dtype=self.dtype)
            out.generation_time = self.current_time

    def finalize(self):
        super().finalize()
        if self.log_file is None:
            return
        if self.mimo:
            events = {f"modes {self.modes}": [dict(frame=k, event=e, **info)
                                              for k, e, info in self.obs.log]}
        else:
            events = {f"mode {m}": [dict(frame=k, event=e, **info) for k, e, info in o.log]
                      for m, o in zip(self.modes, self.obs)}
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(events, f, indent=1, default=float)
