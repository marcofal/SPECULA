from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams


class AVC(BaseProcessingObj):
    """
    Active Vibration Cancellation (AVC) processing object.

    Implements the adaptive vibration cancellation algorithm described in
    Muradore, Pettazzi & Fedrigo, "Adaptive Vibration Cancellation in
    Adaptive Optics: an Experimental Validation" (ECC 2014), as adapted for
    ESO's HRTC (see ``initAVC.m`` / ``updateAVC.m``).

    There is one AVC instance per vibration: each instance tracks the
    frequency, phase and amplitude of a single sinusoidal disturbance
    superimposed on a scalar measurement (typically the residual of one
    modal coefficient), and generates a counter-vibration command that,
    once added to the loop command, cancels it.

    This object vectorizes an arbitrary number ``n_avc`` of independent AVC
    instances: ``in_measurement`` and ``out_comm`` are both vectors of
    length ``n_avc``, one element per AVC/vibration. Several AVC elements
    can be fed with the same value (e.g. several vibrations affecting the
    same mode) by connecting them to the same upstream signal, and their
    corrections can be summed back onto the relevant mode with a downstream
    combiner object (e.g. :py:class:`~specula.processing_objects.linear_combination.LinearCombination`).

    Algorithm (per AVC element, at loop iteration k)
    --------------------------------------------------
    Given the current internal state **x** = (x1, x2, x3, x4), phase
    estimate alpha[k], (angular) frequency estimate omega[k] and current
    correction u[k] (all computed at the previous iteration):

    1. **Apply correction**: ``out_comm[k] = u[k]`` (Step 1 of the HRTC
       process: the correction computed at the previous iteration is
       output first, ready to be added to the loop command).
    2. **Update**: given the new measurement y[k], compute the regressor
       ``W(alpha)``, the prediction error, update **x**, the frequency
       omega, the phase alpha (with Kahan-compensated summation, matching
       ``AVC.alpha_c`` in the Matlab code) and the correction u[k+1] ready
       for the next iteration.

    This closely follows the actual Matlab implementation (``updateAVC.m``)
    rather than the idealized equations in the reference document: notably
    the tuning constant ``c`` only multiplies the update of x1, x2 (not
    x3, x4), and the phase update uses compensated (Kahan-Babuska)
    summation via the extra ``alpha_c`` state.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters; only ``time_step`` (the AO loop sampling
        period Ts) is used.
    n_avc : int [1]
        Number of independent AVC instances (vibrations) tracked by this
        object. ``in_measurement`` and ``out_comm`` have this length.
    freq : float or array-like [Hz], length n_avc
        Initial guess of the vibration frequency for each AVC.
    gx : float or array-like, length n_avc
        Update gain for the AVC state vector **x** (``AVC.gx``).
    gomega : float or array-like, length n_avc
        Update gain for the frequency estimate (``AVC.gomega``). Set to 0
        to disable frequency tracking for an AVC (fixed-frequency mode).
    k : float or array-like, length n_avc
        Phase/frequency coupling gain (``AVC.k``, also referred to as
        ``Kph`` in the calibration code).
    c : float or array-like, length n_avc
        Tuning constant multiplying the update of the first two state
        components x1, x2 (``AVC.c``).
    n_oversample : float or array-like, length n_avc, optional
        Oversampling factor ``AVC.N`` used in the frequency/phase update.
        Default: 1 (no oversampling).
    x0 : float or array-like, length n_avc, optional
        Initial value of state component x1 (real part of the plant
        response estimate at the vibration frequency). Default: 1.0.
        x0 and x1 cannot both be zero for a given AVC.
    x1 : float or array-like, length n_avc, optional
        Initial value of state component x2 (imaginary part of the plant
        response estimate). Default: 0.0.
    alpha0 : float or array-like, length n_avc, optional
        Initial phase estimate [rad]. Default: 0.0.
    theta_min_energy : float, optional
        Numerical safeguard: when x1^2 + x2^2 drops below this threshold,
        the amplitude estimate hatTheta is frozen to (1, 0) instead of
        dividing by a near-zero number (matches the ``1e-2`` threshold
        hard-coded in ``updateAVC.m``). Default: 1e-2.
    target_device_idx : int, optional
    precision : int, optional

    Inputs
    ------
    in_measurement : BaseValue, length n_avc
        Current measurement y[k] for each AVC (e.g. per-mode residual).

    Outputs
    -------
    out_comm : BaseValue, length n_avc
        Correction command u[k], computed at the previous iteration, ready
        to be added to the loop command.
    out_freq : BaseValue, length n_avc
        Current vibration frequency estimate [Hz] (omega / 2*pi), provided
        for monitoring/diagnostics.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 n_avc: int,
                 freq,
                 gx,
                 gomega,
                 k,
                 c,
                 n_oversample=1.0,
                 x0=1.0,
                 x1=0.0,
                 alpha0=0.0,
                 theta_min_energy: float = 1e-2,
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if n_avc < 1:
            raise ValueError('n_avc must be at least 1')
        self._n_avc = n_avc
        self._T = float(simul_params.time_step)
        self._theta_min_energy = theta_min_energy

        # Tuning parameters (constant for the object lifetime), broadcast
        # to n_avc elements if given as scalars.
        self._gx = self._to_param(gx, 'gx')
        self._gomega = self._to_param(gomega, 'gomega')
        self._k = self._to_param(k, 'k')
        self._c = self._to_param(c, 'c')
        self._n = self._to_param(n_oversample, 'n_oversample')

        freq_arr = self._to_param(freq, 'freq')

        # States: x = (x1, x2, x3, x4), omega, alpha, alpha_c (Kahan
        # compensation term for the phase accumulation)
        x0_arr = self._to_param(x0, 'x0')
        x1_arr = self._to_param(x1, 'x1')
        self._x = self.xp.zeros((n_avc, 4), dtype=self.dtype)
        self._x[:, 0] = x0_arr
        self._x[:, 1] = x1_arr
        # x3, x4 always start at zero, as in the Matlab initialization
        self._omega = self.xp.array(2 * self.xp.pi * freq_arr, dtype=self.dtype)
        self._alpha = self._to_param(alpha0, 'alpha0')
        self._alpha_c = self.xp.zeros(n_avc, dtype=self.dtype)

        # Correction u[k]; starts at zero since x3 = x4 = 0 initially
        self._correction = self.xp.zeros(n_avc, dtype=self.dtype)

        # Working buffers for measurement and outputs
        self._measurement = self.xp.zeros(n_avc, dtype=self.dtype)
        self._out_buffer = self.xp.zeros(n_avc, dtype=self.dtype)
        self._freq_buffer = self.xp.zeros(n_avc, dtype=self.dtype)

        # Outputs
        self._out_comm = BaseValue(
            'AVC correction command',
            value=self.xp.zeros(n_avc, dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision)
        self._out_freq = BaseValue(
            'AVC vibration frequency estimate [Hz]',
            value=self.xp.zeros(n_avc, dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision)

        self.inputs['in_measurement'] = InputValue(type=BaseValue)
        self.outputs['out_comm'] = self._out_comm
        self.outputs['out_freq'] = self._out_freq

    def _to_param(self, value, name):
        """Convert a scalar or array-like parameter to a static xp array of
        length n_avc, broadcasting scalars. Only used at init time."""
        arr = self.xp.atleast_1d(self.xp.asarray(value, dtype=self.dtype))
        if arr.size == 1:
            arr = self.xp.full(self._n_avc, arr.reshape(-1)[0], dtype=self.dtype)
        elif arr.size != self._n_avc:
            raise ValueError(
                f"Parameter '{name}' has size {arr.size} but n_avc={self._n_avc}")
        return arr

    @classmethod
    def input_names(cls):
        return {'in_measurement': InputDesc(BaseValue, 'Input measurement signal, one value per AVC instance')}

    @classmethod
    def output_names(cls):
        return {
            'out_comm': OutputDesc(BaseValue, 'Output correction command, one value per AVC instance'),
            'out_freq': OutputDesc(BaseValue, 'Estimated vibration frequency [Hz], one value per AVC instance'),
        }

    def setup(self):
        super().setup()
        self.build_stream()

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        measurement = self.local_inputs['in_measurement'].value
        measurement_array = self.xp.asarray(measurement, dtype=self.dtype)
        measurement_array = self.xp.atleast_1d(measurement_array).ravel()

        if measurement_array.size != self._n_avc:
            if measurement_array.size == 1:
                measurement_array = self.xp.full(
                    self._n_avc, measurement_array[0], dtype=self.dtype)
            else:
                raise ValueError(
                    f"Input in_measurement has size {measurement_array.size} "
                    f"but AVC expects {self._n_avc} values")

        self._measurement[:] = measurement_array

    def trigger_code(self):
        xp = self.xp

        # output the correction computed at the previous iteration, and the frequency estimate
        self._out_buffer[:] = self._correction
        self._freq_buffer[:] = self._omega / (2 * xp.pi)

        # AVC update: compute the regressor W(alpha), prediction error, update x, alpha, omega and the correction for the next iteration
        x1 = self._x[:, 0]
        x2 = self._x[:, 1]
        x3 = self._x[:, 2]
        x4 = self._x[:, 3]
        alpha = self._alpha
        omega = self._omega
        alpha_c = self._alpha_c

        denom = x1 * x1 + x2 * x2
        small = denom < self._theta_min_energy
        safe_denom = xp.where(small, 1.0, denom)
        thetac = xp.where(small, 1.0, -(x1 * x3 - x2 * x4) / safe_denom)
        thetas = xp.where(small, 0.0, -(x1 * x4 + x2 * x3) / safe_denom)

        cos_a = xp.cos(alpha)
        sin_a = xp.sin(alpha)

        w1 = thetac * cos_a + thetas * sin_a
        w2 = thetas * cos_a - thetac * sin_a
        w3 = cos_a
        w4 = sin_a

        # prediction error: t = W.x - measurement
        err = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 - self._measurement

        new_x1 = x1 - self._gx * self._T * w1 * err * self._c
        new_x2 = x2 - self._gx * self._T * w2 * err * self._c
        new_x3 = x3 - self._gx * self._T * w3 * err
        new_x4 = x4 - self._gx * self._T * w4 * err
        self._x[:, 0] = new_x1
        self._x[:, 1] = new_x2
        self._x[:, 2] = new_x3
        self._x[:, 3] = new_x4

        # Phase update with Kahan compensated summation
        alpha_input = (self._T * omega / self._n +
                       2.0 * self._k * self._T / self._n * self._gomega * sin_a * err)
        alpha_y = alpha_input - alpha_c
        alpha_t = alpha + alpha_y
        self._alpha_c[:] = (alpha_t - alpha) - alpha_y
        alpha_new = alpha_t
        alpha_new = xp.where(alpha_new > xp.pi, alpha_new - 2 * xp.pi, alpha_new)
        alpha_new = xp.where(alpha_new < -xp.pi, alpha_new + 2 * xp.pi, alpha_new)
        self._alpha[:] = alpha_new

        # Frequency update (uses alpha before this iteration's update)
        self._omega[:] = omega + 2.0 * self._T * self._gomega * sin_a * err

        # Recompute hatTheta with the updated state, for the next correction
        denom2 = new_x1 * new_x1 + new_x2 * new_x2
        small2 = denom2 < self._theta_min_energy
        safe_denom2 = xp.where(small2, 1.0, denom2)
        thetac2 = xp.where(small2, 1.0, -(new_x1 * new_x3 - new_x2 * new_x4) / safe_denom2)
        thetas2 = xp.where(small2, 0.0, -(new_x1 * new_x4 + new_x2 * new_x3) / safe_denom2)

        self._correction[:] = xp.cos(alpha_new) * thetac2 + xp.sin(alpha_new) * thetas2

    def post_trigger(self):
        super().post_trigger()
        self._out_comm.value[:] = self._out_buffer
        self._out_comm.generation_time = self.current_time
        self._out_freq.value[:] = self._freq_buffer
        self._out_freq.generation_time = self.current_time
