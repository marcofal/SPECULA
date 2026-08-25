from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams


class AVCSimplified(BaseProcessingObj):
    """
    Minimal Adaptive Vibration Cancellation: a self-tuning resonant controller.

    This is the reduced form of :py:class:`~specula.processing_objects.avc.AVC`
    obtained by freezing the plant estimate and setting the regularizer to
    zero. It is not a different algorithm: with ``adapt_plant=False`` and
    ``theta_min_energy=0`` the full object reduces to this one exactly (see
    ``test/test_avc_simplified.py``, which checks the two agree to float
    precision over 20000 steps). What changes is that everything the
    reduction makes vacuous has been removed, so what remains admits a
    complete stability statement.

    **What was removed, and why**

    - *The plant adaptation* ``x1, x2``. From a single tone the averaged
      regressor covariance ``<W W^T>`` has rank 2, so the plant components
      are not identifiable; they contribute two zero eigenvalues to the
      closed loop, and under measurement noise they perform an unbounded
      random walk which eventually carries ``arg(P_hat)`` outside the
      stability cone below. The plant is now a *calibration*, measured once
      in closed loop (``avc_full_demo/params_measure_plant.yml``).

    - *The regularizer* ``epsilon`` / ``theta_min_energy`` *and the division
      guard*. With the plant fixed, ``|P_hat|^2`` is a constant, not a
      state; it cannot approach zero. In fact no division is done at run
      time at all: the map from ``(x3, x4)`` to the control quadratures is
      the fixed 2x2 matrix built once in the constructor.

    - *The cost function and the pseudo-gradient framing*. With
      ``epsilon = 0`` the prediction term vanishes identically,
      ``W^T x == 0``, hence ``e == -y``: the equation-error being
      "minimized" carries no dependence on the state at all. The update is
      therefore written for what it is -- integral action on the measured
      residual -- rather than as a gradient step in a cost that is
      degenerate. Correspondingly the ``c``, ``soft_clamp`` and
      ``exact_gradient`` knobs have no counterpart here.

    **The algorithm.** Per sample, with ``c = cos(alpha)``, ``s = sin(alpha)``
    and ``mu = gx * Ts``::

        u  = thetac * c + thetas * s          # command, added to the loop
        y  = in_measurement                   # residual of the mode
        x3 += mu * c * y                      # integrate, in the rotating
        x4 += mu * s * y                      #   frame (a lock-in)
        omega -= 2 * gomega * Ts * s * y      # frequency tracking (PLL)
        alpha += Ts * omega / N - 2 * k * gomega * Ts / N * s * y

    followed by the fixed map ``(x3, x4) -> (thetac, thetas)``. Multiplying
    the residual by ``(cos alpha, sin alpha)`` demodulates it into the frame
    rotating at the vibration frequency, and accumulating integrates it
    there; in complex form the two state lines are simply
    ``Lambda_hat += mu * exp(i*alpha) * y``.

    **What can be guaranteed.** Because the plant is constant, the averaged
    dynamics is a *linear time-invariant* scalar complex system,
    ``d(Lambda_hat)/dt = gamma * (Lambda_star - kappa * Lambda_hat)`` with
    ``gamma = gx/2`` and ``kappa = conj(P_star / P_hat)``. Hence, globally:

    - a unique equilibrium, at which the tone is cancelled exactly;
    - exponential stability **if and only if**
      ``|arg(P_star) - arg(P_hat)| < 90 deg``;
    - a closed-form rate, ``0.5 * gx * Re(kappa)``;
    - a step-size limit ``gx * Ts < 2 * Re(kappa) / |kappa|^2``, typically
      three orders of magnitude above the gain actually used.

    Note what the stability condition does *not* involve: the plant
    **magnitude**. A calibration wrong by a factor of ten only slows
    convergence; only the phase matters, and it has a +-90 degree tolerance.
    Note also that what converges is the residual, not an identification:
    ``Lambda_hat`` settles on whatever value cancels the tone *given* the
    calibration in use, and equals the true disturbance only if the
    calibration is exact.

    See ``avc_full_demo/avc_specula_report.pdf`` for the derivations, and
    ``avc_full_demo/avc_margin_check.py`` for a numerical test of the 90
    degree condition.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters; only ``time_step`` (the AO loop period Ts)
        is used.
    n_avc : int
        Number of independent instances (vibrations). ``in_measurement``
        and ``out_comm`` have this length. Several instances may be fed the
        same measurement, and their corrections summed downstream (e.g.
        with :py:class:`~specula.processing_objects.linear_combination.LinearCombination`).
    freq : float or array-like [Hz], length n_avc
        Initial guess of the vibration frequency. The PLL will pull it to
        the true value provided the tone dominates the residual near it.
    gx : float or array-like, length n_avc
        Gain of the disturbance integrator. The convergence rate is
        ``0.5 * gx * Re(kappa)`` [1/s]; see the step-size limit above.
    gomega : float or array-like, length n_avc
        Gain of the frequency loop. Set to 0 to hold the frequency fixed at
        ``freq``. Must be re-tuned when the tone's signal-to-noise ratio
        changes, since the PLL is driven by whatever dominates the error.
    k : float or array-like, length n_avc
        Phase/frequency coupling in the PLL (proportional path).
    plant_re, plant_im : float or array-like, length n_avc, optional
        Calibrated closed-loop plant response at the vibration frequency,
        in cartesian form: ``P_hat = plant_re + 1j * plant_im``. Defaults
        to ``1 + 0j``. These are the counterpart of ``x0, x1`` in
        :py:class:`~specula.processing_objects.avc.AVC`, but here they are
        a specification rather than a seed: the accuracy requirement is the
        90 degree phase condition above. Must not be zero.
    plant_gain, plant_phase : float or array-like, length n_avc, optional
        The same calibration in polar form, ``|P_hat|`` and
        ``arg(P_hat)`` in **degrees**. Mutually exclusive with
        ``plant_re``/``plant_im``; provided because the calibration is
        normally measured as a magnitude and a phase.
    n_oversample : float or array-like, length n_avc, optional
        Oversampling factor ``N`` in the phase/frequency update, unchanged
        from the original algorithm. Default 1.
    alpha0 : float or array-like, length n_avc, optional
        Initial phase estimate [rad]. Default 0.
    target_device_idx : int, optional
    precision : int, optional

    Inputs
    ------
    in_measurement : BaseValue, length n_avc
        Current measurement y[k] (e.g. the residual of one modal
        coefficient).

    Outputs
    -------
    out_comm : BaseValue, length n_avc
        Correction u[k], computed at the previous iteration, ready to be
        added to the loop command.
    out_freq : BaseValue, length n_avc
        Current frequency estimate [Hz].
    out_state : BaseValue, shape (n_avc, 4)
        ``[plant_re, plant_im, x3, x4]``. The first two are the frozen
        calibration and never change; they are reported so that this output
        stays layout-compatible with
        :py:class:`~specula.processing_objects.avc.AVC` and with the
        existing analysis scripts.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 n_avc: int,
                 freq,
                 gx,
                 gomega,
                 k,
                 plant_re=None,
                 plant_im=None,
                 plant_gain=None,
                 plant_phase=None,
                 n_oversample=1.0,
                 alpha0=0.0,
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if n_avc < 1:
            raise ValueError('n_avc must be at least 1')
        self._n_avc = n_avc
        self._T = float(simul_params.time_step)

        # Tuning parameters, broadcast to n_avc elements if scalar
        self._gx = self._to_param(gx, 'gx')
        self._gomega = self._to_param(gomega, 'gomega')
        self._k = self._to_param(k, 'k')
        self._n = self._to_param(n_oversample, 'n_oversample')

        # --- the plant calibration -------------------------------------
        # Accepted either cartesian (plant_re, plant_im) or polar
        # (plant_gain, plant_phase [deg]), never both.
        cartesian = (plant_re is not None) or (plant_im is not None)
        polar = (plant_gain is not None) or (plant_phase is not None)
        if cartesian and polar:
            raise ValueError(
                'Give the plant calibration either as plant_re/plant_im or as '
                'plant_gain/plant_phase, not both')
        if polar:
            rho = self._to_param(1.0 if plant_gain is None else plant_gain, 'plant_gain')
            psi = self._to_param(0.0 if plant_phase is None else plant_phase, 'plant_phase')
            psi = psi * (self.xp.pi / 180.0)
            p_re = rho * self.xp.cos(psi)
            p_im = rho * self.xp.sin(psi)
        else:
            p_re = self._to_param(1.0 if plant_re is None else plant_re, 'plant_re')
            p_im = self._to_param(0.0 if plant_im is None else plant_im, 'plant_im')

        energy = p_re * p_re + p_im * p_im
        if bool(self.xp.any(energy <= 0)):
            raise ValueError(
                'The plant calibration must be non-zero: plant_re and plant_im '
                'cannot both be zero for any AVC instance')
        self._plant_re = p_re
        self._plant_im = p_im

        # Fixed map (x3, x4) -> (thetac, thetas). With
        # theta = -Lambda_hat / conj(P_hat) = -(1/rho) exp(i psi) Lambda_hat,
        # this is a rotation by the plant phase and a scaling by the inverse
        # plant gain, i.e.
        #     thetac = -( x3 cos psi - x4 sin psi) / rho
        #     thetas = -( x3 sin psi + x4 cos psi) / rho
        # which, written in cartesian form, is -(P_re, P_im)/|P|^2 contracted
        # with (x3, x4). Evaluated once here; no division at run time.
        self._m11 = -p_re / energy
        self._m12 = p_im / energy
        self._m21 = -p_im / energy
        self._m22 = -p_re / energy

        # --- states ----------------------------------------------------
        # Only the disturbance quadratures and the frequency loop. alpha_c
        # is the Kahan compensation carried by the phase accumulator.
        freq_arr = self._to_param(freq, 'freq')
        self._x3 = self.xp.zeros(n_avc, dtype=self.dtype)
        self._x4 = self.xp.zeros(n_avc, dtype=self.dtype)
        self._omega = self.xp.array(2 * self.xp.pi * freq_arr, dtype=self.dtype)
        self._alpha = self._to_param(alpha0, 'alpha0')
        self._alpha_c = self.xp.zeros(n_avc, dtype=self.dtype)

        # Correction u[k]; zero at start since x3 = x4 = 0
        self._correction = self.xp.zeros(n_avc, dtype=self.dtype)

        # Working buffers
        self._measurement = self.xp.zeros(n_avc, dtype=self.dtype)
        self._out_buffer = self.xp.zeros(n_avc, dtype=self.dtype)
        self._freq_buffer = self.xp.zeros(n_avc, dtype=self.dtype)
        self._state_buffer = self.xp.zeros((n_avc, 4), dtype=self.dtype)
        self._state_buffer[:, 0] = self._plant_re
        self._state_buffer[:, 1] = self._plant_im

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
        self._out_state = BaseValue(
            'AVC state [plant_re, plant_im, x3, x4] (diagnostic)',
            value=self.xp.zeros((n_avc, 4), dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision)

        self.inputs['in_measurement'] = InputValue(type=BaseValue)
        self.outputs['out_comm'] = self._out_comm
        self.outputs['out_freq'] = self._out_freq
        self.outputs['out_state'] = self._out_state

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
            'out_state': OutputDesc(BaseValue, 'State [plant_re, plant_im, x3, x4] per AVC instance (diagnostic)'),
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
                    f"but AVCSimplified expects {self._n_avc} values")

        self._measurement[:] = measurement_array

    def trigger_code(self):
        xp = self.xp

        # Step 1: emit the correction computed at the previous iteration,
        # and the current frequency estimate.
        self._out_buffer[:] = self._correction
        self._freq_buffer[:] = self._omega / (2 * xp.pi)

        alpha = self._alpha
        omega = self._omega
        alpha_c = self._alpha_c
        cos_a = xp.cos(alpha)
        sin_a = xp.sin(alpha)

        # Step 2: the measured residual. There is no prediction to subtract:
        # with epsilon = 0 the regressor term W^T x vanishes identically, so
        # the equation error of the original formulation is exactly -y.
        y = self._measurement

        # Step 3: integrate the residual in the frame rotating at the
        # estimated vibration frequency. Multiplying by (cos, sin) is the
        # demodulation, accumulating is the integration; together they are
        # Lambda_hat += mu * exp(i alpha) * y.
        mu = self._gx * self._T
        new_x3 = self._x3 + mu * cos_a * y
        new_x4 = self._x4 + mu * sin_a * y
        self._x3[:] = new_x3
        self._x4[:] = new_x4
        self._state_buffer[:, 2] = new_x3
        self._state_buffer[:, 3] = new_x4

        # Step 4: frequency loop. Same PLL as the original algorithm. Note
        # the sign: the reference implementation drives the frequency loop
        # with -e (unlike the state update, which uses +e), so with e = -y
        # both PLL terms carry -y. The phase accumulator uses Kahan
        # compensated summation, since it runs for the whole simulation and
        # a naive sum drifts.
        alpha_input = (self._T * omega / self._n -
                       2.0 * self._k * self._T / self._n * self._gomega * sin_a * y)
        alpha_y = alpha_input - alpha_c
        alpha_t = alpha + alpha_y
        self._alpha_c[:] = (alpha_t - alpha) - alpha_y
        alpha_new = alpha_t
        alpha_new = xp.where(alpha_new > xp.pi, alpha_new - 2 * xp.pi, alpha_new)
        alpha_new = xp.where(alpha_new < -xp.pi, alpha_new + 2 * xp.pi, alpha_new)
        self._alpha[:] = alpha_new

        # Frequency update (uses alpha from before this iteration's update)
        self._omega[:] = omega - 2.0 * self._T * self._gomega * sin_a * y

        # Step 5: refresh the control quadratures through the fixed map, and
        # form the correction for the next iteration.
        thetac = self._m11 * new_x3 + self._m12 * new_x4
        thetas = self._m21 * new_x3 + self._m22 * new_x4
        self._correction[:] = xp.cos(alpha_new) * thetac + xp.sin(alpha_new) * thetas

    def post_trigger(self):
        super().post_trigger()
        self._out_comm.value[:] = self._out_buffer
        self._out_comm.generation_time = self.current_time
        self._out_freq.value[:] = self._freq_buffer
        self._out_freq.generation_time = self.current_time
        self._out_state.value[:] = self._state_buffer
        self._out_state.generation_time = self.current_time
