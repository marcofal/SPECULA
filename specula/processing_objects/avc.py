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
        Numerical safeguard threshold on x1^2 + x2^2. Its effect depends on
        ``soft_clamp`` (see below). Default: 1e-2.
    soft_clamp : bool, optional
        Selects how ``theta_min_energy`` regularizes the hatTheta ratio
        (``hatTheta = -(x3+i*x4)/conj(x1+i*x2)``, eq. 11 in Muradore et al.):

        - ``False`` (default, matches ``updateAVC.m`` exactly, and is what
          ``test_matches_matlab_reference_implementation`` validates against):
          whenever ``x1^2 + x2^2 < theta_min_energy``, hatTheta is hard-reset
          to ``(1, 0)`` -- a discontinuous switch right at the threshold
          boundary. Because the ratio's gain (and, through the regressor,
          the feedback into x1/x2's own next update) formally diverges as
          x1^2+x2^2 -> 0, this discontinuity sitting on a boundary the
          state keeps crossing is prone to limit-cycle chattering right at
          the clamp (observed empirically in avc_full_demo -- see its
          README.md).
        - ``True``: replaces the hard branch with an unconditional soft
          floor, ``safe_denom = x1^2 + x2^2 + theta_min_energy``, applied
          to every sample regardless of magnitude. This bounds the ratio's
          worst-case gain to ``~1/theta_min_energy`` without ever jumping
          discontinuously, at the cost of a small, smooth bias in hatTheta
          even away from the origin. Not part of the original algorithm or
          of ``updateAVC.m``; an experimental alternative regularization.
    adapt_plant : bool, optional
        Whether to update the plant-response estimate ``x1, x2`` online
        (``True``, default, matches the original algorithm and
        ``updateAVC.m``), or hold them fixed at their ``x0, x1``
        calibration and adapt only the disturbance components ``x3, x4``
        (and ``omega, alpha``) (``False``).

        Rationale for ``False``: from a *single* sinusoidal excitation the
        regressor ``W`` (eq. 9) spans only a 2-D subspace, so the averaged
        regressor covariance ``<W W^T>`` has exactly two nonzero and two
        zero eigenvalues -- the 4-component state is only rank-2
        observable, and the two unobservable directions each carry unit
        weight on a *plant* axis (``x1`` or ``x2``). Adapting ``x1, x2``
        online therefore cannot identify them (the estimate drifts along
        the null space), and because they sit in the ``1/(x1^2+x2^2)``
        denominator of ``hatTheta`` that drift feeds back and diverges.
        Since a good ``x0, x1`` is normally obtained offline anyway
        (``avc_full_demo/calibrate_avc.py``,
        ``specula/lib/avc_plant_model.py``), freezing the unobservable
        plant axes and adapting only the observable disturbance
        quadrature ``x3, x4`` removes the drift and recovers a working
        canceller. See ``avc_full_demo/README.md`` /
        ``avc_full_demo/avc_observability_check.py`` for the full
        analysis. Default ``True`` preserves exact original behaviour
        (and the ``test_matches_matlab_reference_implementation`` check).
    exact_gradient : {False, True, 'plant', 'full', 'none'}, optional
        Selects which state components get the dropped chain-rule term
        added back to their update direction. The regressor ``W`` depends
        on ``theta``, and ``theta`` depends on ``x`` through eq. 11, so the
        true error derivative is
        ``de/dx_i = W_i + sum_j (dW_j/dx_i) x_j``; the paper (eq. 10) and
        ``updateAVC.m`` keep only ``W_i`` and drop the second term. The
        exact term is added back analytically as
        ``g = W + J^T x``, ``J[j,i] = dW_j/dx_i``.

        - ``False`` / ``'none'`` (default): pseudo-gradient on all four
          components -- exact original behaviour (and what the
          MATLAB-reference test checks).
        - ``True`` / ``'full'``: exact gradient on all four. This removes
          the plant drift, but it also rewrites the *disturbance* update
          ``x3, x4`` -- and there the pseudo-gradient's ``x3 -= gx*T*cos*e``
          is doing the useful work (integral action on the residual, the
          thing that ramps the correction up until the tone cancels). The
          exact version turns that into equation-error minimisation whose
          trivial solution is ``x3 = x4 = 0`` (no correction), so the plant
          is pinned but cancellation is lost. Diagnostic only.
        - ``'plant'`` (**recommended for online plant adaptation**): exact
          gradient on ``x1, x2`` only, pseudo-gradient on ``x3, x4``. Keeps
          the integral cancellation action while giving the plant estimate
          the true (non-runaway) descent direction. With ``adapt_plant=True``
          and a *damped* plant step (small ``c`` -- the plant update carries
          a ``(|theta|^2 + 1)`` gain, so it needs ``gx*T*(|theta|^2+1)*c <~ 2``
          for stability; e.g. ``c=0.1`` at ``gx=3``), the plant makes a
          small bounded, settling excursion around its calibrated seed
          rather than wandering (pseudo) or collapsing (full), and
          cancellation is as good as or better than a frozen plant
          (demo: 47 Hz residual ~2 nm vs. ~13 nm frozen). Note this still
          does not *identify* the plant (unobservable from one tone) -- it
          self-optimises the cancellation point near the seed.

        Neither ``True`` nor ``'plant'`` makes the plant observable from a
        single tone (``g`` is still confined to the same 2-D per-cycle
        subspace). See ``avc_full_demo/avc_chainrule_check.py`` and
        ``README.md``.
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
    out_state : BaseValue, shape (n_avc, 4)
        Internal state ``[x1, x2, x3, x4]`` per AVC instance, *after* this
        iteration's update (diagnostic only, not needed for normal use).
        ``x1``/``x2`` are the online plant-response estimate
        (``P_r``/``P_i``, seeded from the ``x0``/``x1`` constructor
        parameters); ``x3``/``x4`` are the raw disturbance quadrature
        estimate (driven directly by ``cos(alpha)``/``sin(alpha)``, with
        no plant model involved). ``hatTheta = -(x3+i*x4)/conj(x1+i*x2)``
        is the amplitude/phase estimate actually used for the correction;
        useful for checking whether the plant estimate ``x1+i*x2`` is
        converging/stable or wandering (its magnitude sits in the
        denominator of that ratio, so small/noisy values there can get
        amplified into a growing correction -- see
        ``avc_full_demo/README.md`` for a worked example).
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
                 soft_clamp: bool = False,
                 adapt_plant: bool = True,
                 exact_gradient=False,
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if n_avc < 1:
            raise ValueError('n_avc must be at least 1')
        self._n_avc = n_avc
        self._T = float(simul_params.time_step)
        self._theta_min_energy = theta_min_energy
        self._soft_clamp = bool(soft_clamp)
        self._adapt_plant = bool(adapt_plant)

        # exact_gradient selects which state components get the chain-rule
        # correction term added to their update direction:
        #   False / 'none'  -> pseudo-gradient on all (paper default)
        #   True  / 'full'  -> exact gradient on all 4 components
        #   'plant'         -> exact on x1,x2 (plant) only, pseudo on x3,x4
        #                      (the hybrid: kill the plant drift but keep the
        #                       integral cancellation action on x3,x4)
        if exact_gradient is False or exact_gradient is None or \
                (isinstance(exact_gradient, str) and exact_gradient.lower() in ('none', 'off', 'false')):
            self._exact_plant, self._exact_dist = False, False
        elif exact_gradient is True or \
                (isinstance(exact_gradient, str) and exact_gradient.lower() in ('full', 'all', 'true')):
            self._exact_plant, self._exact_dist = True, True
        elif isinstance(exact_gradient, str) and exact_gradient.lower() == 'plant':
            self._exact_plant, self._exact_dist = True, False
        else:
            raise ValueError(
                "exact_gradient must be False, True, 'plant', 'full' or 'none', "
                f"got {exact_gradient!r}")
        self._exact_gradient = self._exact_plant or self._exact_dist

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
        self._state_buffer = self.xp.zeros((n_avc, 4), dtype=self.dtype)

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
            'AVC internal state [x1, x2, x3, x4] (diagnostic)',
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
            'out_state': OutputDesc(BaseValue, 'Internal state [x1, x2, x3, x4] per AVC instance (diagnostic)'),
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
        if self._soft_clamp:
            safe_denom = denom + self._theta_min_energy
            thetac = -(x1 * x3 - x2 * x4) / safe_denom
            thetas = -(x1 * x4 + x2 * x3) / safe_denom
        else:
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
        err = self._measurement - (w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4)

        # Update direction. The paper (eq. 10) descends along W, treating W
        # as independent of x. With exact_gradient we add back the dropped
        # chain-rule term so the direction is the true gradient of the
        # squared error, g = W + J^T x with J[j,i] = dW_j/dx_i (through
        # theta). Computed analytically; verified against finite
        # differences in avc_full_demo/avc_chainrule_check.py.
        if self._exact_gradient:
            A = x1 * cos_a - x2 * sin_a
            B = x1 * sin_a + x2 * cos_a
            inv_sd = 1.0 / safe_denom
            # d(theta_c)/dx_i and d(theta_s)/dx_i, using theta = N / safe_denom
            dtc1 = (-x3 - 2.0 * x1 * thetac) * inv_sd
            dtc2 = ( x4 - 2.0 * x2 * thetac) * inv_sd
            dtc3 = -x1 * inv_sd
            dtc4 =  x2 * inv_sd
            dts1 = (-x4 - 2.0 * x1 * thetas) * inv_sd
            dts2 = (-x3 - 2.0 * x2 * thetas) * inv_sd
            dts3 = -x2 * inv_sd
            dts4 = -x1 * inv_sd
            cr1 = A * dtc1 + B * dts1
            cr2 = A * dtc2 + B * dts2
            cr3 = A * dtc3 + B * dts3
            cr4 = A * dtc4 + B * dts4
            if not self._soft_clamp:
                # where the hard clamp froze theta to (1, 0), its gradient
                # is zero, so the chain-rule term vanishes there
                cr1 = xp.where(small, 0.0, cr1)
                cr2 = xp.where(small, 0.0, cr2)
                cr3 = xp.where(small, 0.0, cr3)
                cr4 = xp.where(small, 0.0, cr4)
            # apply the correction selectively: plant axes (x1,x2) if
            # _exact_plant, disturbance axes (x3,x4) if _exact_dist
            g1 = w1 + cr1 if self._exact_plant else w1
            g2 = w2 + cr2 if self._exact_plant else w2
            g3 = w3 + cr3 if self._exact_dist else w3
            g4 = w4 + cr4 if self._exact_dist else w4
        else:
            g1, g2, g3, g4 = w1, w2, w3, w4

        # plant-estimate update gain: normally gx, but 0 when adapt_plant
        # is off, which freezes x1, x2 at their calibrated x0, x1 (the
        # unobservable plant axes -- see class docstring)
        gx_plant = self._gx if self._adapt_plant else self._gx * 0.0
        new_x1 = x1 - gx_plant * self._T * g1 * (-err) * self._c
        new_x2 = x2 - gx_plant * self._T * g2 * (-err) * self._c
        new_x3 = x3 - self._gx * self._T * g3 * (-err)
        new_x4 = x4 - self._gx * self._T * g4 * (-err)
        self._x[:, 0] = new_x1
        self._x[:, 1] = new_x2
        self._x[:, 2] = new_x3
        self._x[:, 3] = new_x4
        self._state_buffer[:, 0] = new_x1
        self._state_buffer[:, 1] = new_x2
        self._state_buffer[:, 2] = new_x3
        self._state_buffer[:, 3] = new_x4

        # Phase update with Kahan compensated summation
        alpha_input = (self._T * omega / self._n -
                       2.0 * self._k * self._T / self._n * self._gomega * sin_a * err)
        alpha_y = alpha_input - alpha_c
        alpha_t = alpha + alpha_y
        self._alpha_c[:] = (alpha_t - alpha) - alpha_y
        alpha_new = alpha_t
        alpha_new = xp.where(alpha_new > xp.pi, alpha_new - 2 * xp.pi, alpha_new)
        alpha_new = xp.where(alpha_new < -xp.pi, alpha_new + 2 * xp.pi, alpha_new)
        self._alpha[:] = alpha_new

        # Frequency update (uses alpha before this iteration's update)
        self._omega[:] = omega - 2.0 * self._T * self._gomega * sin_a * err

        # Recompute hatTheta with the updated state, for the next correction
        denom2 = new_x1 * new_x1 + new_x2 * new_x2
        if self._soft_clamp:
            safe_denom2 = denom2 + self._theta_min_energy
            thetac2 = -(new_x1 * new_x3 - new_x2 * new_x4) / safe_denom2
            thetas2 = -(new_x1 * new_x4 + new_x2 * new_x3) / safe_denom2
        else:
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
        self._out_state.value[:] = self._state_buffer
        self._out_state.generation_time = self.current_time
