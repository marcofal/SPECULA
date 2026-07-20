"""
Analytic closed-loop plant model for AVC initial-state calibration.

Ported from ESO's HRTC AVC calibration chain (``generateAVCInitConditions.m``
/ ``generateAVCInitConditionsPlot.m``, used by ``computeAVCparamsModeCIAO.m``
/ ``computeAVCparamsCIAO.m``). The Matlab source computes an analytic model
of a single-mode AO loop's closed-loop transfer function and evaluates it at
a candidate vibration frequency to get a correctly phased and scaled initial
condition for :py:class:`~specula.processing_objects.avc.AVC`'s internal
state (``x0``, ``x1``), instead of the generic default (``x0=1, x1=0``,
i.e. unity gain / zero phase).

Why this matters
-----------------
:py:class:`~specula.processing_objects.avc.AVC`'s amplitude-tracking state
(``x1``/``x2`` in the running object, seeded from the ``x0``/``x1``
constructor parameters) is updated by a gradient-descent law that assumes
the regressor ``W(alpha)`` it builds from its own phase estimate is
proportional to the *actual* transfer function from "AVC's own correction"
to "what AVC measures next". In an open-loop tracking test (AVC sees the
raw disturbance directly, no delay, no other controller) that assumption
holds trivially. Once AVC is closed around a real AO loop -- with its own
integrator also acting on the same mode, WFS/DAC sample-and-hold, and a
multi-frame RTC delay -- "1 unit of correction in, 1 unit of raw signal
out" is no longer true: the loop's own closed-loop sensitivity function
rotates and scales that relationship at the vibration frequency. Seeding
AVC with a generic ``x0=1, x1=0`` starts the adaptive filter far from the
loop's actual fixed point, and the gradient update can fail to converge to
it -- observed in practice as the correction amplitude drifting/growing
over many seconds instead of settling (see ``avc_full_demo/README.md`` for
a worked numerical example of this failure mode).

This module computes that closed-loop transfer function analytically from
the loop's own known parameters (sampling period, integrator gain, RTC
delay) -- the SPECULA equivalent of measuring it from real telemetry the
way ``computeCIAOSS.m`` does for CIAO -- and uses it to seed AVC correctly.

Loop model
----------
Single-mode, linear, discrete-time-sampled loop, evaluated at continuous
frequency ``f`` via ``z = exp(2j*pi*f*T)`` (matches the Matlab source):

* ``WFS(f)``: WFS/CCD box-car frame integration (zero-order hold over one
  sample period ``T``).
* ``RTC(f)``: pure computational delay, ``loop_delay`` seconds.
* ``CTR(f)``: discrete integrator, gain ``loop_gain`` (SPECULA
  ``Integrator.int_gain`` for the mode of interest).
* ``DAC(f)``: DM/actuator zero-order hold over one sample period ``T``
  (same functional form as ``WFS``).
* ``DM(f)``: DM response. SPECULA's ``DM`` processing object has no
  internal dynamics (it is an ideal, instantaneous modal mirror), so by
  default this is just a constant gain ``dm_dc_gain`` (1.0). Pass
  ``dm_f0``/``dm_csi`` to instead model a damped 2nd-order mechanical
  resonance, for a DM that does have real dynamics.

The open-loop transfer function is ``HOL = WFS*RTC*CTR*DAC*DM`` and the
closed-loop sensitivity (disturbance rejection) is ``S = 1/(1+HOL)``.

AVC's own correction enters the loop at the same point as the main
integrator's command (in the SPECULA architecture used in
``avc_full_demo``, they are summed before the DM), so the transfer
function from "AVC command" to "measured residual" is

    P_avc(f) = WFS(f) * RTC(f) * delAVC(f) * DAC(f) * DM(f) * S(f)

where ``delAVC`` is any *additional* delay specific to AVC's own command
path beyond the main loop's RTC delay (0 if AVC's correction is combined
and applied within the same simulation step as the main loop's, as in
``avc_full_demo``).
"""
import numpy as np


def _as_freq_array(f):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    if np.any(f <= 0):
        raise ValueError("frequencies must be > 0 Hz (f=0 is a removable "
                          "singularity of the WFS/DAC zero-order-hold terms "
                          "and is not evaluated by this model)")
    return f


def _zoh(f, T):
    """Zero-order-hold transfer function over one sample period T,
    evaluated at frequency f [Hz]. Used for both the WFS/CCD frame
    integration and the DAC/DM sample-and-hold."""
    w = 2 * np.pi * f
    return (1 - np.exp(-1j * w * T)) / (1j * w * T)


def _dm_response(f, dm_dc_gain, dm_f0, dm_csi):
    if dm_f0 is None:
        return np.full(f.shape, dm_dc_gain, dtype=complex)
    w = 2 * np.pi * f
    w0 = 2 * np.pi * dm_f0
    return dm_dc_gain / ((1j * w)**2 / w0**2 + 2 * (1j * w) * dm_csi / w0 + 1)


def _lpf_response(f, fc, T):
    """Discrete transfer function of SPECULA's ``LowPassFilter`` with
    ``n_ord=1`` (first-order Butterworth via bilinear transform), evaluated
    at frequency ``f`` [Hz], for sample period ``T`` [s].

    Matches ``IirFilterData.lpf_from_fc(fc, fs=1/T, n_ord=1)`` exactly: that
    factory produces the difference equation
    ``y[n] = a0*x[n] + (1-a0)*y[n-1]`` with
    ``a0 = omega/(1+omega)``, ``omega = tan(pi*fc*T)`` -- a standard
    exponential-moving-average digital low-pass. Its z-transform is
    ``H(z) = a0 / (1 - (1-a0)*z^-1)``, evaluated here at ``z = exp(j*2*pi*f*T)``.
    """
    w = 2 * np.pi * f
    omega = np.tan(np.pi * fc * T)
    a0 = omega / (1 + omega)
    z_inv = np.exp(-1j * w * T)
    return a0 / (1 - (1 - a0) * z_inv)


def open_loop_transfer_function(f, T, loop_gain, loop_delay,
                                 dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                                 lpf_fc=None):
    """Open-loop transfer function ``HOL(f)`` of a single-mode AO loop.

    Parameters
    ----------
    f : float or array-like [Hz]
        Frequencies to evaluate at (must be > 0).
    T : float [s]
        Loop sampling period (``simul_params.time_step``).
    loop_gain : float
        Integrator gain for the mode of interest (SPECULA
        ``Integrator.int_gain``).
    loop_delay : float [s]
        Total pure RTC computational delay (SPECULA ``Integrator.delay``
        expressed in seconds, i.e. ``delay * T`` for an integer-frame
        SPECULA ``Integrator``).
    dm_dc_gain : float, optional
        DM DC gain. Default 1.0 (SPECULA's ideal, uncalibrated modal DM).
    dm_f0, dm_csi : float, optional
        Natural frequency [Hz] and damping ratio of an optional 2nd-order
        DM resonance. Both None (default) models an ideal, dynamics-free
        DM, matching SPECULA's ``DM`` processing object.
    lpf_fc : float [Hz], optional
        Cutoff frequency of an optional first-order actuator/DM low-pass
        filter placed in the command path (matching a SPECULA
        ``LowPassFilter(n_ord=1, cutoff_freq=lpf_fc)`` inserted between the
        command source and the DM). None (default) disables it. Distinct
        from ``dm_f0``/``dm_csi`` (a resonance model); this is a plain
        real, physically-inserted low-pass, see :py:func:`_lpf_response`.

    Returns
    -------
    HOL : ndarray, complex
    """
    f = _as_freq_array(f)
    w = 2 * np.pi * f
    wfs = _zoh(f, T)
    dac = _zoh(f, T)
    ctr = loop_gain / (1 - np.exp(-1j * w * T))
    rtc = np.exp(-1j * w * loop_delay)
    dm = _dm_response(f, dm_dc_gain, dm_f0, dm_csi)
    lpf = 1.0 if lpf_fc is None else _lpf_response(f, lpf_fc, T)
    return wfs * rtc * ctr * dac * dm * lpf


def closed_loop_sensitivity(f, T, loop_gain, loop_delay,
                             dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                             lpf_fc=None):
    """Closed-loop sensitivity (disturbance rejection) function
    ``S(f) = 1 / (1 + HOL(f))``. See :py:func:`open_loop_transfer_function`
    for the parameters."""
    hol = open_loop_transfer_function(f, T, loop_gain, loop_delay,
                                       dm_dc_gain, dm_f0, dm_csi, lpf_fc)
    return 1.0 / (1.0 + hol)


def avc_command_to_residual_tf(f, T, loop_gain, loop_delay,
                                avc_extra_delay=0.0,
                                dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                                lpf_fc=None):
    """Closed-loop transfer function from an AVC correction command
    (injected at the same point as the main loop's DM command) to the
    measured residual, ``P_avc(f) = WFS*RTC*delAVC*DAC*DM*LPF*S(f)``.

    Parameters
    ----------
    avc_extra_delay : float [s], optional
        Additional delay specific to AVC's own command path, beyond the
        main loop's ``loop_delay``. 0.0 (default) if AVC's correction is
        computed from the same measurement and combined into the DM
        command within the same simulation step as the main loop's (the
        architecture used in ``avc_full_demo``).

    See :py:func:`open_loop_transfer_function` for the remaining
    parameters (including ``lpf_fc``).

    Returns
    -------
    P_avc : ndarray, complex
    """
    f = _as_freq_array(f)
    w = 2 * np.pi * f
    wfs = _zoh(f, T)
    dac = _zoh(f, T)
    rtc = np.exp(-1j * w * loop_delay)
    del_avc = np.exp(-1j * w * avc_extra_delay)
    dm = _dm_response(f, dm_dc_gain, dm_f0, dm_csi)
    lpf = 1.0 if lpf_fc is None else _lpf_response(f, lpf_fc, T)
    s = closed_loop_sensitivity(f, T, loop_gain, loop_delay,
                                 dm_dc_gain, dm_f0, dm_csi, lpf_fc)
    return wfs * rtc * del_avc * dac * dm * lpf * s


def generate_avc_init_conditions(freq_avc, T, loop_gain, loop_delay,
                                  avc_extra_delay=0.0,
                                  dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                                  lpf_fc=None):
    """Compute a closed-loop-consistent initial state (``x0``, ``x1``) for
    :py:class:`~specula.processing_objects.avc.AVC`, in place of the
    generic default (``x0=1.0, x1=0.0``).

    Evaluates :py:func:`avc_command_to_residual_tf` at ``freq_avc`` and
    returns its real and imaginary parts. Port of ESO's
    ``generateAVCInitConditions.m``.

    Parameters
    ----------
    freq_avc : float [Hz]
        AVC's target vibration frequency (its ``freq`` constructor
        parameter for this instance).

    See :py:func:`avc_command_to_residual_tf` for the remaining
    parameters (including ``lpf_fc``).

    Returns
    -------
    x0, x1 : float
        Real and imaginary part of ``P_avc(freq_avc)``. Falls back to
        ``(1.0, 0.0)`` if ``P_avc`` evaluates to exactly zero at that
        frequency (degenerate/open-loop case).
    """
    p_avc = avc_command_to_residual_tf(freq_avc, T, loop_gain, loop_delay,
                                        avc_extra_delay, dm_dc_gain,
                                        dm_f0, dm_csi, lpf_fc)
    value = complex(p_avc[0])
    if abs(value) == 0:
        return 1.0, 0.0
    return float(value.real), float(value.imag)
