"""
Vibration peak detection from a residual PSD.

Ported from PASSATA's ``find_vib_peaks.pro`` / ``fit_ar2.pro`` (G. Agapito,
2022), itself derived from the Matlab ``optAVC`` library (P. Haguenauer)
and used by ESO's AVC calibration chain (``fitAR2MultipleRev.m``,
``computeAVCparamsModeCIAO.m``).

Given the PSD of a residual modal coefficient, this module finds narrowband
peaks that stand out above the smoothed noise floor, fits each one with a
second order (AR2 / damped resonance) model to extract its center
frequency, damping and integrated power, and discards peaks that don't
carry a significant fraction of the total power. The result is the set of
vibrations, and their initial frequency guesses, that an
:py:class:`~specula.processing_objects.avc.AVC` instance should be seeded
with -- this function performs *detection*, not tracking: the AVC object
itself refines the frequency estimate online once it is running.

Note
----
The Matlab source (which predates and matches the equations more closely)
was used as the primary reference where it disagreed with the IDL port,
which has a few transcription issues relative to it:

* ``fit_ar2`` in IDL calls the internal LS fit on a candidate secondary
  peak as ``fit_ar2_ls(PowerRef, PowerRef, T)``, passing the power array
  in place of the frequency array. The Matlab ``fitAR2MultipleRev.m``
  correctly uses ``AR2LeastSquare(PowerRef, FreqRef, T)``; this module
  follows the latter.
* IDL computes ``cumPow`` as a *cumulative* sum (``total(Power, /cum)``),
  while Matlab uses a plain scalar ``sum(Power)*df``. The scalar form is
  the one actually used in the comparison further down (against a single
  fitted peak's integrated power), so this module uses the scalar form.
* IDL's ``f_vibT(1,1)`` is a direct (and here, 0-indexing-incompatible)
  transliteration of Matlab's 1-indexed ``f_vibT(1,1)`` -- i.e. "the
  first stored frequency". This module uses ``f_vib[0]``, matching the
  Matlab intent of comparing the candidate secondary peak against the
  primary one.
* This module also keeps a safety check present in the Matlab
  ``AR2LeastSquare`` but missing from the IDL ``fit_ar2_ls``: if the AR2
  denominator is exactly zero at some frequency bin, the (otherwise
  unbounded) inverted PSD is not computed.
"""
from collections import namedtuple

import numpy as np


#: Result of :func:`find_vib_peaks`.
#:
#: freq : ndarray
#:     Estimated vibration frequencies [Hz], one per detected peak.
#: damping : ndarray
#:     Estimated AR2 damping ratio for each peak (dimensionless).
#: power : ndarray
#:     Estimated integrated power (variance) of each peak, in the same
#:     units as ``psd`` integrated over frequency.
VibPeaks = namedtuple('VibPeaks', ['freq', 'damping', 'power'])

_EMPTY_VIB_PEAKS = VibPeaks(freq=np.array([]), damping=np.array([]), power=np.array([]))


def _ar2_psd_model(freq, t, f_vib, k):
    """
    Power response of a discrete-time, second order (AR2) resonance
    centered at ``f_vib`` with damping ``k``, evaluated at frequencies
    ``freq`` [Hz], for a sampling period ``t`` [s].

    Matches ``AR2LeastSquare``'s inner model in ``fitAR2MultipleRev.m``
    (and ``fit_ar2_ls.pro``): a discrete complex pole pair at radius
    ``exp(-2*pi*k*f_vib*t)`` and angle ``2*pi*f_vib*t*sqrt(1-k**2)``.
    """
    a1 = 2 * np.exp(-2 * np.pi * k * f_vib * t) * np.cos(2 * np.pi * f_vib * t * np.sqrt(1 - k ** 2))
    a2 = -np.exp(-4 * np.pi * k * f_vib * t)
    denom = np.abs(1 - a1 * np.exp(-2j * np.pi * freq * t) - a2 * np.exp(-4j * np.pi * freq * t)) ** 2
    psd = np.zeros_like(denom)
    nonzero = denom != 0
    psd[nonzero] = denom[nonzero] ** (-1)
    return psd


def _ar2_least_squares(power, freq, t):
    """
    Fit a single AR2 resonance to a PSD segment ``(freq, power)``.

    The peak frequency is taken as the frequency of the maximum power
    sample (it is not part of the fit). The damping ratio ``k`` is found
    with a simple monotonic line search: starting from a near-zero value,
    ``k`` is increased in fixed steps as long as the least-squares fit
    residual keeps decreasing, and the last improving value is kept. This
    matches the coarse, fast search used operationally in
    ``AR2LeastSquare`` / ``fit_ar2_ls.pro`` -- it is not a general
    nonlinear optimizer.

    Parameters
    ----------
    power : ndarray
        PSD samples of the segment to fit.
    freq : ndarray
        Frequency vector [Hz] matching ``power``.
    t : float
        Sampling period [s] (``1 / Fs``).

    Returns
    -------
    psd : ndarray
        Fitted AR2 PSD model, evaluated at ``freq``.
    f_vib : float
        Estimated vibration frequency [Hz].
    k : float
        Estimated damping ratio.
    """
    power = np.asarray(power, dtype=float)
    freq = np.asarray(freq, dtype=float)

    peak_loc = int(np.argmax(power))
    peak_pwr = power[peak_loc]
    f_vib = freq[peak_loc]

    k = 0.0001
    k_step = 0.0005
    cost = 1e10
    num_iter = 0
    converging = True

    while converging and num_iter < 100:
        psd = _ar2_psd_model(freq, t, f_vib, k)
        if not np.any(psd):
            # Degenerate case: AR2 denominator vanished somewhere.
            # Matches the early-return guard in AR2LeastSquare.
            return psd, f_vib, k
        sigma0 = peak_pwr / np.max(psd)
        psd = sigma0 * psd
        new_cost = float(np.sum((psd - power) ** 2))
        if cost - new_cost > 0:
            cost = new_cost
            k += k_step
            num_iter += 1
        else:
            converging = False

    k = max(k - k_step, 0.0001)
    psd = _ar2_psd_model(freq, t, f_vib, k)
    sigma0 = np.max(power) / np.max(psd)
    psd = sigma0 * psd

    return psd, f_vib, k


def fit_ar2(power, freq, t, threshold, df):
    """
    Fit up to two AR2 resonances to a PSD segment.

    A first AR2 fit is performed on the whole segment. If the relative fit
    error exceeds ``threshold`` over a single compact (contiguous) sub-range
    of frequencies, a second AR2 fit is attempted on that enlarged
    sub-range, to capture a secondary peak that the first fit missed. The
    secondary peak is kept only if it carries at least 25% of the segment's
    total power and is separated from the primary peak by more than
    ``2 * df``.

    Ported from ``fitAR2MultipleRev.m`` / ``fit_ar2.pro``.

    Parameters
    ----------
    power : ndarray
        PSD samples of the segment to fit.
    freq : ndarray
        Frequency vector [Hz] matching ``power``.
    t : float
        Sampling period [s] (``1 / Fs``).
    threshold : float
        Relative-error threshold used to detect a poorly fit sub-range.
    df : float
        Frequency bin spacing [Hz], used to integrate power.

    Returns
    -------
    sigma : ndarray, shape (1,) or (2,)
        Integrated power of each fitted peak, sorted by ascending frequency.
    damping : ndarray, shape (1,) or (2,)
        Damping ratio of each fitted peak, sorted by ascending frequency.
    freq_vib : ndarray, shape (1,) or (2,)
        Frequency [Hz] of each fitted peak, sorted by ascending frequency.
    """
    power = np.asarray(power, dtype=float)
    freq = np.asarray(freq, dtype=float)

    sigma_t = [0.0]
    f_vib_t = [0.0]
    k_t = [0.0]

    cum_pow = float(np.sum(power) * df)

    psd, f_vib, k = _ar2_least_squares(power, freq, t)
    cum_psd = float(np.sum(psd) * df)
    sigma_t[0] = cum_psd
    f_vib_t[0] = f_vib
    k_t[0] = k

    # Check whether the fit leaves a compact (contiguous) region with a
    # large relative error: a sign that a second peak is hiding there.
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.abs(psd - power) / psd
    err = np.nan_to_num(err, nan=0.0, posinf=np.inf)
    bad = np.where(err > threshold)[0]

    is_compact = True
    if bad.size > 1:
        gaps = np.diff(bad)
        if np.any(gaps > 1):
            is_compact = False

    if bad.size > 0 and is_compact:
        init_ind = bad[0] - 1 if bad[0] > 0 else bad[0]
        end_ind = bad[-1] + 1 if bad[-1] < power.size - 1 else bad[-1]
        power_ref = power[init_ind:end_ind + 1]
        freq_ref = freq[init_ind:end_ind + 1]

        psd2, f_vib2, k2 = _ar2_least_squares(power_ref, freq_ref, t)
        cum_psd2 = float(np.sum(psd2) * df)

        if cum_psd2 > 0.25 * cum_pow and abs(f_vib_t[0] - f_vib2) > 2 * df:
            sigma_t.append(cum_psd2)
            f_vib_t.append(f_vib2)
            k_t.append(k2)

    sigma_t = np.asarray(sigma_t)
    f_vib_t = np.asarray(f_vib_t)
    k_t = np.asarray(k_t)

    order = np.argsort(f_vib_t)
    return sigma_t[order], k_t[order], f_vib_t[order]


def find_vib_peaks(psd, freq, fmin, fmax,
                    num_sampl_init=50,
                    smooth_alpha=0.99,
                    detection_threshold=3.0,
                    power_ratio_threshold=0.01,
                    fit_threshold=0.5):
    """
    Detect vibration peaks in a residual PSD.

    Processing steps (matching ``find_vib_peaks.pro`` /
    ``computeAVCparamsModeCIAO.m``):

    1. Restrict the PSD to the ``[fmin, fmax]`` band.
    2. Compute a smoothed estimate of the noise floor by averaging a
       forward and a backward exponential (single-pole) filter of the PSD.
       Using both directions avoids the smoothed floor lagging behind a
       peak's rising or falling edge.
    3. Flag samples exceeding ``detection_threshold`` times the smoothed
       floor, and group contiguous flagged frequency ranges together.
    4. Keep only groups whose integrated power exceeds
       ``power_ratio_threshold`` times the total in-band power.
    5. Fit each kept group with :func:`fit_ar2` (up to two AR2 peaks per
       group) and keep peaks below ``fmax`` whose total fitted power is
       less than twice the group's measured power (a sanity check against
       runaway fits).

    Parameters
    ----------
    psd : array-like
        Power spectral density of the residual signal.
    freq : array-like
        Frequency vector [Hz] matching ``psd`` (uniformly spaced).
    fmin, fmax : float
        Frequency band [Hz] to search for vibrations.
    num_sampl_init : int, optional
        Number of samples used to initialize the backward smoothing
        filter, and the minimum number of in-band samples required to
        proceed at all. Default: 50.
    smooth_alpha : float, optional
        Pole of the exponential smoothing filter used to estimate the
        noise floor, in (0, 1); higher is smoother/slower. Default: 0.99.
    detection_threshold : float, optional
        A sample is flagged as part of a peak if it exceeds
        ``detection_threshold`` times the smoothed floor. Default: 3.0.
    power_ratio_threshold : float, optional
        Minimum fraction of the total in-band power a candidate peak group
        must carry to be kept. Default: 0.01.
    fit_threshold : float, optional
        Relative-error threshold passed to :func:`fit_ar2` to decide
        whether a secondary peak should be fit within a group.
        Default: 0.5.

    Returns
    -------
    VibPeaks
        Namedtuple with ``freq`` [Hz], ``damping`` and ``power`` arrays,
        one entry per detected vibration. Empty arrays if none are found.
    """
    psd = np.asarray(psd, dtype=float)
    freq = np.asarray(freq, dtype=float)

    if psd.shape != freq.shape:
        raise ValueError(f'psd and freq must have the same shape, got {psd.shape} and {freq.shape}')

    fs = freq[-1] * 2
    dfreq = freq[1] - freq[0]
    expand_freq = dfreq * 1.1

    band = (freq >= fmin) & (freq <= fmax)
    n_red = int(np.count_nonzero(band))
    if n_red == 0 or n_red <= num_sampl_init:
        return _EMPTY_VIB_PEAKS

    freq_red = freq[band]
    psd_red = psd[band]
    cum_psd_total = float(np.sum(psd_red) * dfreq)

    # Forward and backward exponential smoothing of the PSD; the smoothed
    # noise floor is their average, to avoid a directional lag near peaks.
    psd_filt_fwd = np.empty(n_red)
    x = psd_red[0]
    for kk in range(n_red):
        x = smooth_alpha * x + (1 - smooth_alpha) * psd_red[kk]
        psd_filt_fwd[kk] = x

    psd_filt_bwd = np.empty(n_red)
    x = float(np.mean(psd_red[-num_sampl_init:]))
    for kk in range(n_red):
        y = psd_red[n_red - 1 - kk]
        x = smooth_alpha * x + (1 - smooth_alpha) * y
        psd_filt_bwd[kk] = x

    psd_smooth = (psd_filt_fwd + psd_filt_bwd[::-1]) / 2

    above = psd_red > detection_threshold * psd_smooth
    if not np.any(above):
        return _EMPTY_VIB_PEAKS

    samples = psd_red[above]
    samp_freq = freq_red[above]

    # Group contiguous (within expand_freq) detections together.
    if samp_freq.size == 1:
        group_bounds = np.array([0, 1])
    else:
        gaps = np.diff(samp_freq)
        breaks = np.where(gaps > expand_freq)[0]
        group_bounds = np.concatenate(([0], breaks + 1, [samp_freq.size]))

    power_content = np.array([
        np.sum(samples[group_bounds[ii]:group_bounds[ii + 1]]) * dfreq
        for ii in range(len(group_bounds) - 1)
    ])

    freq_list, k_list, sigma_list = [], [], []

    for ii in range(power_content.size):
        if power_content[ii] <= cum_psd_total * power_ratio_threshold:
            continue

        f_lo = samp_freq[group_bounds[ii]] - expand_freq
        f_hi = samp_freq[group_bounds[ii + 1] - 1] + expand_freq
        fit_mask = (freq_red > f_lo) & (freq_red < f_hi)
        freq_fit = freq_red[fit_mask]
        power_fit = psd_red[fit_mask]

        if freq_fit.size == 0:
            continue

        sigma_t, k_t, f_vib_t = fit_ar2(power_fit, freq_fit, 1.0 / fs, fit_threshold, dfreq)

        if np.sum(sigma_t) < 2 * power_content[ii] and np.max(f_vib_t) < fmax:
            for jj in range(f_vib_t.size):
                if f_vib_t[jj] > 0:
                    freq_list.append(f_vib_t[jj])
                    k_list.append(k_t[jj])
                    sigma_list.append(sigma_t[jj])

    if not freq_list:
        return _EMPTY_VIB_PEAKS

    return VibPeaks(freq=np.array(freq_list), damping=np.array(k_list), power=np.array(sigma_list))
