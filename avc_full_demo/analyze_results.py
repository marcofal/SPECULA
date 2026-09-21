#!/usr/bin/env python3
"""
Analysis script for the AVC full closed-loop demo.

Compares the two latest runs found in ./output/:
  - the most recent AVC run (has avc_freq.fits / avc_comm.fits), i.e. the
    most recent Simul('params_scao_avc_full_demo.yml').run()
  - the most recent baseline run (no avc_*.fits), i.e. the most recent
    Simul('params_scao_vibration_baseline.yml').run()

and produces a main figure (avc_analysis.pdf) with:
  1. AVC frequency estimate vs. time, with the true injected vibration
     frequency as a reference line (frequency convergence).
  2. Mode 0 (tip, the vibrating mode) residual vs. time, baseline vs AVC,
     plus a moving-window RMS overlay to make the trend visible through
     the oscillation.
  3. Overall residual (RMS across all 40 modes) vs. time, baseline vs AVC,
     with the same moving-window RMS treatment.
  4. AVC's internal state magnitudes vs. time (requires 'out_state' in
     avc's outputs and 'avc_state-avc.out_state' in data_store's
     input_list): |x1+i*x2| (the online plant-response estimate) and
     |x3+i*x4| (the raw disturbance quadrature estimate), with the
     theta_min_energy safety-clamp floor marked. AVC's correction is
     essentially -(x3+i*x4)/conj(x1+i*x2): if the plant estimate's
     magnitude is small or noisy, that division amplifies noise into the
     correction -- this panel checks that directly instead of inferring
     it from the correction's own behaviour.

A second, separate figure (avc_psd_comparison.pdf) overlays the PSD of the
injected vibration (mode 0, as commanded -- "original") against the PSD of
the integrator-only (baseline) residual and the AVC-corrected residual, to
check directly whether AVC actually notches down the vibration frequency
(narrowband rejection) beyond what the integrator alone achieves,
independent of the broadband/mean-level comparisons in the main figure.
Each curve's peak near the vibration line is marked with its frequency and
PSD value.

Also prints summary numbers (frequency lock error, RMS residual over the
last third of the run, rejection at the vibration frequency in dB, etc.)
to stdout.

Usage (from this folder, after running both simulations at least once):

    python analyze_results.py
"""
import glob
import os
import sys

import numpy as np
import yaml
from astropy.io import fits
from scipy.signal import welch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, 'output')

MOVING_WINDOW_S = 0.1  # [s] moving-RMS window for the residual trend overlay
PSD_FMAX_PLOT = 150.0  # [Hz] crop the PSD panel to this frequency for readability


def find_runs():
    """Return (baseline_dir, avc_dir): the most recent run of each kind,
    read from output/no_AVC/ and output/with_AVC/ respectively."""
    no_avc_dir = os.path.join(OUTPUT_DIR, 'no_AVC')
    with_avc_dir = os.path.join(OUTPUT_DIR, 'with_AVC')

    baseline_dirs = sorted(glob.glob(os.path.join(no_avc_dir, '2*')))
    avc_dirs = sorted(glob.glob(os.path.join(with_avc_dir, '2*')))

    if not baseline_dirs:
        sys.exit(f"No baseline run found under {no_avc_dir}. "
                  "Run Simul('params_scao_vibration_baseline.yml').run() first.")
    if not avc_dirs:
        sys.exit(f"No AVC run found under {with_avc_dir}. "
                  "Run Simul('params_scao_avc_full_demo.yml').run() first.")

    return baseline_dirs[-1], avc_dirs[-1]


def load_run_params(run_dir):
    """Read back the resolved params.yml SPECULA stores with each run, so
    we pick up the actual time_step/vibration frequency used, rather than
    assuming fixed values that could drift out of sync with the .yml files."""
    with open(os.path.join(run_dir, 'params.yml')) as f:
        return yaml.safe_load(f)


def time_step_of(params):
    return float(params['main']['time_step'])


def true_vib_freq_of(params):
    try:
        return float(params['vib_wave']['freq'][0])
    except (KeyError, TypeError, IndexError):
        return None  # baseline-only params or vib_wave missing/renamed


def moving_rms(x, window):
    """Simple centered moving-window RMS, same length as x (edge-padded)."""
    if window < 2:
        return np.abs(x)
    kernel = np.ones(window) / window
    return np.sqrt(np.convolve(x**2, kernel, mode='same'))


def load_vib(run_dir):
    """Load the raw injected vibration command (vib.fits), if this run's
    DataStore was configured to save it. Returns None for older runs that
    predate this input_list entry."""
    path = os.path.join(run_dir, 'vib.fits')
    if not os.path.exists(path):
        return None
    return fits.getdata(path).ravel()


def compute_psd(x, fs):
    """Welch PSD estimate of a 1D signal sampled at fs [Hz]. Detrends the
    mean first so the DC/near-DC bin doesn't dominate the plotted range.
    Returns (freq [Hz], psd [nm^2/Hz])."""
    x = np.asarray(x, dtype=float)
    nperseg = min(4096, max(256, x.size // 4))
    freq, psd = welch(x - x.mean(), fs=fs, window='hann', nperseg=nperseg,
                       detrend='constant')
    return freq, psd


def psd_at(freq, psd, f0):
    """PSD value at the frequency bin closest to f0."""
    idx = np.argmin(np.abs(freq - f0))
    return psd[idx]


def psd_peak_near(freq, psd, f0, half_width=3.0):
    """(freq, psd) of the highest bin within +/- half_width [Hz] of f0,
    i.e. the vibration line's peak rather than just the nearest bin."""
    mask = np.abs(freq - f0) <= half_width
    if not np.any(mask):
        idx = np.argmin(np.abs(freq - f0))
        return freq[idx], psd[idx]
    idx_local = np.argmax(psd[mask])
    freq_masked = freq[mask]
    psd_masked = psd[mask]
    return freq_masked[idx_local], psd_masked[idx_local]


def load_avc_state(run_dir):
    """Load AVC's diagnostic internal state (avc_state.fits), if present
    (requires 'out_state' in avc's outputs and 'avc_state-avc.out_state'
    in data_store.inputs.input_list). Returns an (n, 4) array
    [x1, x2, x3, x4] for AVC instance 0, or None if not available."""
    path = os.path.join(run_dir, 'avc_state.fits')
    if not os.path.exists(path):
        return None
    state = fits.getdata(path)  # shape (n, n_avc, 4)
    return state[:, 0, :]


def main():
    baseline_dir, avc_dir = find_runs()
    print(f"Baseline run: {baseline_dir}")
    print(f"AVC run:      {avc_dir}")

    baseline_params = load_run_params(baseline_dir)
    avc_params = load_run_params(avc_dir)

    dt_baseline = time_step_of(baseline_params)
    dt_avc = time_step_of(avc_params)
    true_freq = true_vib_freq_of(avc_params) or true_vib_freq_of(baseline_params) or 47.0

    modes_baseline = fits.getdata(os.path.join(baseline_dir, 'res_modes.fits'))
    modes_avc = fits.getdata(os.path.join(avc_dir, 'res_modes.fits'))
    avc_freq = fits.getdata(os.path.join(avc_dir, 'avc_freq.fits')).ravel()
    avc_comm = fits.getdata(os.path.join(avc_dir, 'avc_comm.fits')).ravel()

    t_baseline = np.arange(modes_baseline.shape[0]) * dt_baseline
    t_avc = np.arange(modes_avc.shape[0]) * dt_avc
    t_avc_freq = np.arange(avc_freq.size) * dt_avc

    window_baseline = max(2, int(round(MOVING_WINDOW_S / dt_baseline)))
    window_avc = max(2, int(round(MOVING_WINDOW_S / dt_avc)))

    mode0_baseline = modes_baseline[:, 0]  # tip = vibrating mode
    mode0_avc = modes_avc[:, 0]
    rms_all_baseline = np.sqrt(np.mean(modes_baseline**2, axis=1))
    rms_all_avc = np.sqrt(np.mean(modes_avc**2, axis=1))

    # ---- summary printed to stdout ----
    last_third = slice(2 * len(avc_freq) // 3, None)
    print()
    print(f"True vibration frequency: {true_freq:.2f} Hz")
    print(f"AVC frequency estimate, last third of run: "
          f"mean={avc_freq[last_third].mean():.3f} Hz, "
          f"std={avc_freq[last_third].std():.3f} Hz, "
          f"error={avc_freq[last_third].mean() - true_freq:+.3f} Hz")
    print()

    def last_third_stats(x, dt):
        sl = slice(2 * len(x) // 3, None)
        return x[sl].std(), np.sqrt(np.mean(x[sl]**2))

    b_mode0_std, b_mode0_rms = last_third_stats(mode0_baseline, dt_baseline)
    a_mode0_std, a_mode0_rms = last_third_stats(mode0_avc, dt_avc)
    b_all_std, b_all_rms = last_third_stats(rms_all_baseline, dt_baseline)
    a_all_std, a_all_rms = last_third_stats(rms_all_avc, dt_avc)

    print("Mode 0 (tip) residual, last third of run:")
    print(f"  baseline: RMS={b_mode0_rms:8.2f} nm  std={b_mode0_std:8.2f} nm")
    print(f"  AVC:      RMS={a_mode0_rms:8.2f} nm  std={a_mode0_std:8.2f} nm")
    print(f"  change:   {100 * (a_mode0_rms - b_mode0_rms) / b_mode0_rms:+.1f} % RMS")
    print()
    print("Overall residual (RMS across all 40 modes), last third of run:")
    print(f"  baseline: RMS={b_all_rms:8.2f} nm  std={b_all_std:8.2f} nm")
    print(f"  AVC:      RMS={a_all_rms:8.2f} nm  std={a_all_std:8.2f} nm")
    print(f"  change:   {100 * (a_all_rms - b_all_rms) / b_all_rms:+.1f} % RMS")

    # ---- PSD at the vibration frequency: is either config actually
    # notching it down relative to the raw injected disturbance? ----
    vib_baseline = load_vib(baseline_dir)
    vib_avc = load_vib(avc_dir)
    vib = vib_avc if vib_avc is not None else vib_baseline
    have_psd = vib is not None

    if have_psd:
        fs_baseline = 1.0 / dt_baseline
        fs_avc = 1.0 / dt_avc
        freq_vib, psd_vib = compute_psd(vib, fs_avc if vib_avc is not None else fs_baseline)
        freq_b, psd_b = compute_psd(mode0_baseline, fs_baseline)
        freq_a, psd_a = compute_psd(mode0_avc, fs_avc)

        p_vib_f0 = psd_at(freq_vib, psd_vib, true_freq)
        p_b_f0 = psd_at(freq_b, psd_b, true_freq)
        p_a_f0 = psd_at(freq_a, psd_a, true_freq)

        print()
        print(f"PSD at the vibration frequency ({true_freq:.1f} Hz):")
        print(f"  input (injected vib.):  {p_vib_f0:.3e} nm^2/Hz")
        print(f"  baseline residual:      {p_b_f0:.3e} nm^2/Hz  "
              f"({10*np.log10(p_b_f0/p_vib_f0):+.1f} dB vs. input)")
        print(f"  AVC residual:           {p_a_f0:.3e} nm^2/Hz  "
              f"({10*np.log10(p_a_f0/p_vib_f0):+.1f} dB vs. input)")
        print(f"  AVC vs. baseline at f0: {10*np.log10(p_a_f0/p_b_f0):+.1f} dB "
              f"({'less' if p_a_f0 < p_b_f0 else 'more'} residual power with AVC)")
    else:
        print()
        print("No vib.fits found in one or both runs (add 'vib-vib_wave.output' "
              "to data_store.inputs.input_list and re-run to enable the PSD panel).")

    # ---- AVC internal state: is the online plant estimate (x1,x2) stable? ----
    avc_state = load_avc_state(avc_dir)
    have_state = avc_state is not None

    if have_state:
        t_state = np.arange(avc_state.shape[0]) * dt_avc
        x1, x2, x3, x4 = avc_state[:, 0], avc_state[:, 1], avc_state[:, 2], avc_state[:, 3]
        plant_mag = np.sqrt(x1**2 + x2**2)
        dist_mag = np.sqrt(x3**2 + x4**2)
        theta_min_energy = float(avc_params.get('avc', {}).get('theta_min_energy', 1e-2))
        clamp_floor = np.sqrt(theta_min_energy)
        clamped_fraction = np.mean(plant_mag**2 < theta_min_energy)

        print()
        print("AVC internal state (last two-thirds of run, past initial transient):")
        st = slice(len(plant_mag) // 3, None)
        print(f"  plant estimate |x1+i*x2|:      min={plant_mag[st].min():.4f}  "
              f"mean={plant_mag[st].mean():.3f}  max={plant_mag[st].max():.3f}")
        # a converging disturbance estimate flattens out; measure the growth
        # rate over the last third vs. the third before it as a settling check
        third = len(dist_mag) // 3
        d_prev = dist_mag[third:2 * third]
        d_last = dist_mag[2 * third:]
        growth_prev = d_prev[-1] - d_prev[0]
        growth_last = d_last[-1] - d_last[0]
        settling = ("settling (growth per third decreasing)"
                    if abs(growth_last) < 0.6 * abs(growth_prev) + 1e-9
                    else "still ramping / not settling")
        print(f"  disturbance estimate |x3+i*x4|: start={dist_mag[st][0]:.2f}  "
              f"end={dist_mag[st][-1]:.2f}  ({settling})")
        print(f"  safety clamp (|x1+i*x2|^2 < theta_min_energy={theta_min_energy:g}) "
              f"active {100*clamped_fraction:.1f}% of the run")
    else:
        print()
        print("No avc_state.fits found (add 'out_state' to avc's outputs and "
              "'avc_state-avc.out_state' to data_store.inputs.input_list, "
              "then re-run, to enable the internal-state panel).")

    # ---- figure ----
    n_panels = 3 + int(have_state)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.6 * n_panels), sharex=False)

    ax = axes[0]
    ax.plot(t_avc_freq, avc_freq, color='tab:blue', lw=1, label='AVC frequency estimate')
    ax.axhline(true_freq, color='k', ls='--', lw=1, label=f'true vibration freq ({true_freq:.1f} Hz)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('AVC frequency convergence')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_baseline, mode0_baseline, color='tab:orange', lw=0.25, alpha=0.15)
    ax.plot(t_avc, mode0_avc, color='tab:blue', lw=0.25, alpha=0.15)
    ax.plot(t_baseline, moving_rms(mode0_baseline, window_baseline),
            color='tab:orange', lw=1.8, label=f'baseline (moving RMS, {MOVING_WINDOW_S*1000:.0f} ms window)')
    ax.plot(t_avc, moving_rms(mode0_avc, window_avc),
            color='tab:blue', lw=1.8, label=f'AVC (moving RMS, {MOVING_WINDOW_S*1000:.0f} ms window)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Residual [nm]')
    ax.set_title('Mode 0 (tip, vibrating mode) residual vs. time\n(faint: raw signal, bold: moving RMS trend)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_baseline, rms_all_baseline, color='tab:orange', lw=0.25, alpha=0.15)
    ax.plot(t_avc, rms_all_avc, color='tab:blue', lw=0.25, alpha=0.15)
    ax.plot(t_baseline, moving_rms(rms_all_baseline, window_baseline),
            color='tab:orange', lw=1.8, label=f'baseline (moving RMS, {MOVING_WINDOW_S*1000:.0f} ms window)')
    ax.plot(t_avc, moving_rms(rms_all_avc, window_avc),
            color='tab:blue', lw=1.8, label=f'AVC (moving RMS, {MOVING_WINDOW_S*1000:.0f} ms window)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RMS residual [nm]')
    ax.set_title('Overall residual (RMS across all 40 modes) vs. time\n(faint: raw signal, bold: moving RMS trend)')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    next_panel = 3
    if have_state:
        ax = axes[next_panel]
        next_panel += 1
        ax.semilogy(t_state, plant_mag, color='tab:red', lw=1,
                   label='plant estimate |x1+i*x2| (should converge & stay put)')
        ax.semilogy(t_state, dist_mag, color='tab:green', lw=1,
                   label='disturbance estimate |x3+i*x4|')
        ax.axhline(clamp_floor, color='k', ls=':', lw=1, alpha=0.7,
                   label=f'safety-clamp floor (sqrt(theta_min_energy)={clamp_floor:.3f})')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('State magnitude [nm or a.u.]')
        ax.set_title("AVC internal state: is the online plant estimate (x1,x2) stable?\n"
                     "(a stable estimator would flatten out; a wandering/near-zero plant "
                     "estimate here explains a growing correction)")
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    out_path = os.path.join(HERE, 'avc_analysis.pdf')
    fig.savefig(out_path, dpi=130)
    print(f"\nFigure saved to {out_path}")

    # ---- separate PSD comparison figure: input vs. AVC vs. integrator-only,
    # with each curve's peak near the vibration frequency marked ----
    if have_psd:
        peak_vib = psd_peak_near(freq_vib, psd_vib, true_freq)
        peak_b = psd_peak_near(freq_b, psd_b, true_freq)
        peak_a = psd_peak_near(freq_a, psd_a, true_freq)

        fig_psd, ax = plt.subplots(figsize=(9, 5.5))
        ax.semilogy(freq_vib, psd_vib, color='k', lw=1.2, label='original (injected vibration, mode 0)')
        ax.semilogy(freq_b, psd_b, color='tab:orange', lw=1.2, label='integrator-only  (mode 0)')
        ax.semilogy(freq_a, psd_a, color='tab:blue', lw=1.2, label='AVC-corrected  (mode 0)')
        ax.axvline(true_freq, color='k', ls='--', lw=1, alpha=0.5,
                   label=f'({true_freq:.1f} Hz)')

        # stack labels above their own marker in peak-height order so they
        # don't overlap when the three peaks sit close together in frequency
        peaks = sorted([(peak_vib, 'k'), (peak_b, 'tab:orange'), (peak_a, 'tab:blue')],
                       key=lambda item: item[0][1])
        for rank, ((fp, pp), color) in enumerate(peaks):
            ax.plot(fp, pp, marker='o', ms=6, color=color, mec='k', mew=0.6, zorder=5)
            ax.annotate(f'{fp:.1f} Hz, {pp:.2e}', xy=(fp, pp),
                        xytext=(10, 10 + 22 * rank), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.7, alpha=0.7))

        ax.set_xlim(0, PSD_FMAX_PLOT)
        # ymin, ymax = ax.get_ylim()
        # ax.set_ylim(ymin, ymax * 1e4)  # headroom for the stacked peak labels
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('PSD [nm$^2$/Hz]')
        ax.set_title('Mode 0 PSD comparison', pad=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3, which='both')
        fig_psd.tight_layout()

        psd_out_path = os.path.join(HERE, 'avc_psd_comparison.pdf')
        fig_psd.savefig(psd_out_path, dpi=130)
        print(f"PSD comparison figure saved to {psd_out_path}")

    plt.show()


if __name__ == '__main__':
    main()
