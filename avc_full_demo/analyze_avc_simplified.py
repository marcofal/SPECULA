"""Figures for the AVCSimplified section of the report.

Produces
--------
avc_simplified_psd.pdf
    Mode-0 residual spectrum with the simplified AVC on and off, against the
    injected disturbance.
avc_simplified_margin.pdf
    Behaviour as a function of the plant-calibration phase error: steady
    residual and disturbance-state trajectories, in and out of the 90 degree
    stability cone.

Both read the runs produced by ``run_avc_simplified.py``.
"""
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

import avc_plotstyle as st

st.apply_rc(plt)

FS = 1000.0        # [Hz] loop rate
FVIB = 47.0        # [Hz] vibration frequency

# The plant calibration is measured by injecting a probe at the command
# combiner, one node downstream of the AVC output, so it does not include the
# AVC's own one-sample output delay. At 47 Hz that delay is
# 360 * FVIB / FS = 16.9 deg, and the measured stability cone is centred
# there rather than at zero (see the report).
DELAY_DEG = 360.0 * FVIB / FS

# run name -> nominal phase error relative to the calibrated value
SWEEP = {
    'cm80': -96.9, 'sm85': -85.0, 'sm60': -60.0, 'sm30': -30.0,
    'c00': -16.9, 'f0': 0.0, 'sp30': 30.0, 'cp80': 63.1, 'f60': 60.0,
    'sp70': 70.0, 'sp80': 80.0, 'f85': 85.0, 'f95': 95.0, 'f120': 120.0,
}


def latest(name):
    hits = sorted(glob.glob(f'output/SIMPL_{name}/*'))
    if not hits:
        raise FileNotFoundError(f'no run found for {name}')
    return hits[-1]


def residual(name):
    return fits.getdata(latest(name) + '/res_modes.fits').astype(float)[:, 0]


def disturbance(name):
    return fits.getdata(latest(name) + '/vib.fits').astype(float).ravel()


def lam(name):
    s = fits.getdata(latest(name) + '/avc_state.fits').astype(float)[:, 0, :]
    return np.abs(s[:, 2] + 1j * s[:, 3])


def psd(x, fs=FS):
    """One-sided amplitude spectrum of the steady-state half of x."""
    h = x[len(x) // 2:]
    w = np.hanning(len(h))
    spec = 2 * np.abs(np.fft.rfft(h * w)) / w.sum()
    return np.fft.rfftfreq(len(h), 1 / fs), spec


def line_peak(x, f0=FVIB):
    """Location and height of the spectral line nearest f0."""
    f, s = psd(x)
    i = int(np.argmin(abs(f - f0)))
    lo = max(0, i - 2)
    j = lo + int(np.argmax(s[lo:i + 3]))
    return f[j], s[j]


def line_amplitude(x, f0=FVIB):
    return line_peak(x, f0)[1]


# --------------------------------------------------------------- figure 1
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))

f, s_dist = psd(disturbance('d0'))
_, s_base = psd(residual('baseline'))
_, s_avc = psd(residual('d0'))

a = ax[0]
a.loglog(f[1:], s_dist[1:], lw=1.4, color='0.62', label='injected disturbance')
a.loglog(f[1:], s_base[1:], lw=0.9, **st.line(0, 2), label='no AVC')
a.loglog(f[1:], s_avc[1:], lw=0.9, **st.line(1, 2), label='AVCSimplified')
for k, lab in ((1, r'$f_0$'), (3, r'$3f_0$'), (5, r'$5f_0$')):
    a.axvline(k * FVIB, color='0.4', lw=0.8, ls=':')
    a.text(k * FVIB * 1.05, 1.4e-3, lab, fontsize=8, color='0.35')
a.set_xlabel('frequency [Hz]')
a.set_ylabel('mode-0 amplitude [nm]')
a.set_xlim(2, FS / 2)
a.set_ylim(1e-3, 3e3)
a.set_title('(a) residual spectrum', fontsize=10)
# Anchor each label on the actual peak bin, not on the nominal frequency,
# and keep the leader lines short so they do not cross the data.
for tag, color, dy in (('baseline', st.COLORS[0], 1.0),
                       ('d0', st.COLORS[1], 5.0)):
    fp, ap = line_peak(residual(tag))
    a.annotate(f'{ap:.1f} nm', xy=(fp, ap), xytext=(fp * 0.30, ap * dy),
               fontsize=8, color=color, ha='right', va='center',
               arrowprops=dict(arrowstyle='-', lw=0.7, color=color,
                               shrinkA=2, shrinkB=2))
a.legend(fontsize=8, loc='lower left')

b = ax[1]
t = np.arange(len(residual('baseline'))) / FS
b.plot(t, residual('baseline'), lw=0.4, color=st.COLORS[0], label='no AVC')
b.plot(t, residual('d0'), lw=0.4, color=st.COLORS[1], label='AVCSimplified')
b.set_xlabel('time [s]')
b.set_ylabel('mode-0 residual [nm]')
b.set_title('(b) time series', fontsize=10)
b.legend(fontsize=8, loc='upper right')

for x in ax:
    x.grid(alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('avc_simplified_psd.pdf')

print(f'baseline RMS   = {residual("baseline")[5000:].std():8.1f} nm')
print(f'AVC RMS        = {residual("d0")[5000:].std():8.1f} nm')
print(f'47 Hz, no AVC  = {line_amplitude(residual("baseline")):8.1f} nm')
print(f'47 Hz, AVC     = {line_amplitude(residual("d0")):8.1f} nm')

# Harmonics: the injected tone is a pure sine, so anything at k*f0 is
# generated inside the loop (WFS centroid saturation, odd-symmetric hence
# odd harmonics only). Removing the fundamental removes the excursion that
# drives it, so the harmonics go too.
print('\n harmonic   f [Hz]    input   no AVC      AVC   change')
for k in range(1, 8):
    f0 = k * FVIB
    if f0 > FS / 2:
        break
    ib = line_amplitude(residual('baseline'), f0)
    ia = line_amplitude(residual('d0'), f0)
    iv = line_amplitude(disturbance('d0'), f0)
    print(f'{k:9d} {f0:8.1f} {iv:8.3f} {ib:8.2f} {ia:8.3f} '
          f'{20*np.log10(ia/ib):7.1f} dB')

# --------------------------------------------------------------- figure 2
names = sorted(SWEEP, key=lambda n: SWEEP[n])
dphi = np.array([SWEEP[n] + DELAY_DEG for n in names])   # corrected frame
rms = np.array([residual(n)[5000:].std() for n in names])
base_rms = residual('baseline')[5000:].std()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))

a = ax[0]
a.axvspan(-90, 90, facecolor='0.90', edgecolor='none', zorder=0)
for v in (-90, 90):
    a.axvline(v, color='0.45', lw=1.2, ls='--')
a.axhline(base_rms, color='0.45', lw=1.2, ls=':', label='no AVC')
inside = np.abs(dphi) < 90
a.semilogy(dphi[inside], rms[inside], ms=5, lw=1.4, **st.series(0, 2),
           label='inside the cone')
a.semilogy(dphi[~inside], rms[~inside], ms=7,
           **dict(st.series(1, 2), linestyle='none', markeredgecolor='k',
                  markeredgewidth=0.6), label='outside')
a.set_xlabel(r'plant phase error $\Delta\varphi$  [deg]')
a.set_ylabel('steady mode-0 RMS [nm]')
a.set_title('(a) rejection against calibration phase error', fontsize=10)
a.legend(fontsize=8, loc='lower right')

b = ax[1]
for k, (n, lab) in enumerate([('c00', r'$\Delta\varphi=0^\circ$'),
                              ('sp30', r'$\Delta\varphi=+47^\circ$'),
                              ('cp80', r'$\Delta\varphi=+80^\circ$'),
                              ('sp80', r'$\Delta\varphi=+97^\circ$'),
                              ('f120', r'$\Delta\varphi=+137^\circ$')]):
    v = lam(n)
    b.plot(np.arange(len(v)) / FS, v, lw=1.6, **st.line(k, 5), label=lab)
b.set_xlabel('time [s]')
b.set_ylabel(r'$|\hat\Lambda|$')
b.set_title(r'(b) disturbance state: settling vs. ramping', fontsize=10)
b.legend(fontsize=8, loc='upper left')

for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('avc_simplified_margin.pdf')

print('\n  dphi(corr)   RMS [nm]   47 Hz [nm]')
for n in names:
    print(f'{SWEEP[n] + DELAY_DEG:+10.1f} {residual(n)[5000:].std():10.1f} '
          f'{line_amplitude(residual(n)):11.1f}   ({n})')
