"""Steady-state mode-0 spectra for the ablation of Sec. 2 of the report.

Regenerates ``avc_ablation_psd.pdf`` from the ``cmp_*`` runs, in the
grey-scale-safe style of ``avc_plotstyle``. Curves only, no annotated peak
values: the numbers quoted in the report come from the ablation table, which
uses its own estimator.
"""
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

import avc_plotstyle as st

FS = 1000.0
FVIB = 47.0

# label -> run directory, in the order of the ablation table
RUNS = [
    ('no AVC',                                    'cmp_A2_baseline'),
    (r'$x_3,x_4$; $\omega$ fixed wrong (45 Hz)',  'cmp_G_nopll_wrongf'),
    (r'$\omega,\alpha+x_3,x_4+x_1,x_2$ (published)', 'cmp_D2_pll_adapt'),
    (r'$x_3,x_4$; $\omega$ fixed true (47 Hz)',   'cmp_H_nopll_rightf'),
    (r'$\omega,\alpha+x_3,x_4$ (recommended)',    'cmp_F2_pll_frozen'),
]


def load(run, name='res_modes'):
    path = glob.glob(f'{run}/*/{name}.fits')
    if not path:
        return None
    a = fits.getdata(path[0]).astype(float)
    return a[:, 0] if a.ndim == 2 else a.ravel()


def spectrum(x):
    h = x[len(x) // 2:]
    w = np.hanning(len(h))
    return np.fft.rfftfreq(len(h), 1 / FS), 2 * np.abs(np.fft.rfft(h * w)) / w.sum()


st.apply_rc(plt)
fig, ax = plt.subplots(figsize=(8.2, 4.6))

vib = load(RUNS[0][1], 'vib')
if vib is not None:
    f, s = spectrum(vib)
    ax.loglog(f[1:], s[1:], lw=2.6, color='0.78', zorder=0,
              label='injected disturbance')

sty = st.styles(len(RUNS), markers=False)
for k, (label, run) in enumerate(RUNS):
    y = load(run)
    if y is None:
        print(f'missing: {run}')
        continue
    f, s = spectrum(y)
    ax.loglog(f[1:], s[1:], lw=1.1, **sty[k], label=label)

ax.axvline(FVIB, color='0.5', lw=0.8, ls=':')
ax.text(FVIB * 1.05, 1.4e-3, r'$f_0$', fontsize=8, color='0.35')
ax.set_xlim(2, FS / 2)
ax.set_ylim(1e-3, 3e3)
ax.set_xlabel('frequency [Hz]')
ax.set_ylabel('mode-0 amplitude [nm]')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=8, loc='lower left')
fig.tight_layout()
fig.savefig('avc_ablation_psd.pdf')
print('written avc_ablation_psd.pdf')
