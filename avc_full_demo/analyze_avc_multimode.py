"""Per-mode comparison for the ten-channel AVCSimplified run.

Produces ``avc_multimode.pdf``:
  (a) residual RMS per mode, with and without the ten AVC channels;
  (b) the calibrated plant against frequency, which is what makes the two
      lowest channels the delicate ones;
  (c) the PLL estimate against time, showing the runaway at gomega = 0.2 and
      the lock at 0.05.

Reads the runs produced by ``run_avc_multimode.py``.
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

import avc_plotstyle as st

FS = 1000.0
STEADY = slice(5000, None)      # second half of a 10 s run
PLANT = json.load(open('avc_multimode_plant.json'))
FREQS = PLANT['freq']
N = len(FREQS)
AMPS = [80.0, 66.0, 55.0, 46.0, 38.0, 32.0, 26.0, 22.0, 18.0, 15.0]


def latest(tag):
    hits = sorted(glob.glob(f'output/MM_{tag}/*'))
    if not hits:
        raise FileNotFoundError(f'no run found for MM_{tag}')
    return hits[-1]


def modes(tag):
    return fits.getdata(latest(tag) + '/res_modes.fits').astype(float)


def freq_track(tag):
    return fits.getdata(latest(tag) + '/avc_freq.fits').astype(float)


base = modes('baseline')
avc = modes('avc_slowpll')          # the recommended configuration
fast = modes('avc')                 # gomega = 0.2, one channel unstable

rms_b = base[STEADY, :N].std(axis=0)
rms_a = avc[STEADY, :N].std(axis=0)
rms_f = fast[STEADY, :N].std(axis=0)

st.apply_rc(plt)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.7))

# ---- (a) the headline: RMS per mode ------------------------------------
a = ax[0]
x = np.arange(N)
C = st.palette(2)          # tab:blue / tab:orange; dashes do the separating
a.bar(x - 0.2, rms_b, 0.4, facecolor=C[1], edgecolor='k', lw=0.6,
      label='no AVC')
a.bar(x + 0.2, rms_a, 0.4, facecolor=C[0], edgecolor='k', lw=0.6,
      hatch='///', label='10 AVC channels')
a.set_yscale('log')
a.set_xticks(x)
a.set_xticklabels([f'{i}\n{f:.0f}' for i, f in enumerate(FREQS)], fontsize=7.5)
a.set_xlabel('mode / vibration frequency [Hz]')
a.set_ylabel('residual RMS [nm]')
a.set_title('(a) per-mode residual', fontsize=10)
a.legend(fontsize=8)
for i in range(N):
    a.text(i, max(rms_b[i], rms_a[i]) * 1.35,
           f'{20*np.log10(rms_a[i]/rms_b[i]):.0f}', ha='center', fontsize=7,
           color='0.25')

# ---- (b) the calibrated plant ------------------------------------------
b = ax[1]
b.plot(FREQS, PLANT['gain'], 'o-', color=C[0], ms=5, lw=1.6,
       label=r'$|\hat P|$')
b.set_xlabel('frequency [Hz]')
b.set_ylabel(r'$|\hat P|$  (solid, filled)', color=C[0])
b.tick_params(axis='y', labelcolor=C[0])
b2 = b.twinx()
b2.plot(FREQS, PLANT['phase'], 's', ls=st.DASHES[1], color=C[1], ms=5,
        mfc='white', mew=1.4, lw=1.6, label=r'$\arg\hat P$')
b2.set_ylabel(r'$\arg\hat P$ [deg]  (dashed, open)', color=C[1])
b2.tick_params(axis='y', labelcolor=C[1])
b2.grid(False)
b.set_title('(b) calibrated plant vs. frequency', fontsize=10)

# ---- (c) the frequency loop --------------------------------------------
c = ax[2]
t = np.arange(len(freq_track('avc'))) / FS
for tag, style, color, width, lab in [
        ('avc_slowpll', '-', C[1], 2.6, r'$g_\omega=0.05$'),
        ('avc', st.DASHES[1], C[0], 1.3, r'$g_\omega=0.2$')]:
    fr = freq_track(tag)
    for ch in range(N):
        c.plot(t, fr[:, ch], ls=style, lw=width, color=color,
               label=lab if ch == 0 else None)
c.set_xlabel('time [s]')
c.set_ylabel('frequency estimate [Hz]')
c.set_title('(c) the PLL is the fragile part', fontsize=10)
c.legend(fontsize=8, loc='upper left')

for p in ax:
    p.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('avc_multimode.pdf')

# ---- numbers -----------------------------------------------------------
print(f"{'mode':>4} {'f [Hz]':>7} {'amp':>6} {'no AVC':>9} {'AVC':>8} {'change':>9}")
for i in range(N):
    print(f'{i:4d} {FREQS[i]:7.0f} {AMPS[i]:6.0f} {rms_b[i]:9.2f} {rms_a[i]:8.2f} '
          f'{20*np.log10(rms_a[i]/rms_b[i]):8.1f} dB')

q = lambda v: np.sqrt((v ** 2).sum())
print(f'\nquadrature sum, modes 0-9 : {q(rms_b):8.1f} -> {q(rms_a):7.1f} nm '
      f'({20*np.log10(q(rms_a)/q(rms_b)):.1f} dB, {100*(q(rms_a)/q(rms_b)-1):.1f} %)')
print(f'with gomega = 0.2         : {q(rms_f):8.1f} nm  (channel 1 unstable)')
qa = np.sqrt((avc[STEADY, 10:].std(axis=0) ** 2).sum())
qb = np.sqrt((base[STEADY, 10:].std(axis=0) ** 2).sum())
print(f'modes 10-39, no vibration : {qb:8.2f} -> {qa:7.2f} nm  (unaffected)')
