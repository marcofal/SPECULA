"""Divergence of the disturbance estimate when the plant phase violates the
90-degree margin condition (eq. `margin` of the report).

The exact component-wise AVC update is run with the plant half frozen at
P_hat = |P_star| exp(i(arg P_star + dphi)), so that |kappa| = 1 and the only
varying quantity is the phase error dphi. A single noiseless tone is used, so
the only mechanism at play is the one under test.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import avc_plotstyle as st

TS   = 1e-3                                     # loop period [s]
FVIB = 47.0                                     # vibration frequency [Hz]
GX   = 3.0                                      # state adaptation gain
PSTAR = 0.555 * np.exp(1j * np.radians(119.0))  # measured plant at 47 Hz
LSTAR = 120.0 * np.exp(1j * 0.7)                # disturbance phasor


def run(dphi_deg, tmax):
    """Exact AVC update, plant frozen at a phase error of dphi_deg."""
    Phat = abs(PSTAR) * np.exp(1j * (np.angle(PSTAR) + np.radians(dphi_deg)))
    kappa = np.conj(PSTAR / Phat)
    Linf = LSTAR / kappa
    x1, x2 = Phat.real, Phat.imag
    x3, x4 = 0.0, 0.0
    w, a, mu = 2 * np.pi * FVIB, 0.0, GX * TS
    n = int(tmax / TS)
    out = np.empty(n)
    for k in range(n):
        c, s = np.cos(a), np.sin(a)
        D = x1 * x1 + x2 * x2
        tc = -(x1 * x3 - x2 * x4) / D
        ts = -(x1 * x4 + x2 * x3) / D
        Z = (tc + 1j * ts) * np.conj(PSTAR) + LSTAR
        y = Z.real * c + Z.imag * s
        e = (tc * c + ts * s) * x1 + (ts * c - tc * s) * x2 + c * x3 + s * x4 - y
        x3 -= mu * c * e
        x4 -= mu * s * e
        a += w * TS
        out[k] = abs(x3 + 1j * x4 - Linf)
        if out[k] > 1e30:
            out[k:] = np.nan
            break
    return np.arange(n) * TS, out / abs(Linf)


# ---- panel (a): a few representative phase errors -------------------------
TMAX = 40.0
cases = [0, 60, 85, 89, 91, 95, 120]
# grey ladder plus distinct dashes: seven curves, no colour needed
sty = dict(zip(cases, st.styles(len(cases), markers=False)))
traj = {d: run(d, TMAX) for d in cases}

# ---- panel (b): measured growth rate against cos(dphi) --------------------
angles = np.arange(0, 181, 5.0)
rates = []
for d in angles:
    tmax = 40.0 if abs(d - 90) < 25 else 6.0   # slow rates need a long baseline
    t, v = run(d, tmax)
    m = (t > 0.1 * tmax) & (t < 0.95 * tmax) & np.isfinite(v) & (v > 1e-250)
    rates.append(-np.polyfit(t[m], np.log(v[m]), 1)[0] if m.sum() > 50 else np.nan)
rates = np.array(rates)

st.apply_rc(plt)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

a = ax[0]
for d in cases:
    t, v = traj[d]
    a.semilogy(t, np.maximum(v, 1e-16), lw=1.6, **sty[d],
               label=rf'$\Delta\varphi={d}^\circ$')
a.axhline(1.0, color='0.5', lw=0.8, ls=':')
a.set_xlabel('time [s]')
a.set_ylabel(r'$|\hat\Lambda-\hat\Lambda_\infty|\,/\,|\hat\Lambda_\infty|$')
a.set_ylim(1e-14, 1e10)
a.set_title(r'(a) disturbance error for a frozen, mis-phased plant', fontsize=10)
a.legend(fontsize=7.5, ncol=2, loc='lower left')

b = ax[1]
b.axhline(0, color='k', lw=0.8)
b.axvspan(0, 90, facecolor='0.90', edgecolor='none', zorder=0)
b.axvspan(90, 180, facecolor='0.97', edgecolor='none', zorder=0,
          hatch='///')
b.axvline(90, color='0.35', lw=1.2, ls='--')
b.plot(angles, 0.5 * GX * np.cos(np.radians(angles)), lw=2.0, color='0.62',
       label=r'prediction $\frac{1}{2}g_x\cos\Delta\varphi$')
b.plot(angles, rates, 'o', ms=4.5, color=st.COLORS[0], mfc='white',
       mew=1.2, label='measured')
b.set_xlabel(r'$\Delta\varphi=\arg\hat P-\arg P_\star$  [deg]')
b.set_ylabel(r'convergence rate  $\gamma\,\mathrm{Re}\,\kappa$  [s$^{-1}$]')
b.set_xlim(0, 180)
b.set_xticks([0, 45, 90, 135, 180])
b.set_title('(b) rate changes sign exactly at $90^\\circ$', fontsize=10)
b.text(0.05, 0.12, 'converges', transform=b.transAxes, fontsize=9, color='0.3')
b.text(0.72, 0.80, 'diverges', transform=b.transAxes, fontsize=9, color='0.3')
b.legend(fontsize=8, loc='center left')

for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('avc_margin_check.pdf')

print('  dphi   measured    predicted')
for i, d in enumerate(angles):
    if d % 15 == 0 or abs(d - 90) <= 10:
        print(f'{d:6.0f} {rates[i]:+10.4f} {0.5*GX*np.cos(np.radians(d)):+10.4f}')
