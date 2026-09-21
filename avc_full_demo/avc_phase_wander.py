"""Plot the wandering of arg(P_hat) relative to the +-90 deg stability cone.

Panel (a): phase error  dphi = arg(P_hat) - arg(P_star)  for the original
           algorithm (plant adapted) and for the exact-gradient variant.
Panel (b): |Lambda_hat| for the adapted and the frozen-plant runs.
"""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

TS = 1e-3                       # loop period [s]
ARG_PSTAR = 169.52              # measured plant phase at 47 Hz [deg]

def load(run):
    x = fits.getdata(glob.glob(run + '/*/avc_state.fits')[0]).astype(float)[:, 0, :]
    P = x[:, 0] + 1j * x[:, 1]
    L = x[:, 2] + 1j * x[:, 3]
    return np.arange(len(x)) * TS, P, L

t_d, P_d, L_d = load('cmp_D_orig_right')
t_e, P_e, L_e = load('cmp_E_hyb_right')
t_f, P_f, L_f = load('cmp_F_frozen')

def dphi(P):
    return (np.degrees(np.angle(P)) - ARG_PSTAR + 180.0) % 360.0 - 180.0

d_d, d_e = dphi(P_d), dphi(P_e)
inside = np.mean(np.abs(d_d) < 90.0)
turns = np.ptp(np.degrees(np.unwrap(np.angle(P_d)))) / 360.0

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.5))

# --- (a) zoom on the first 0.5 s: individual excursions out of the cone -----
a = ax[0]
m = t_d <= 0.5
a.axhspan(-90, 90, color='tab:green', alpha=0.13, zorder=0)
a.axhline(90, color='tab:green', lw=1.2, ls='--')
a.axhline(-90, color='tab:green', lw=1.2, ls='--')
a.plot(t_d[m], d_d[m], lw=0.9, color='tab:red', label='original (plant adapted)')
a.plot(t_e[m], d_e[m], lw=2.0, color='tab:blue', label='exact gradient (radial)')
a.set_ylim(-180, 180); a.set_yticks([-180, -90, 0, 90, 180])
a.set_xlabel('time [s]')
a.set_ylabel(r'$\Delta\varphi=\arg\hat P-\arg P_\star$  [deg]')
a.set_title('(a) phase error vs. cone, first 0.5 s', fontsize=10)
a.legend(fontsize=7.5, loc='lower right')

# --- (b) cumulative unwrapped phase over the whole run ---------------------
b = ax[1]
b.plot(t_d, np.degrees(np.unwrap(np.angle(P_d))) / 360.0, lw=1.2,
       color='tab:red', label='original (plant adapted)')
b.plot(t_e, np.degrees(np.unwrap(np.angle(P_e))) / 360.0, lw=1.6,
       color='tab:blue', label='exact gradient (radial)')
b.set_xlabel('time [s]')
b.set_ylabel(r'unwrapped $\arg\hat P$  [turns]')
b.set_title('(b) accumulated rotation of the plant estimate', fontsize=10)
b.text(0.04, 0.55, f'{turns:.0f} turns in 10 s\ninside cone {100*inside:.0f}% of the time',
       transform=b.transAxes, fontsize=8,
       bbox=dict(fc='white', ec='0.7', alpha=0.85))
b.legend(fontsize=7.5, loc='lower right')

# --- (c) consequence on the disturbance state ------------------------------
c = ax[2]
c.plot(t_d, np.abs(L_d), lw=1.2, color='tab:red', label='original (plant adapted)')
c.plot(t_f, np.abs(L_f), lw=1.2, color='tab:green', label='plant frozen')
c.plot(t_e, np.abs(L_e), lw=1.2, color='tab:blue', label='exact gradient')
c.set_xlabel('time [s]'); c.set_ylabel(r'$|\hat\Lambda|$')
c.set_title(r'(c) disturbance state: ramp vs. settling', fontsize=10)
c.legend(fontsize=7.5, loc='upper left')

for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('avc_phase_wander.pdf')
print('inside cone %.3f, turns %.1f' % (inside, turns))
