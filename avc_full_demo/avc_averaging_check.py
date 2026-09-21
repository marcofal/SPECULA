"""Why the one-cycle average is legitimate, and what it costs.

The exact update is periodically time-varying: the per-sample increment of
the disturbance state contains a term rotating at twice the vibration
frequency. Averaging that term away replaces the exact recursion by an
autonomous linear ODE. This script checks the two things that makes that
step legitimate:

(a) the averaged solution tracks the exact recursion,
(b) the discrepancy is O(mu), mu = gx*Ts being the small parameter, so it
    vanishes in the slow-adaptation limit the method assumes.

Produces avc_averaging.pdf.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import avc_plotstyle as st

TS = 1e-3
FVIB = 47.0
W = 2 * np.pi * FVIB
PSTAR = 0.555 * np.exp(1j * np.radians(119.0))
LSTAR = 120.0 * np.exp(0.7j)
PHAT = PSTAR                       # exact calibration, so kappa = 1
KAPPA = np.conj(PSTAR / PHAT)
LINF = LSTAR / KAPPA


def exact(gx, n):
    """The recursion as the object actually runs it."""
    mu, lam, alpha = gx * TS, 0j, 0.0
    out = np.empty(n, dtype=complex)
    for k in range(n):
        z = (-lam / np.conj(PHAT)) * np.conj(PSTAR) + LSTAR
        y = (z * np.exp(-1j * alpha)).real
        lam += mu * np.exp(1j * alpha) * y
        out[k] = lam
        alpha += W * TS
    return out


def averaged(gx, n):
    """Solution of the averaged ODE  dL/dt = gamma (Lstar - kappa L)."""
    t = np.arange(1, n + 1) * TS
    return LINF + (0 - LINF) * np.exp(-(gx / 2) * KAPPA * t)


GX = 3.0
N = 8000
xe, xa = exact(GX, N), averaged(GX, N)
t = np.arange(1, N + 1) * TS

st.apply_rc(plt)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))

a = ax[0]
EX = st.COLORS[1]          # tab:orange, the thick 'exact' trace
a.plot(t, xe.real, lw=2.4, color=EX, solid_capstyle='round',
       label=r'exact recursion, $\mathrm{Re}\,\hat\Lambda$')
a.plot(t, xe.imag, lw=2.4, color=EX, ls=(0, (7, 2)),
       label=r'exact, $\mathrm{Im}\,\hat\Lambda$')
a.plot(t, xa.real, lw=1.0, ls='-', color='k', label='averaged ODE')
a.plot(t, xa.imag, lw=1.0, ls=(0, (5, 1.6)), color='k')
a.set_xlabel('time [s]')
a.set_ylabel(r'$\hat\Lambda$')
a.set_title(r'(a) the two agree at the loop scale', fontsize=10)
a.legend(fontsize=7.5, loc='center right')

b = ax[1]
m = (t > 0.30) & (t < 0.40)
b.plot(t[m], xe.real[m], lw=2.4, color=EX, label='exact')
b.plot(t[m], xa.real[m], lw=1.1, ls=(0, (5, 1.6)), color='k', label='averaged')
b.set_xlabel('time [s]')
b.set_ylabel(r'$\mathrm{Re}\,\hat\Lambda$')
b.set_title(r'(b) zoom: the $2\omega$ ripple that is averaged away', fontsize=10)
b.legend(fontsize=8, loc='lower right')

# --- error scaling -------------------------------------------------------
gains = np.array([24.0, 12.0, 6.0, 3.0, 1.5, 0.75, 0.375])
errs = []
for g in gains:
    n = int(8.0 / (g / 2) / TS)          # same number of time constants
    errs.append(np.abs(exact(g, n) - averaged(g, n)).max())
errs = np.array(errs)
mus = gains * TS

c = ax[2]
ref = errs[3] / abs(LINF) * (mus / mus[3])
c.loglog(mus, ref, lw=1.6, color=EX, ls=(0, (5, 1.6)), label=r'$\propto\mu$')
c.loglog(mus, errs / abs(LINF), 'o', ms=6, color='k', mfc='white',
         mew=1.3, label='measured')
c.axvline(GX * TS, color='0.35', lw=1.0, ls=':')
c.text(GX * TS * 1.15, 3e-3, r'$g_x=3$', fontsize=8, color='0.35')
c.set_xlabel(r'$\mu=g_xT_s$')
c.set_ylabel(r'max $|$exact$-$averaged$|\,/\,|\hat\Lambda_\infty|$')
c.set_title(r'(c) the error is $O(\mu)$, as the theorem says', fontsize=10)
c.legend(fontsize=8, loc='upper left')

for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('avc_averaging.pdf')

print(f'timescale separation: 1/gamma = {2/GX:.3f} s vs pi/omega = '
      f'{np.pi/W*1e3:.2f} ms  ->  {(2/GX)/(np.pi/W):.0f}')
print(f'max deviation at gx={GX}: {np.abs(xe-xa).max():.4f} '
      f'({np.abs(xe-xa).max()/abs(LINF)*100:.2f} % of |Lambda_inf|)')
print(f'steady ripple: {np.ptp(np.abs(xe[-2000:]))/2/abs(LINF)*100:.4f} % of |Lambda_inf|')
print('\n   mu        max error / |Lambda_inf|     ratio to mu')
for mu, e in zip(mus, errs):
    print(f'{mu:8.5f} {e/abs(LINF):24.6f} {e/abs(LINF)/mu:15.2f}')


# ---------------------------------------------------------------------------
# Scope of the averaged description: which of its conclusions survive when mu
# is no longer small. Two boundaries are measured on the EXACT recursion:
#   - the phase cone, i.e. the largest |dphi| that still converges;
#   - the step-size threshold, i.e. the largest mu that still converges.
# ---------------------------------------------------------------------------

def diverges(dphi_deg, mu, scale=1.0, ncycles=600):
    """Does the exact recursion grow or decay?

    Magnitude thresholds are useless near the boundary, where the growth rate
    goes to zero: a run that is formally unstable may not have grown much in
    any affordable horizon. So the test is whether the error is larger at the
    end of the run than half-way through, which detects the sign of the rate
    rather than its size. The run length is fixed in vibration cycles, so it
    covers the same number of adaptation time constants at every gain.
    """
    phat = scale * abs(PSTAR) * np.exp(1j * (np.angle(PSTAR) + np.radians(dphi_deg)))
    linf = LSTAR / np.conj(PSTAR / phat)
    n = int(ncycles / FVIB / TS)
    lam, alpha, mid = 0j, 0.0, None
    for k in range(n):
        z = (-lam / np.conj(phat)) * np.conj(PSTAR) + LSTAR
        lam += mu * np.exp(1j * alpha) * (z * np.exp(-1j * alpha)).real
        alpha += W * TS
        if not np.isfinite(lam.real) or abs(lam) > 1e12:
            return True
        if k == n // 2:
            mid = abs(lam - linf)
    end = abs(lam - linf)
    if end < 1e-9 * abs(linf):        # converged to round-off: clearly stable
        return False
    return end > mid


def bisect(f, lo, hi, log=False, n=32):
    """Largest argument for which f is True, by bisection."""
    for _ in range(n):
        mid = np.sqrt(lo * hi) if log else 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return lo


print('\nphase cone against gain (|kappa| = 1)')
print('     gx        mu     cone [deg]')
for g in (3.0, 30.0, 300.0, 1000.0, 1500.0):
    mu = g * TS
    cone = bisect(lambda d: not diverges(d, mu), 0.0, 179.0)
    print(f'{g:8.1f} {mu:9.4f} {cone:13.2f}')

print('\nstep-size threshold against phase error')
print('   dphi   mu_thr    2cos(dphi)     ratio')
for d in (0.0, 30.0, 60.0, 85.0):
    thr = bisect(lambda m: not diverges(d, m), 1e-5, 1e3, log=True, n=40)
    pred = 2 * np.cos(np.radians(d))
    print(f'{d:7.0f} {thr:8.3f} {pred:12.3f} {thr/pred:9.2f}')
