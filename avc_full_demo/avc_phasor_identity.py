"""Graphical proof of the projection identity used throughout the report.

(a) For any phasor A,  Re(A)cos(a) + Im(A)sin(a) is the orthogonal projection
    of A on the direction a; equivalently Re(A e^{-ia}).
(b) The same picture rotated by -a: the projection becomes a real part.
(c) Applied to W'x, whose phasor is Zhat = theta*conj(Phat) + Lambda_hat.
    The control law makes the two contributions exactly antiparallel, so
    Zhat = Lambda_hat*eps/D and W'x vanishes as eps -> 0.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = 2.6 * np.exp(1j * np.radians(63.0))     # a generic phasor
ANG = np.radians(24.0)                      # the instantaneous phase alpha

def arrow(ax, z0, z1, color, lw=2.0, ls='-', **kw):
    ax.annotate('', xy=(z1.real, z1.imag), xytext=(z0.real, z0.imag),
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color,
                                linestyle=ls, shrinkA=0, shrinkB=0), **kw)

def frame(ax, lim=3.2):
    ax.axhline(0, color='0.55', lw=0.9, zorder=0)
    ax.axvline(0, color='0.55', lw=0.9, zorder=0)
    ax.set_xlim(-0.9, lim); ax.set_ylim(-0.9, lim)
    ax.set_aspect('equal'); ax.axis('off')

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.4))

# ---------------------------------------------------------------- panel (a)
a = ax[0]; frame(a)
u = np.exp(1j * ANG)                       # unit direction of the phase
proj = (A.real * np.cos(ANG) + A.imag * np.sin(ANG))   # = Re(A e^{-i a})
foot = proj * u

a.plot([0, 3.05 * u.real], [0, 3.05 * u.imag], color='tab:green',
       lw=1.2, ls='--', zorder=1)
a.plot([A.real, foot.real], [A.imag, foot.imag], color='0.45', lw=1.0, ls=':')
arrow(a, 0, A, 'tab:blue', 2.4)
arrow(a, 0, foot, 'tab:green', 3.0)
a.plot([A.real, A.real], [0, A.imag], color='tab:blue', lw=0.9, ls=':')
a.plot([0, A.real], [A.imag, A.imag], color='tab:blue', lw=0.9, ls=':')

th = np.linspace(0, ANG, 40)
a.plot(0.75 * np.cos(th), 0.75 * np.sin(th), color='tab:green', lw=1.0)
a.text(0.88 * np.cos(ANG / 2), 0.88 * np.sin(ANG / 2), r'$\alpha$',
       color='tab:green', fontsize=12)
a.text(A.real + .06, A.imag + .12, r'$A$', color='tab:blue', fontsize=13)
a.text(A.real + .06, -.28, r'$\mathrm{Re}\,A$', color='tab:blue', fontsize=10)
a.text(-.85, A.imag, r'$\mathrm{Im}\,A$', color='tab:blue', fontsize=10)
a.text(foot.real + .10, foot.imag - .38,
       r'$\mathrm{Re}(A)\cos\alpha+\mathrm{Im}(A)\sin\alpha$',
       color='tab:green', fontsize=10)
a.text(2.55 * u.real, 2.55 * u.imag + .22, r'$e^{i\alpha}$',
       color='tab:green', fontsize=11)
a.set_title(r'(a) the sum is a projection on the direction $\alpha$',
            fontsize=10.5)

# ---------------------------------------------------------------- panel (b)
b = ax[1]; frame(b)
Ar = A * np.exp(-1j * ANG)
b.plot([Ar.real, Ar.real], [0, Ar.imag], color='0.45', lw=1.0, ls=':')
arrow(b, 0, Ar, 'tab:blue', 2.4)
arrow(b, 0, complex(Ar.real, 0), 'tab:green', 3.0)
th = np.linspace(np.angle(A), np.angle(Ar), 60)
b.plot(2.0 * np.cos(th), 2.0 * np.sin(th), color='0.55', lw=1.0, ls='--')
arrow(b, 2.0 * np.exp(1j * (np.angle(Ar) + 0.06)),
      2.0 * np.exp(1j * np.angle(Ar)), '0.55', 1.4)
b.text(1.55, 2.30, r'rotate by $-\alpha$', color='0.35', fontsize=10)
b.text(Ar.real + .08, Ar.imag + .12, r'$A\,e^{-i\alpha}$',
       color='tab:blue', fontsize=13)
b.text(Ar.real - .55, -.40, r'$\mathrm{Re}\left(A e^{-i\alpha}\right)$',
       color='tab:green', fontsize=11)
b.set_title(r'(b) $\ldots$ i.e. a real part, after rotating back',
            fontsize=10.5)

# ---------------------------------------------------------------- panel (c)
c = ax[2]
c.axhline(0, color='0.55', lw=0.9, zorder=0)
c.axvline(0, color='0.55', lw=0.9, zorder=0)
c.set_xlim(-3.4, 3.4); c.set_ylim(-3.0, 3.1)
c.set_aspect('equal'); c.axis('off')

Lh = 1.9 * np.exp(1j * np.radians(70.0))          # Lambda_hat
c.plot([0, Lh.real], [0, Lh.imag], color='0.8', lw=6, solid_capstyle='round')
arrow(c, 0, Lh, 'tab:purple', 2.4)
arrow(c, 0, -Lh * 0.94, 'tab:orange', 2.4)        # theta*conj(Phat), eps>0
c.text(Lh.real + .10, Lh.imag, r'$\hat\Lambda$', color='tab:purple', fontsize=13)
c.text(-3.30, -1.30,
       r'$\vartheta\,\overline{\hat P}=-\dfrac{|\hat P|^{2}}{D}\,\hat\Lambda$',
       color='tab:orange', fontsize=11)
arrow(c, 0, Lh * 0.06, 'tab:red', 3.2)
c.text(.34, .30, r'$\hat Z=\hat\Lambda\,\epsilon/D\;\to\;0$',
       color='tab:red', fontsize=11)
c.text(-3.30, -2.75,
       r'$\mathbf{W}^\top\mathbf{x}=\mathrm{Re}\!\left(\hat Z e^{-i\alpha}\right)$'
       r'$\;\Rightarrow\;\mathbf{W}^\top\mathbf{x}\equiv0,\quad e=-y$',
       fontsize=11)
c.set_title(r'(c) the prediction phasor cancels itself', fontsize=10.5)

fig.tight_layout()
fig.savefig('avc_phasor_identity.pdf')
print('written')
