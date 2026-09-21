#!/usr/bin/env python
"""Per-layer effective wind for a tracked, fast-moving object.

While the telescope tracks at angular rate omega, the line of sight sweeps
through the atmosphere, so the pierce point at layer height h moves at
h*omega/cos(zenith) on top of the real wind.  The two add as vectors:

    v_eff,i = | v_wind,i  +  h_i * omega / cos(zenith) * e_slew |

The ground layer is untouched; the high layers dominate completely.
Feed the printed 'constant:' lines to wind_speed / wind_direction.

  python slew_winds.py --rate 0.61
  python slew_winds.py --rate 0.20 --zenith 60
"""
import argparse
import numpy as np

# RAMA atmosphere, as in params_rama.yml
HEIGHTS = np.array([0., 1000., 5000., 10000., 12000.])
CN2     = np.array([0.45, 0.10, 0.10, 0.25, 0.10])
WIND    = np.array([5., 4., 8., 10., 2.])
WDIR    = np.array([0., 72., 144., 216., 288.])
PITCH, FS, D = 0.6 / 72, 1000., 0.6

p = argparse.ArgumentParser()
p.add_argument('--rate', type=float, default=0.61, help='tracking rate [deg/s]')
p.add_argument('--zenith', type=float, default=0.0, help='zenith angle [deg]')
p.add_argument('--slew-dir', type=float, default=90.0,
               help='direction the LOS sweeps, same convention as wind_direction [deg]')
p.add_argument('--r0', type=float, default=0.10, help='r0 at 500 nm [m]')
a = p.parse_args()

r = np.deg2rad(WDIR)
wx, wy = WIND * np.sin(r), WIND * np.cos(r)

# slew contribution, along --slew-dir
om = np.deg2rad(a.rate) / np.cos(np.deg2rad(a.zenith))
s = np.deg2rad(a.slew_dir)
sx, sy = wx + HEIGHTS * om * np.sin(s), wy + HEIGHTS * om * np.cos(s)

v = np.hypot(sx, sy)
d = np.rad2deg(np.arctan2(sx, sy)) % 360.

v_bar = (np.sum(CN2 * v ** (5 / 3))) ** (3 / 5)
v_rms = np.sqrt(np.sum(CN2 * v ** 2))
fmt = lambda x: '[' + ', '.join('%.2f' % q for q in x) + ']'

print(f'tracking rate {a.rate} deg/s, zenith {a.zenith} deg, r0 {a.r0} m\n')
print('wind_speed:\n  constant:  ' + fmt(v) + '   # [m/s]')
print('wind_direction:\n  constant:  ' + fmt(d) + '   # [degrees]\n')
print('  v_bar (Cn2-weighted 5/3) : %7.1f m/s' % v_bar)
print('  f_Greenwood              : %7.1f Hz' % (0.426 * v_bar / a.r0))
print('  tau0                     : %7.3f ms' % (0.314 * a.r0 / v_bar * 1e3))
print('  TT knee ~0.3 v_rms/D     : %7.1f Hz' % (0.3 * v_rms / D))
print('\n  shift/frame [px] : ' + fmt(v / PITCH / FS))
print('  fractional part  : ' + fmt(v / PITCH / FS % 1))
print('  sub-pixel interpolation line, aliased to 1 kHz [Hz]:')
f = (v / PITCH / FS % 1) * FS
print('                     ' + fmt(np.minimum(f % FS, FS - f % FS)))
