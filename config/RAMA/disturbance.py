#!/usr/bin/env python
"""Extract the open-loop modal disturbance from a SPECULA run, for control design.

The loop obeys, exactly (verified to 0.00 nm on a 4000-frame run):

    res[k] = ol[k] - (IF @ m2c) @ comm[k-1]          (piston removed per frame)

where the '-' is the DM 'sign: -1' and the one-frame lag is the 'dm.out_layer:-1'
in prop.common_layer_list.  So with

    base = IF @ m2c                 columns = OPD [nm] per unit command
    d[k] = pinv(base) @ ol[k]       the disturbance in COMMAND coordinates

a perfect controller would output comm[k-1] = d[k], and the residual is exactly
base @ (d[k] - comm[k-1]).  That makes d the signal to design against: feed its
PSD to your filter synthesis, or its time series to a predictor.

Two projections are produced, and they are NOT the same thing:

  'command'  d = pinv(base) @ ol     least squares onto the KL modes as they are.
             Same units and same coordinates as comm.fits / modes.fits, so this
             is what the controller sees.  The KL modes are not orthogonal over
             the pupil, so these coefficients are correlated between modes.

  'nm_rms'   c = ol @ Q / npix       onto a QR-orthonormalised version of the
             same modes (ordering preserved).  Coefficients are nm rms and add
             in quadrature, so this is the one for error budgets and for
             comparing mode groups.  Do NOT feed it to the controller.

Usage, from config/RAMA:
    python disturbance.py output/20260916_143302
    python disturbance.py output/A --save dist_A.npz --psd
"""
import argparse
import glob
import os

import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, 'calib')


def modal_base(nmodes, ifunc, m2c):
    """Columns = pupil OPD [nm] produced by one unit of each modal command."""
    ifs = fits.getdata(os.path.join(CALIB, 'ifunc', ifunc + '.fits'))
    kl = fits.getdata(os.path.join(CALIB, 'm2c', m2c + '.fits'))[:, :nmodes]
    base = ifs @ kl
    return base - base.mean(0)               # piston out, to match the phase


def open_loop_phase(run_dir):
    """(nframes, npix) open-loop phase [nm], piston removed, plus the pupil mask."""
    path = os.path.join(run_dir, 'ol_ef.fits')
    if not os.path.exists(path):
        raise SystemExit(
            f'{run_dir}: no ol_ef.fits.\n'
            "Add the prop_ol branch and 'ol_ef-prop_ol.out_ngs_source_ef' to the "
            'data_store input_list (see params_rama_tracking.yml).')
    a = fits.getdata(path)                   # (nframes, 2, ny, nx): 0=amplitude, 1=phase
    pup = a[0, 0] > 0                        # the amplitude plane IS the pupil mask
    ol = a[:, 1][:, pup].astype(float)
    return ol - ol.mean(1, keepdims=True), pup


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', help='run directory containing ol_ef.fits')
    p.add_argument('--nmodes', type=int, default=83)
    p.add_argument('--skip', type=int, default=0,
                   help='frames to drop; the open loop has no transient, so 0')
    p.add_argument('--fs', type=float, default=1000.0, help='[Hz]')
    p.add_argument('--ifunc', default='rama_dm97_72p')
    p.add_argument('--m2c', default='rama_kl_90')
    p.add_argument('--save', help='write the modal time series to this .npz')
    p.add_argument('--psd', action='store_true', help='print a PSD summary per mode group')
    a = p.parse_args()

    run = sorted(glob.glob(a.run)) or [a.run]
    run = run[-1]
    ol, pup = open_loop_phase(run)
    ol = ol[a.skip:]
    base = modal_base(a.nmodes, a.ifunc, a.m2c)

    d = ol @ np.linalg.pinv(base).T          # command coordinates
    q, r = np.linalg.qr(base)
    q *= np.sign(np.diag(r))
    q *= np.sqrt(base.shape[0])
    c = ol @ q / base.shape[0]               # nm rms coordinates

    fitting = np.sqrt(((ol - d @ base.T) ** 2).mean())
    print(f'{run}')
    print(f'  {ol.shape[0]} frames, {int(pup.sum())} pupil pixels, {a.nmodes} modes')
    print(f'  open-loop phase        {np.sqrt((ol ** 2).mean()):8.1f} nm rms')
    print(f'  captured by the modes  {np.sqrt((c ** 2).mean(0).sum()):8.1f} nm rms')
    print(f'  fitting error (out)    {fitting:8.1f} nm rms')
    print(f'  command range          {np.abs(d).max():8.3f} units peak')

    groups = {'tip/tilt 0-1': slice(0, 2), 'modes 2-9': slice(2, 10),
              'modes 10-39': slice(10, 40), 'modes 40-82': slice(40, a.nmodes)}
    print('\n  disturbance per mode group [nm rms]')
    for g, sl in groups.items():
        print(f'    {g:16s} {np.sqrt((c[:, sl] ** 2).mean(0).sum()):8.1f}')

    if a.psd:
        from scipy.signal import welch
        f, pxx = welch(c, fs=a.fs, nperseg=512, axis=0)
        print('\n  cumulative disturbance above f [nm rms], orthonormal coefficients')
        print('    ' + ''.join(f'{g:>16s}' for g in groups))
        for fmin in (1, 10, 30, 60, 120, 250):
            row = ''.join(
                f'{np.sqrt(pxx[f >= fmin][:, sl].sum() * (f[1] - f[0])):16.1f}'
                for sl in groups.values())
            print(f'  >{fmin:4.0f} Hz' + row)

    if a.save:
        np.savez_compressed(a.save, command=d.astype(np.float32),
                            nm_rms=c.astype(np.float32), fs=a.fs,
                            nmodes=a.nmodes, run=run)
        print(f"\n  saved -> {a.save}  ('command' for the controller, 'nm_rms' for budgets)")


if __name__ == '__main__':
    main()
