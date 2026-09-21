#!/usr/bin/env python
"""Compare SPECULA RAMA runs modally: error budget and rejection transfer function.

Complements output_analysis.py, which plots a single run.  This one projects the
residual (and, if present, the open-loop) phase onto the KL basis and compares
any number of runs side by side.

Needs, in each run directory:
    res_ef.fits    residual field   (from 'res_ef-prop.out_ngs_source_ef')
    ol_ef.fits     open-loop field  (from 'ol_ef-prop_ol.out_ngs_source_ef'),
                   optional -- without it the |RTF| table is skipped.
                   params_rama_tracking.yml has the prop_ol branch that makes it;
                   add the same to params_rama.yml if you want it there too.

Coefficients are in nm rms over the pupil: the KL maps are orthonormalised with
a QR (which preserves the mode ordering) and scaled by sqrt(npix).

Usage, from config/RAMA:
    python compare_runs.py output/20260916_143302
    python compare_runs.py sidereal=output/A leo=output/B --skip 500
    python compare_runs.py 'output/*'            # every run, oldest first
"""
import argparse
import glob
import os

import numpy as np
from astropy.io import fits
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(HERE, 'calib')

GROUPS = {'tip/tilt 0-1': slice(0, 2), 'modes 2-9': slice(2, 10),
          'modes 10-39': slice(10, 40), 'modes 40-82': slice(40, 83)}
BANDS = [(2, 10), (10, 30), (30, 60), (60, 120), (120, 250), (250, 500)]


def basis(nmodes, pupilstop, ifunc, m2c, dm_sign):
    """Orthonormal KL maps inside the pupil, one column per mode, nm rms."""
    pup = fits.getdata(os.path.join(CALIB, 'pupilstop', pupilstop + '.fits'))
    pup = (pup[0] if pup.ndim == 3 else pup) > 0
    ifs = fits.getdata(os.path.join(CALIB, 'ifunc', ifunc + '.fits'))
    kl = fits.getdata(os.path.join(CALIB, 'm2c', m2c + '.fits'))[:, :nmodes]

    maps = dm_sign * (ifs @ kl)              # (npix_in_pupil, nmodes)
    maps = maps - maps.mean(0)               # piston out
    q, r = np.linalg.qr(maps)
    q *= np.sign(np.diag(r))                 # fix QR sign convention
    q *= np.sqrt(maps.shape[0])              # coefficient <-> nm rms
    return pup, q


def phase(run_dir, name, pup, skip):
    path = os.path.join(run_dir, name + '.fits')
    if not os.path.exists(path):
        return None
    ef = fits.getdata(path)[skip:, 1][:, pup].astype(float)
    return ef - ef.mean(1, keepdims=True)    # piston out, frame by frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='run dirs, optionally label=dir; globs expanded')
    p.add_argument('--skip', type=int, default=500, help='frames to drop (loop transient)')
    p.add_argument('--nmodes', type=int, default=83)
    p.add_argument('--fs', type=float, default=1000.0, help='[Hz]')
    p.add_argument('--wavelength', type=float, default=1550.0, help='[nm] for the Strehl')
    p.add_argument('--pupilstop', default='rama_pupil_calib_72p')
    p.add_argument('--ifunc', default='rama_dm97_72p')
    p.add_argument('--m2c', default='rama_kl_90')
    p.add_argument('--dm-sign', type=int, default=-1, help="must match dm 'sign' in the yml")
    a = p.parse_args()

    runs = []
    for item in a.runs:
        label, _, spec = item.rpartition('=')
        for d in sorted(glob.glob(spec)) or [spec]:
            if not os.path.isdir(d):
                raise SystemExit(f'not a directory: {d}')
            runs.append((label or os.path.basename(d.rstrip('/')), d))

    pup, q = basis(a.nmodes, a.pupilstop, a.ifunc, a.m2c, a.dm_sign)
    npix = int(pup.sum())
    res = {}

    print(f'{len(runs)} run(s), {a.nmodes} modes, {a.skip} frames skipped, '
          f'{npix} pupil pixels\n')
    hdr = f"{'run':22s} {'OL':>8s} {'RES':>8s} {'TT':>8s} {'HO':>8s} {'out':>8s} {'SR':>7s}"
    print(hdr + f'   (nm rms; SR at {a.wavelength:.0f} nm)')
    print('-' * len(hdr))

    for label, d in runs:
        rs = phase(d, 'res_ef', pup, a.skip)
        if rs is None:
            raise SystemExit(f'{d}: no res_ef.fits')
        ol = phase(d, 'ol_ef', pup, a.skip)
        c = rs @ q / npix
        res[label] = (c, (ol @ q / npix) if ol is not None else None)
        sr = np.exp(-((2 * np.pi / a.wavelength) ** 2) * (rs ** 2).mean(1)).mean()
        print(f'{label:22s} '
              f'{np.sqrt((ol ** 2).mean()) if ol is not None else np.nan:8.1f} '
              f'{np.sqrt((rs ** 2).mean()):8.1f} '
              f'{np.sqrt((c[:, :2] ** 2).mean(0).sum()):8.1f} '
              f'{np.sqrt((c[:, 2:] ** 2).mean(0).sum()):8.1f} '
              f'{np.sqrt(((rs - c @ q.T) ** 2).mean()):8.1f} '
              f'{sr:7.3f}')

    print('\nResidual per mode group [nm rms]')
    print(f"{'group':16s}" + ''.join(f'{l:>14s}' for l, _ in runs))
    for g, sl in GROUPS.items():
        print(f'{g:16s}' + ''.join(
            f'{np.sqrt((res[l][0][:, sl] ** 2).mean(0).sum()):14.1f}' for l, _ in runs))

    if all(res[l][1] is not None for l, _ in runs):
        print('\nRejection |RTF|^2 = PSD(res)/PSD(ol) [dB], all 83 modes pooled')
        print(f"{'run':22s}" + ''.join(f'{f"{lo}-{hi} Hz":>12s}' for lo, hi in BANDS))
        for label, _ in runs:
            c, co = res[label]
            f, pr = welch(c, fs=a.fs, nperseg=512, axis=0)
            _, po = welch(co, fs=a.fs, nperseg=512, axis=0)
            pr, po = pr.sum(1), po.sum(1)
            print(f'{label:22s}' + ''.join(
                f'{10 * np.log10(pr[m].mean() / po[m].mean()):12.1f}'
                for m in ((f >= lo) & (f < hi) for lo, hi in BANDS)))
        print('\npositive = the loop amplifies the disturbance in that band')
    else:
        print('\n(no ol_ef.fits in some runs -> |RTF| table skipped)')


if __name__ == '__main__':
    main()
