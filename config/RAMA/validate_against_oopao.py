#!/usr/bin/env python
"""
Cross-validate the SPECULA RAMA port against the OOPAO RAMA twin.

Both codes are driven with the *same* DM command vector (the KL basis written by
make_rama_calib.py) and the resulting 120x120 pyramid frames are compared
directly in image space. Working in image space avoids any dependence on the
valid-pixel selection or on the signal ordering, which differ between the two
codes.

Checks performed:
  1. pupil masks
  2. DM optical path difference for a given command
  3. flat-wavefront pyramid frame
  4. modal response  dI = I(+a) - I(-a)  for a set of KL modes
  5. stroke dependence of the sensitivity ratio

Requires OOPAO importable (jsonpickle, numba, numexpr, joblib) and the RAMA
data set (IF_97.npy). Run make_rama_calib.py and the two calibration steps first.

Usage:
    python validate_against_oopao.py --if-file /path/to/IF_97.npy \
                                     [--oopao-root /path/to/OOPAO]
"""

import argparse
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np

# response sign: SPECULA's pyramid signal is the negative of OOPAO's for the
# same wavefront, because the two codes use opposite phase sign conventions in
# the electric field. See README.md, "Validation against OOPAO".
SPECULA_VS_OOPAO_SIGN = -1.0

RES, PITCH, DM_SIGN = 72, 0.6 / 72, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--if-file', required=True)
    parser.add_argument('--oopao-root', default='/Users/marcofalotico/Documents/Repos/OOPAO')
    parser.add_argument('--calib-dir',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'calib'))
    parser.add_argument('--amp', type=float, default=10.0, help='[nm rms] modal stroke')
    args = parser.parse_args()

    rama_dir = os.path.join(args.oopao_root, 'tutorials', 'RAMA')
    sys.path.insert(0, args.oopao_root)
    sys.path.insert(0, rama_dir)

    # ------------------------------------------------------------ OOPAO model
    from parameter_files.parameterFile_ramatwin import initializeParameterFile
    from compute_ramatwin import compute_rama_model

    param = initializeParameterFile()
    param['dm_inf_funct_location'] = os.path.abspath(args.if_file)
    param['path_data'] = os.path.dirname(os.path.abspath(args.if_file)) + os.sep

    tel, ngs, src, dm, wfs, atm = compute_rama_model(
        param=param, loc=param['path_data'], source=True, IFreal=False)
    wfs.cam.photonNoise = False
    wfs.modulation = 0

    # ---------------------------------------------------------- SPECULA model
    import specula
    specula.init(-1, precision=0)
    from specula.data_objects.electric_field import ElectricField
    from specula.data_objects.ifunc import IFunc
    from specula.data_objects.m2c import M2C
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.processing_objects.modulated_pyramid import ModulatedPyramid

    calib = os.path.abspath(args.calib_dir)
    ifunc = IFunc.restore(os.path.join(calib, 'ifunc', 'rama_dm97_72p.fits'))
    m2c = M2C.restore(os.path.join(calib, 'm2c', 'rama_kl_90.fits'))
    pstop = Pupilstop.restore(os.path.join(calib, 'pupilstop', 'rama_pupil_calib_72p.fits'))

    simul_params = SimulParams(pixel_pupil=RES, pixel_pitch=PITCH)
    pyr = ModulatedPyramid(simul_params=simul_params, wavelengthInNm=640, fov=16.0,
                           pup_diam=36, pup_dist=52, output_resolution=120, mod_amp=0.0)

    IF_nm = np.asarray(ifunc.influence_function)
    mask = np.asarray(ifunc.mask_inf_func)
    idx = np.where(mask)
    M2C_mat = np.asarray(m2c.m2c)

    def oopao_frame(command):
        dm.coefs = command
        ngs**tel*dm*wfs                      # noqa: E225  (OOPAO operator syntax)
        return np.array(wfs.cam.frame, dtype=float)

    def specula_phase(command):
        phase = np.zeros((RES, RES))
        phase[idx] = DM_SIGN * (command @ IF_nm)
        return phase

    def specula_frame(command):
        ef = ElectricField(RES, RES, PITCH, S0=1.0)
        ef.A = pstop.A
        ef.phaseInNm = specula_phase(command)
        ef.generation_time = 1
        pyr.inputs['in_ef'].set(ef)
        pyr.setup()
        pyr.check_ready(1)
        pyr.trigger()
        pyr.post_trigger()
        return np.array(specula.cpuArray(pyr.outputs['out_i'].i), dtype=float)

    def norm(f):
        return f / f.sum()

    # ----------------------------------------------------------- 1. pupils
    print('\n=== 1. Pupil mask ===')
    pup_o = np.array(tel.pupil, dtype=float)
    pup_s = np.array(specula.cpuArray(pstop.A), dtype=float)
    print(f'  OOPAO {int(pup_o.sum())} px, SPECULA {int(pup_s.sum())} px, '
          f'differing pixels {int(np.abs(pup_o - pup_s).sum())}')

    # -------------------------------------------------------------- 2. DM OPD
    print('\n=== 2. DM optical path difference (mode 10, 10 nm rms) ===')
    pup = pup_o.astype(bool)
    cmd = M2C_mat[:, 10] * 10.0
    dm.coefs = cmd
    opd_o = np.array(dm.OPD, dtype=float) * 1e9
    opd_s = specula_phase(cmd)
    print(f'  rms inside pupil: OOPAO {opd_o[pup].std():.6f} nm, '
          f'SPECULA {opd_s[pup].std():.6f} nm')
    print(f'  max |difference| inside pupil: {np.abs(opd_o - opd_s)[pup].max():.3e} nm')
    print(f'  correlation: {np.corrcoef(opd_o[pup], opd_s[pup])[0, 1]:+.8f}')

    # ------------------------------------------------------------- 3. flat
    print('\n=== 3. Flat wavefront ===')
    fo, fs = norm(oopao_frame(np.zeros(97))), norm(specula_frame(np.zeros(97)))
    print(f'  correlation {np.corrcoef(fo.ravel(), fs.ravel())[0, 1]:.6f}, '
          f'relative rms difference '
          f'{np.sqrt(np.mean((fo - fs)**2)) / np.sqrt(np.mean(fo**2)):.4f}')

    # ------------------------------------------------------- 4. modal response
    print(f'\n=== 4. Modal response, dI = I(+a) - I(-a), a = {args.amp} nm rms ===')
    print(f'  (SPECULA response multiplied by {SPECULA_VS_OOPAO_SIGN:+.0f})')
    print('  %-6s %-12s %-12s %-12s' % ('mode', 'corr', 'ampl ratio', 'rel rms diff'))
    corrs, ratios, rels = [], [], []
    for k in range(0, 83, 4):
        c = M2C_mat[:, k] * args.amp
        do = norm(oopao_frame(c)) - norm(oopao_frame(-c))
        ds = SPECULA_VS_OOPAO_SIGN * (norm(specula_frame(c)) - norm(specula_frame(-c)))
        corrs.append(np.corrcoef(do.ravel(), ds.ravel())[0, 1])
        ratios.append(np.std(ds) / np.std(do))
        rels.append(np.sqrt(np.mean((do - ds)**2)) / np.sqrt(np.mean(do**2)))
        if k % 12 == 0:
            print('  %-6d %-12.6f %-12.4f %-12.4f' % (k, corrs[-1], ratios[-1], rels[-1]))
    print(f'  --- over {len(corrs)} modes ---')
    print(f'  correlation       min {min(corrs):.6f}  mean {np.mean(corrs):.6f}')
    print(f'  sensitivity ratio {np.mean(ratios):.4f} +- {np.std(ratios):.4f}')
    print(f'  rel rms diff      {np.mean(rels):.4f} +- {np.std(rels):.4f}')

    # ---------------------------------------------------------- 5. linearity
    print('\n=== 5. Stroke dependence (mode 10) ===')
    print('  %-12s %-14s %-12s' % ('stroke [nm]', 'corr', 'ampl ratio'))
    for a in (1.0, 5.0, 20.0, 50.0):
        c = M2C_mat[:, 10] * a
        do = norm(oopao_frame(c)) - norm(oopao_frame(-c))
        ds = SPECULA_VS_OOPAO_SIGN * (norm(specula_frame(c)) - norm(specula_frame(-c)))
        print('  %-12.1f %-14.6f %-12.4f'
              % (a, np.corrcoef(do.ravel(), ds.ravel())[0, 1], np.std(ds) / np.std(do)))


if __name__ == '__main__':
    main()
