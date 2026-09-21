#!/usr/bin/env python
"""
Generate the SPECULA calibration data for the RAMA bench.

This is the port of the OOPAO RAMA numerical twin
(OOPAO/tutorials/RAMA/{parameter_files/parameterFile_ramatwin.py,compute_ramatwin.py})
to SPECULA.  Everything that OOPAO builds on the fly at model-creation time and
that SPECULA instead expects as a calibration file is produced here:

  ifunc/<ifunc_tag>.fits        DM97 influence functions, resampled on the
                                simulation grid, with the RAMA mis-registration
                                (shift / scaling / flip) already applied
  m2c/<m2c_tag>.fits            KL modal basis computed from the IFs and the
                                atmospheric statistics (OOPAO: compute_KL_basis)
  pupilstop/<tag>_calib.fits    calibration pupil (clear disk)
  pupilstop/<tag>_sky.fits      on-sky pupil (0.34 central obstruction + spiders)
  pupils/<tag>_pupdata_ff.fits  full-frame pyramid signal: the four 60x60
                                detector quadrants (OOPAO fullFrame_sum_flux)

The influence-function interpolation reproduces OOPAO's
`OOPAO.tools.interpolateGeometricalTransformation.interpolate_cube` (vendored
below so that OOPAO does not need to be installed).

Input data: IF_97.npy, from the RAMA data set
  https://nuage.osupytheas.fr/s/YRbHrHSQA9ZSiQP
  (also shipped in ramatwin-install-*/data/IF_97.npy)

Usage:
    python make_rama_calib.py --if-file /path/to/IF_97.npy [--root-dir ../../calib]
"""

import argparse
import os

import numpy as np
import skimage.transform as sk
from astropy.io import fits

# ---------------------------------------------------------------------------
# RAMA parameters (from OOPAO parameterFile_ramatwin.py)
# ---------------------------------------------------------------------------
N_SUBAP = 36                     # PWFS subapertures across the pupil
N_PIX_PER_SUBAP = 2              # phase-screen sampling per subaperture
RESOLUTION = N_SUBAP * N_PIX_PER_SUBAP        # 72 pixels across the pupil
FRAME_SIZE = 120                 # PWFS detector, 2 * size_quadrant_ramatwin
DIAMETER = 0.6                   # [m] telescope diameter
PIXEL_PITCH = DIAMETER / RESOLUTION           # [m/pixel]

N_ACTUATOR = 11                  # actuators across the DM (97 actuators total)
DM_PITCH = 1.5e-3 * DIAMETER / 13.5e-3        # [m] actuator pitch in M1 space
DM_SIGN = -1                     # param['dm_inf_funct_factor']
DM_FLIP_LR = False
DM_FLIP_UD = True

# DM mis-registration (OOPAO MisRegistration: shifts in [m], scalings in
# fraction of the diameter, angles in [deg])
ROTATION_ANGLE = 0.0
SHIFT_X = 0.0
SHIFT_Y = -DM_PITCH / 3.0
ANAMORPHOSIS_ANGLE = 0.0
TANGENTIAL_SCALING = 0.12
RADIAL_SCALING = 0.12

CENTRAL_OBSTRUCTION = 0.34       # on-sky pupil only
SPIDER_ANGLE_OFFSET = 20.0       # [deg]
SPIDER_THICKNESS = 0.015         # [m]

R0 = 0.10                        # [m] @500nm, used for the KL basis
L0 = 30.0                        # [m]

# OOPAO photometry (OOPAO/Source.py), used to reproduce the same photon flux
# [central wavelength [m], bandwidth [m], zero point [ph/m2/s]]
OOPAO_PHOTOMETRY = {'R':  [0.640e-6, 0.150e-6, 4.01e12 / 368],
                    'J2': [1.550e-6, 0.260e-6, 1.49e12 / 368]}
MAGNITUDE = -0.04


# ---------------------------------------------------------------------------
# Vendored from OOPAO.tools.interpolateGeometricalTransformation
# ---------------------------------------------------------------------------
def _rotate_matrix(shape, angle):
    shift_y, shift_x = np.array(shape[:2]) / 2.
    tf_rotate = sk.SimilarityTransform(rotation=np.deg2rad(angle))
    tf_shift = sk.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = sk.SimilarityTransform(translation=[shift_x, shift_y])
    return tf_shift + (tf_rotate + tf_shift_inv)


def _scaling_matrix(shape, scaling):
    shift_y, shift_x = np.array(shape[:2]) / 2.
    tf_scaling = sk.SimilarityTransform(scale=scaling)
    tf_shift = sk.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = sk.SimilarityTransform(translation=[shift_x, shift_y])
    return tf_shift + (tf_scaling + tf_shift_inv)


def _anamorphosis_matrix(shape, direction, scale):
    return _rotate_matrix(shape, direction) + _scaling_matrix(shape, scale) \
           + _rotate_matrix(shape, -direction)


def interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out,
                     rotation_angle=0., shift_x=0., shift_y=0.,
                     anamorphosis_angle=0., tangential_scaling=0.,
                     radial_scaling=0., order=1):
    """Same transformation chain as OOPAO's interpolate_cube().

    cube_in is [n_maps, nx, ny]; shifts are in [m], scalings in fraction of the
    diameter, angles in [deg].
    """
    _, nx, _ = cube_in.shape
    resolution_in = int(nx)
    ratio = pixel_size_in / pixel_size_out
    extra = ratio % 1
    n_pix = resolution_in - resolution_out
    extra = extra / 2 + (np.floor(ratio) - 1) * 0.5
    n_crop = n_pix / 2
    shape = (resolution_in, resolution_in)

    down_scaling = _anamorphosis_matrix(shape, 0, [ratio, ratio])
    anam_matrix = _anamorphosis_matrix(shape, anamorphosis_angle,
                                       [1 + radial_scaling, 1 + tangential_scaling])
    rot_matrix = _rotate_matrix(shape, rotation_angle)
    # note the y/x swap: this is OOPAO's own convention
    shift_matrix = sk.SimilarityTransform(
        translation=[shift_y / pixel_size_out, shift_x / pixel_size_out])
    alignment_matrix = sk.SimilarityTransform(
        translation=[extra - n_crop, extra - n_crop])

    transform = down_scaling + anam_matrix + rot_matrix + shift_matrix + alignment_matrix

    return np.asarray([sk.warp(m, transform.inverse,
                               output_shape=[resolution_out, resolution_out],
                               order=order) for m in cube_in])


# ---------------------------------------------------------------------------
# Pupils
# ---------------------------------------------------------------------------
def apply_spiders(pupil, diameter, angles, thickness_spider):
    """Port of OOPAO Telescope.apply_spiders()."""
    resolution = pupil.shape[0]
    pup = pupil.copy()
    x = np.linspace(-diameter / 2, diameter / 2, resolution)
    X, Y = np.meshgrid(x, x)
    for angle in angles:
        angle_val = (angle + 90) % 360
        map_dist = np.abs(X * np.cos(np.deg2rad(angle_val))
                          + Y * np.sin(np.deg2rad(-angle_val)))
        if 0 <= angle_val < 90:
            map_dist[:resolution // 2, :] = thickness_spider
        if 90 <= angle_val < 180:
            map_dist[:, :resolution // 2] = thickness_spider
        if 180 <= angle_val < 270:
            map_dist[resolution // 2:, :] = thickness_spider
        if 270 <= angle_val < 360:
            map_dist[:, resolution // 2:] = thickness_spider
        pup = pup * (map_dist > thickness_spider / 2)
    return pup


# ---------------------------------------------------------------------------
def zero_point_for_specula(band, magnitude, bandw_nm):
    """e0 [J/s/m2/um] making SPECULA's n_phot() return the OOPAO photon flux.

    SPECULA integrates S0 (ph/s/m2/nm) over `bandw` nm, while OOPAO uses a
    single zero point for the whole band, so the two only agree if e0 is
    rescaled.  n_phot() is linear in e0, hence the closed form below.
    """
    lambda_m, _, zero_point = OOPAO_PHOTOMETRY[band]
    n_phot_oopao = zero_point * 10 ** (-0.4 * magnitude)   # [ph/m2/s], full band
    s0_target = n_phot_oopao / bandw_nm                    # [ph/m2/s/nm]
    h, c = 6.626e-34, 3e8
    # n_phot: S0 = lambda * (width*1e6) * e0 / (h c) * 10**(-mag/2.5), width=1e-9 m
    return s0_target * h * c / (lambda_m * 1e-3 * 10 ** (-magnitude / 2.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--if-file', required=True,
                        help='path to the RAMA IF_97.npy influence-function cube')
    parser.add_argument('--root-dir',
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', '..', 'calib'),
                        help="SPECULA calibration root_dir (default: <repo>/calib)")
    parser.add_argument('--tag', default='rama', help='prefix of the generated tags')
    parser.add_argument('--nmodes', type=int, default=90,
                        help='number of KL modes to compute (default: 90)')
    parser.add_argument('--overwrite', action='store_true', default=True)
    args = parser.parse_args()

    import specula
    specula.init(-1, precision=0)   # CPU, double precision
    from specula.data_objects.ifunc import IFunc
    from specula.data_objects.m2c import M2C
    from specula.data_objects.pupdata import PupData
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.lib.make_mask import make_mask
    from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft

    root = os.path.abspath(args.root_dir)
    for sub in ('ifunc', 'm2c', 'pupilstop', 'pupils'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    # ---------------- influence functions -------------------------------
    print(f'Reading {args.if_file}')
    cube = np.load(args.if_file)                  # [npix, npix, n_act], metres
    cube = np.moveaxis(cube, 2, 0)                # [n_act, npix, npix]
    n_act, n_px, _ = cube.shape
    print(f'  {n_act} influence functions, {n_px}x{n_px} pixels')

    if DM_FLIP_LR:
        cube = np.flip(cube, axis=2)
    if DM_FLIP_UD:
        cube = np.flip(cube, axis=1)

    # OOPAO assumes the input IF map spans exactly the telescope diameter
    pixel_size_in = DIAMETER / n_px

    cube = interpolate_cube(cube,
                            pixel_size_in=pixel_size_in,
                            pixel_size_out=PIXEL_PITCH,
                            resolution_out=RESOLUTION,
                            rotation_angle=ROTATION_ANGLE,
                            shift_x=SHIFT_X, shift_y=SHIFT_Y,
                            anamorphosis_angle=ANAMORPHOSIS_ANGLE,
                            tangential_scaling=TANGENTIAL_SCALING,
                            radial_scaling=RADIAL_SCALING)

    # SPECULA masks: the DM layer amplitude is the ifunc mask, so use the full
    # (unobstructed) disk and let the Pupilstop object define the real pupil.
    dm_mask = make_mask(RESOLUTION, obsratio=0.0, diaratio=1.0, xp=np)
    idx = np.where(dm_mask)

    # OOPAO: OPD [m] = modes [m/unit] @ coefs [DM units]
    # SPECULA: phase [nm] = ifunc [nm/unit] @ command [DM units]
    ifunc_2d = np.array([m[idx] for m in cube]) * 1e9      # [n_act, n_valid_pix]

    ifunc = IFunc(ifunc=ifunc_2d, mask=dm_mask)
    ifunc_tag = f'{args.tag}_dm97_{RESOLUTION}p'
    ifunc_file = os.path.join(root, 'ifunc', ifunc_tag + '.fits')
    ifunc.save(ifunc_file, overwrite=args.overwrite)
    print(f'  wrote {ifunc_file}   (ifunc {ifunc_2d.shape}, '
          f'peak {np.abs(ifunc_2d).max():.1f} nm per DM unit)')

    # ---------------- KL modal basis -------------------------------------
    print(f'Computing the KL basis ({args.nmodes} modes) from the IFs, '
          f'r0={R0} m, L0={L0} m ...')
    # The piston-removal term inside make_modal_base_from_ifs_fft() assumes
    # influence functions of order unity, so feed it a peak-normalised copy;
    # the physical scaling is restored by the 1 nm rms normalisation below.
    ifunc_norm = ifunc_2d / np.abs(ifunc_2d).max()
    _, m2c_matrix, _ = make_modal_base_from_ifs_fft(
        pupil_mask=dm_mask, diameter=DIAMETER, influence_functions=ifunc_norm,
        r0=R0, L0=L0, zern_modes=0, oversampling=2, xp=np, dtype=np.float64)
    m2c_matrix = m2c_matrix[:, :args.nmodes]

    # Normalise every KL mode to 1 nm rms of wavefront over the pupil, so that
    # the modal commands handled by Modalrec/Integrator are in nm rms (SPECULA
    # convention) instead of raw DM units.
    mode_rms = np.std(m2c_matrix.T @ ifunc_2d, axis=1)
    m2c_matrix = m2c_matrix / mode_rms[np.newaxis, :]
    print(f'  KL modes normalised to 1 nm rms '
          f'(DM units per nm rms: {1/mode_rms.max():.3g} .. {1/mode_rms.min():.3g})')
    modes = m2c_matrix.T @ ifunc_2d
    modes /= np.linalg.norm(modes, axis=1)[:, np.newaxis]
    cross = np.abs(modes @ modes.T)[np.triu_indices(args.nmodes, 1)]
    print(f'  basis orthogonality: max |cross-correlation| {cross.max():.3g}, '
          f'mean {cross.mean():.3g}')

    m2c_tag = f'{args.tag}_kl_{args.nmodes}'
    m2c_file = os.path.join(root, 'm2c', m2c_tag + '.fits')
    M2C(m2c_matrix).save(m2c_file, overwrite=args.overwrite)
    print(f'  wrote {m2c_file}   (m2c {m2c_matrix.shape})')

    # ---------------- pupils ---------------------------------------------
    simul_params = SimulParams(pixel_pupil=RESOLUTION, pixel_pitch=PIXEL_PITCH)

    calib_mask = make_mask(RESOLUTION, obsratio=0.0, diaratio=1.0, xp=np)
    calib_tag = f'{args.tag}_pupil_calib_{RESOLUTION}p'
    calib_file = os.path.join(root, 'pupilstop', calib_tag + '.fits')
    Pupilstop(simul_params, input_mask=calib_mask).save(calib_file, overwrite=args.overwrite)
    print(f'  wrote {calib_file}   (clear disk, {int(calib_mask.sum())} px)')

    sky_mask = make_mask(RESOLUTION, obsratio=CENTRAL_OBSTRUCTION, diaratio=1.0, xp=np)
    sky_mask = apply_spiders(sky_mask, DIAMETER,
                             [SPIDER_ANGLE_OFFSET + 90 * i for i in range(4)],
                             SPIDER_THICKNESS)
    sky_tag = f'{args.tag}_pupil_sky_{RESOLUTION}p'
    sky_file = os.path.join(root, 'pupilstop', sky_tag + '.fits')
    Pupilstop(simul_params, input_mask=sky_mask).save(sky_file, overwrite=args.overwrite)
    print(f'  wrote {sky_file}   (obs {CENTRAL_OBSTRUCTION} + 4 spiders, '
          f'{int(sky_mask.sum())} px)')

    # ---------------- full-frame pyramid signal -----------------------------
    # OOPAO 'fullFrame_sum_flux' with lightRatio = 0 uses every pixel of the
    # 120x120 frame, normalised by the total flux.  Declaring the four 60x60
    # detector quadrants as the "pupils" makes PyrSlopec (slopes_from_intensity)
    # do exactly that.  With the thresholded pupils of calib_rama_pupdata.yml
    # (4408 px) the unmodulated closed-loop gain comes out 1.2-1.6x higher than
    # in OOPAO, because the light the pyramid diffracts outside the geometric
    # pupils is thrown away.
    # Quadrant order A, B, C, D = top-right, top-left, bottom-left, bottom-right,
    # the same order PyrPupdataCalibrator finds for this pyramid.
    q = FRAME_SIZE // 2
    frame_idx = np.arange(FRAME_SIZE * FRAME_SIZE).reshape(FRAME_SIZE, FRAME_SIZE)
    quadrants = [frame_idx[:q, q:], frame_idx[:q, :q], frame_idx[q:, :q], frame_idx[q:, q:]]
    ind_pup = np.stack([quad.ravel() for quad in quadrants], axis=1)
    pupdata = PupData(ind_pup=ind_pup,
                      radius=[N_SUBAP / 2] * 4,
                      cx=[q + q / 2, q / 2, q / 2, q + q / 2],
                      cy=[q / 2, q / 2, q + q / 2, q + q / 2],
                      framesize=[FRAME_SIZE, FRAME_SIZE])
    pupdata_tag = f'{args.tag}_pupdata_ff'
    pupdata_file = os.path.join(root, 'pupils', pupdata_tag + '.fits')
    pupdata.save(pupdata_file, overwrite=args.overwrite)
    print(f'  wrote {pupdata_file}   (4 x {ind_pup.shape[0]} px, full frame)')

    # ---------------- photometry ------------------------------------------
    print('\nSource zero points reproducing the OOPAO flux at magnitude '
          f'{MAGNITUDE} (put these in the yml):')
    for band, bandw_nm in (('R', 150.), ('J2', 260.)):
        e0 = zero_point_for_specula(band, MAGNITUDE, bandw_nm)
        lam, _, zp = OOPAO_PHOTOMETRY[band]
        print(f'  band {band:3s} lambda={lam*1e9:6.0f} nm  bandw={bandw_nm:5.0f} nm  '
              f'zero_point={e0:.6g}   '
              f'(OOPAO flux {zp*10**(-0.4*MAGNITUDE):.4g} ph/m2/s)')

    print('\nTags to use in the yml files:')
    print(f'  ifunc_object:     {ifunc_tag!r}')
    print(f'  m2c_object:       {m2c_tag!r}')
    print(f'  pupilstop_object: {calib_tag!r} / {sky_tag!r}')
    print(f'  pupdata_object:   {pupdata_tag!r}')
    print(f'  DM sign in the yml: {DM_SIGN}')


if __name__ == '__main__':
    main()
