# RAMA in SPECULA

Port of the OOPAO RAMA numerical twin (`OOPAO/tutorials/RAMA/`) to SPECULA.

Source files on the OOPAO side:

| OOPAO | role |
|---|---|
| `parameter_files/parameterFile_ramatwin.py` | the parameter dictionary (this is *the* RAMA configuration) |
| `compute_ramatwin.py` | builds tel / ngs / src / dm / wfs / atm from that dictionary |
| `Rama.py`, `ramatwin.py` | wrapper class and closed-loop script |

`parameterFile_RAMA.py` in the same folder is **not** the RAMA file: it is a
leftover GHOST/ekarus parameter file (1.82 m telescope, 42 subaps, `name = 'ekarus'`).
Everything below comes from `parameter_files/parameterFile_ramatwin.py`.

## Files here

| file | what it does |
|---|---|
| `make_rama_calib.py` | converts the RAMA bench data into SPECULA calibration objects (DM influence functions, KL basis, pupil masks) |
| `params_rama.yml` | the simulation itself |
| `calib_rama_pupdata.yml` | override: finds the 4 pyramid pupils on the detector |
| `calib_rama_rec.yml` | override: KL interaction matrix + reconstructor |
| `validate_against_oopao.py` | drives both codes with the same DM commands and compares the pyramid frames |

## How to run

`make_rama_calib.py` needs `IF_97.npy` from the RAMA data set
(<https://nuage.osupytheas.fr/s/YRbHrHSQA9ZSiQP>, also in `ramatwin-install-*/data/`):

```bash
python config/RAMA/make_rama_calib.py --if-file /path/to/IF_97.npy
specula config/RAMA/params_rama.yml config/RAMA/calib_rama_pupdata.yml
specula config/RAMA/params_rama.yml config/RAMA/calib_rama_rec.yml
specula config/RAMA/params_rama.yml
```

The last command closes the loop for 500 iterations and reaches SR ~0.92 at
1550 nm for the nominal r0 = 0.10 m.

## Validation against OOPAO

`validate_against_oopao.py` builds the OOPAO RAMA twin and the SPECULA model
side by side, pushes the *same* KL command vector into both, and compares the
120x120 pyramid frames in image space (image space, so the comparison does not
depend on the valid-pixel selection or the signal ordering, which differ).

```bash
python config/RAMA/validate_against_oopao.py --if-file /path/to/IF_97.npy
```

Results (21 KL modes, 10 nm rms stroke):

| check | result |
|---|---|
| pupil mask | 4060 px both, **0 differing pixels** |
| DM optical path difference | max difference **2.8e-14 nm**, correlation **+1.00000000** |
| flat-wavefront frame | correlation 0.9988, relative rms difference 4.3 % |
| modal response `dI = I(+a) - I(-a)` | correlation **0.9988 .. 0.9992** (mean 0.99906) |
| sensitivity (amplitude ratio) | **1.0037 +- 0.0055** |
| residual frame difference | 4.4 % +- 0.25 % rms |
| stroke dependence (1 to 50 nm) | ratio 1.0002 to 1.0023, no trend |

So: identical pupil, identical wavefront for a given command (to machine
precision), and a pyramid response that matches OOPAO's in shape to ~0.1 % of
correlation and in amplitude to ~0.5 %.

### The response sign is inverted

For the *same* wavefront, SPECULA's pyramid signal is the **negative** of
OOPAO's — consistently, across every mode tested (correlation -0.999034 +-
0.000129 before the sign is applied). The two codes use opposite phase sign
conventions in the electric field. Consequences:

* **Closed loop: none.** SPECULA calibrates its own interaction matrix, so the
  sign cancels and the loop closes normally. `dm.sign: -1` in `params_rama.yml`
  is the setting that makes a given command produce the same wavefront as in
  OOPAO, which is what matters for command fidelity against the bench.
* **Importing bench or OOPAO signals: flip the sign.** If you ever feed the
  measured `IMFull.fits`, an OOPAO-computed interaction matrix, or bench slope
  vectors into SPECULA, they have to be negated first.

### Where the residual 4.4 % comes from

It is the same size on the flat wavefront (4.3 %) as on the modal responses, so
it is not a modal or a scaling error: it is the pupil-edge diffraction, i.e. the
different FFT/padding implementations of the two pyramid models. The number of
illuminated pixels above 1 % of the peak differs accordingly (OOPAO 9136,
SPECULA 8760). The per-quadrant flux split is 0.2500 / 0.2500 / 0.2500 / 0.2500
in both codes.

### What is still not validated

The comparison is **static**. Not checked: the atmosphere (the two codes
generate different phase screens, so a like-for-like test needs the same screen
injected into both), the temporal behaviour of the loop, the photon noise
statistics, and the absolute Strehl. The SR ~0.92 quoted above is a SPECULA
number with no OOPAO counterpart.

## Parameter mapping

| OOPAO (`param[...]`) | value | SPECULA |
|---|---|---|
| `resolution` (= `nSubaperture` * `nPixelPerSubap`) | 72 | `main.pixel_pupil` |
| `diameter` / `resolution` | 0.008333 m | `main.pixel_pitch` |
| `samplingTime` | 1 ms | `main.time_step` |
| `r0`, `L0` | 0.10 m, 30 m | `seeing.constant` = 1.0065", `atmo.L0` |
| `fractionnalR0`, `altitude`, `windSpeed`, `windDirection` | 5 layers | `atmo.Cn2 / heights`, `wind_speed`, `wind_direction` |
| `magnitude`, `opticalBandCalib` = R | -0.04, 640 nm | `ngs_source` (+ `zero_point`, see below) |
| `magnitude`, `opticalBand` = J2 | -0.04, 1550 nm | `science_source`, `psf.wavelengthInNm` |
| `nSubaperture` | 36 | `pyramid.pup_diam` |
| `nSubaperture` + `n_pix_separation` | 36 + 16 | `pyramid.pup_dist` = 52 |
| `size_quadrant_ramatwin` * 2 | 120 | `pyramid.output_resolution`, `detector.size` |
| `modulation` | 0 | `pyramid.mod_amp` |
| `postProcessing` = `fullFrame_sum_flux` | — | `slopec.slopes_from_intensity: True` |
| `nActuator`, `dm_inf_funct_location` | 11 (97 acts), `IF_97.npy` | `dm.ifunc_object` (built by `make_rama_calib.py`) |
| `dm_inf_funct_factor` | -1 | `dm.sign` |
| `shiftX/shiftY`, `radial/tangentialScaling`, `dm_flip_ud` | see below | baked into the ifunc file |
| `centralObstruction` 0.34 + spiders | on-sky pupil | `pupilstop` tag `rama_pupil_sky_72p` |
| calibration pupil (clear disk) | `Rama.set_pupil(calibration=True)` | `pupilstop` tag `rama_pupil_calib_72p` (default) |
| `gainCL`, `frame_delay`, `end_mode` (ramatwin.py) | 0.4, 2, 83 | `control.int_gain / delay / n_modes` |

### DM influence functions

`make_rama_calib.py` reproduces exactly what `compute_ramatwin.py` does to
`IF_97.npy`: up-down flip, resampling from the 128 px native grid to the 72 px
simulation grid, and the RAMA mis-registration (shift Y = -`dm_pitch`/3,
radial and tangential scaling 12 %). The transformation chain is a verbatim copy
of `OOPAO.tools.interpolateGeometricalTransformation.interpolate_cube` and was
checked to be **bitwise identical** to OOPAO's output, so OOPAO does not need to
be installed. Changing a mis-registration means editing the constants at the top
of the script and re-running it.

Units: the IFs are stored in nm of wavefront per DM unit (peak 7114 nm/unit),
so raw DM commands are in the native ALPAO units, as in OOPAO
(`dm.coefs` in `dm.modes` units).

### Modal basis

OOPAO computes the KL basis at run time (`compute_KL_basis(tel, atm, dm)`).
SPECULA wants it as a calibration file, so `make_rama_calib.py` calls
`make_modal_base_from_ifs_fft` (same idea: IF geometric covariance +
turbulence covariance for r0 = 0.10 m, L0 = 30 m) and writes `m2c/rama_kl_90.fits`.

The modes are normalised to **1 nm rms of wavefront per unit coefficient**, so
the modal commands seen by `Modalrec`/`Integrator` are in nm rms and the
push-pull amplitude in `calib_rama_rec.yml` (`amp: 1.0`) is the same 1 nm stroke
that `ramatwin.py` uses.

Note: `make_modal_base_from_ifs_fft` assumes influence functions of order unity
(its piston-removal term is not scale-invariant), so the script feeds it a
peak-normalised copy of the IFs. With the raw nm-scale IFs the resulting basis
collapses to a single repeated mode and the interaction matrix comes out rank 1.

### Photometry

OOPAO uses one zero point per band; SPECULA integrates a per-nm flux over the
detector bandwidth. The `zero_point` values in `params_rama.yml` are computed by
`make_rama_calib.py` so that the two agree: the R source delivers
1.131e10 ph/m²/s, exactly OOPAO's `src.nPhoton` at magnitude -0.04.
`quantum_eff: 1.0` because OOPAO applies no QE.

### Field of view

`pyramid.fov: 16.0` is the natural (unvignetted) field for this grid at 640 nm.
Anything smaller makes SPECULA insert a focal-plane stop that OOPAO does not
have; anything much larger is rejected by `calc_pyr_geometry`. If you change
`pixel_pupil`, `pixel_pitch` or the pyramid wavelength, re-check this number —
the log line "FoV reduction from X to Y" tells you the natural value.

## What did not port

* **Bench data products.** `ramatwin.py` loads the measured interaction matrix
  (`IMFull.fits`), the DAO valid-pixel maps (`valid_pixel*.npy`) and the bench
  `M2C.npy` to compare the twin against the real system. SPECULA computes its
  own pupil data and IM instead. Feeding the measured IM into SPECULA is
  possible (write it as an `Intmat` in `calib/im/`) but the signal ordering and
  the valid-pixel set have to be matched to the SPECULA `PyrSlopec` layout first.
* **`Rama.check_pwfs_pupils()`** — the iterative alignment of the four pyramid
  pupils onto the bench valid-pixel map. The SPECULA knob for the same thing is
  `pyramid.pup_shifts`; there is no automatic fit against a measured pupil map.
* **`Rama.calibrate_mis_registration()`** (SPRINT). SPECULA has `SprintPyr`, but
  it needs its own sensitivity-matrix setup; nothing here uses it.
* **Runtime pupil switching** (`Rama.set_pupil(calibration=...)`) is a config
  change here: swap the `pupilstop` tag.
* **Valid-pixel selection.** OOPAO uses `lightRatio = 0`, and its own printout
  confirms it keeps **14400** pixels, i.e. the entire 120x120 frame. The SPECULA
  pupil calibrator thresholds at `thr1 = 0.1 / thr2 = 0.25` and keeps 1102
  pixels per quadrant, 4408 in total. Lower the thresholds in
  `calib_rama_pupdata.yml` to get closer to OOPAO's behaviour. This is a
  difference in the *signal vector*, not in the detector frame, and the
  validation above is unaffected by it (it works on the frames).
* `psfCentering` (hardcoded `True` in `compute_ramatwin.py`) and `pwfs_rooftop`
  have no SPECULA equivalent. `pwfs_rooftop` is unused in OOPAO too — it is set
  in the parameter file but never passed to the `Pyramid` constructor.
* `ramatwin.py` overrides `atm.r0 = 0.05` right before the closed loop, i.e. it
  runs at seeing 2.013" rather than the 1.0065" of the parameter file. The yml
  keeps the parameter-file value; change `seeing.constant` to 2.013 to reproduce
  the tutorial's loop conditions.
