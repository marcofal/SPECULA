"""Run the AVCSimplified demo, optionally sweeping the plant phase error.

Each run is the full closed-loop SCAO simulation of
``params_scao_avc_simplified.yml``; the only thing that changes between runs
is ``plant_phase``, i.e. the phase of the calibration handed to the AVC
module. The true closed-loop plant phase at 47 Hz is +169.52 deg, so a run
labelled ``d60`` is calibrated 60 deg away from the truth and a run labelled
``d120`` is outside the 90 deg stability cone.

Usage
-----
    python run_avc_simplified.py [name ...]

with names from ``RUNS`` below; with no arguments, everything is run.
Results land in ``output/SIMPL_<name>/<timestamp>/`` and are analysed by
``analyze_avc_simplified.py``.
"""
import sys

import yaml

# Measured closed-loop plant at the vibration frequency
# (params_measure_plant.yml): |P| = 7.223, arg P = +169.52 deg.
PLANT_GAIN = 7.223
PLANT_PHASE = 169.52

# name -> (phase error [deg] added to the true plant phase, track_frequency).
# A phase error of None means "no AVC at all" (the baseline loop).
#
# The 'd*' family is the operational configuration, with the PLL running.
# The 'f*' family holds the frequency fixed at the true 47 Hz (gomega = 0) so
# that the phase-margin effect is seen on its own: with a large calibration
# error the tone is not cancelled, the residual stays large, and the PLL is
# then free to be dragged onto whatever else dominates it -- which confounds
# the comparison. Fixing the frequency removes that second mechanism.
RUNS = {
    'baseline': (None, True),
    'd0':   (0.0, True),
    'd60':  (60.0, True),
    'd85':  (85.0, True),
    'd95':  (95.0, True),
    'd120': (120.0, True),
    'f0':   (0.0, False),
    'f60':  (60.0, False),
    'f85':  (85.0, False),
    'f95':  (95.0, False),
    'f120': (120.0, False),
    # finer sweep, both signs, used for the margin figure
    'sm85': (-85.0, False),
    'sm60': (-60.0, False),
    'sm30': (-30.0, False),
    'sp30': (30.0, False),
    'sp70': (70.0, False),
    'sp80': (80.0, False),
    # the same cone re-centred by one loop period (16.9 deg at 47 Hz), which
    # is what the calibration harness omits -- see the report. These three
    # are what shows the cone is symmetric, and where its centre really is.
    'cm80': (-16.9 - 80.0, False),
    'c00':  (-16.9, False),
    'cp80': (-16.9 + 80.0, False),
}


def build_params(name, spec):
    """Return a params dict for one run."""
    dphi, track = spec
    if dphi is None:
        params = yaml.safe_load(open('params_scao_vibration_baseline.yml'))
    else:
        params = yaml.safe_load(open('params_scao_avc_simplified.yml'))
        avc = params['avc']
        # switch to the polar form of the calibration and offset the phase
        avc.pop('plant_re', None)
        avc.pop('plant_im', None)
        avc['plant_gain'] = PLANT_GAIN
        avc['plant_phase'] = PLANT_PHASE + dphi
        if not track:
            avc['gomega'] = 0.0
            avc['freq'] = [47.0]
    params['data_store']['store_dir'] = f'./output/SIMPL_{name}'
    return params


def main(names):
    import specula
    specula.init(-1)
    from specula.simul import Simul

    for name in names:
        params = build_params(name, RUNS[name])
        path = f'/tmp/params_SIMPL_{name}.yml'
        yaml.safe_dump(params, open(path, 'w'))
        print(f"=== running {name} {RUNS[name]}", flush=True)
        Simul(path).run()


if __name__ == '__main__':
    requested = sys.argv[1:] or list(RUNS)
    unknown = [n for n in requested if n not in RUNS]
    if unknown:
        raise SystemExit(f'unknown run name(s): {unknown}; choose from {list(RUNS)}')
    main(requested)
