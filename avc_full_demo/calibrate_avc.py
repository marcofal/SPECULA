#!/usr/bin/env python3
"""
Compute a closed-loop-consistent AVC initial state (x0, x1) for this demo,
using specula.lib.avc_plant_model (a Python port of ESO's
generateAVCInitConditions.m).

This is the "offline calibration" step the real Matlab/HRTC AVC pipeline
always performs before closing an AVC loop (see
specula/lib/avc_plant_model.py's docstring, and avc_full_demo/README.md for
why the generic default x0=1, x1=0 is unstable in this closed loop).

Run with:

    python calibrate_avc.py

and copy the printed x0/x1 into the avc: block of
params_scao_avc_full_demo.yml. Re-run this whenever any of the loop
parameters below change (freq guess, time_step).

WARNING (sign convention): the analytic model here matches ESO's MATLAB
generateAVCInitConditions.m (verified in test/test_avc_plant_model.py),
but that model does NOT capture the sign/phase of SPECULA's specific loop
wiring. Measuring the true command->residual transfer directly from a
SPECULA probe run (params_measure_plant.yml) gives, at 47 Hz,
|P|=0.555 phase=+119 deg, whereas this analytic model gives |P|=0.61
phase=-78 deg -- i.e. ~180 deg apart (SPECULA's DM correction enters the
residual with the opposite sign to what the generic model assumes).
Feeding AVC the analytic x0,x1 therefore mis-phases the correction by
~180 deg so it AMPLIFIES the vibration instead of cancelling it. For the
SPECULA demo, use the MEASURED value from params_measure_plant.yml (that
is what params_scao_avc_full_demo.yml now does, and it gives real
cancellation: -9.5% on mode 0). This script is kept for reference and for
the (correct) closed-loop shape it produces; see avc_full_demo/README.md.

NOTE 2: LPF_FC models a genuine first-order actuator/DM low-pass filter
(a SPECULA ``LowPassFilter(n_ord=1, cutoff_freq=LPF_FC)`` physically
inserted between avc_inserter and dm in params_scao_avc_full_demo.yml),
in place of the previous "ideal, dynamics-free DM" (dm_dc_gain=1, no
lpf_fc). Without it, the loop's only dynamics are a near-unit-gain,
near-zero-phase one-frame delay -- almost an allpass at the vibration
frequency, which is a poorly conditioned regression target for AVC's own
plant-identification step (see avc_full_demo/README.md). A real low-pass
gives the loop actual, frequency-distinguishing magnitude/phase rolloff
near the vibration frequency instead.
"""
from specula.lib.avc_plant_model import generate_avc_init_conditions

# ---- must match params_scao_avc_full_demo.yml ----
T = 0.001            # main.time_step [s]
LOOP_GAIN = 0.0       # no integrator in this demo anymore -- AVC is the
                       # sole controller on mode 0
LOOP_DELAY = T        # one simulation frame, from prop's 'dm.out_layer:-1'
                       # frame-delayed reference (not an integrator delay)
AVC_EXTRA_DELAY = 0.0  # AVC's own correction path has no additional delay
                        # beyond the one above
FREQ_GUESS = 40.0     # avc.freq[0], the initial frequency guess
LPF_FC = 80.0         # [Hz] first-order actuator/DM low-pass cutoff --
                       # must match the 'lowpass' block's cutoff_freq in
                       # params_scao_avc_full_demo.yml. ~1.7x the true
                       # vibration frequency (47Hz): gives substantial
                       # magnitude attenuation (|P|~0.61) and phase lag
                       # (~-78deg) at 47Hz without sitting right on the
                       # pole (fc==47Hz would be a marginal edge case).
# SPECULA's DM processing object itself has no internal dynamics (ideal,
# instantaneous modal mirror) -> dm_dc_gain=1.0, no resonance (dm_f0/dm_csi
# left at their default None); all the new dynamics come from LPF_FC.


def main():
    x0, x1 = generate_avc_init_conditions(
        FREQ_GUESS, T, LOOP_GAIN, LOOP_DELAY, avc_extra_delay=AVC_EXTRA_DELAY,
        lpf_fc=LPF_FC)
    magnitude = (x0**2 + x1**2) ** 0.5
    phase_deg = __import__('math').degrees(__import__('math').atan2(x1, x0))

    print(f"Loop parameters: T={T}s, loop_gain={LOOP_GAIN}, "
          f"loop_delay={LOOP_DELAY}s, avc_extra_delay={AVC_EXTRA_DELAY}s, "
          f"lpf_fc={LPF_FC}Hz")
    print(f"AVC frequency guess: {FREQ_GUESS} Hz")
    print()
    print(f"Calibrated initial state: x0={x0:.6f}  x1={x1:.6f}")
    print(f"  (magnitude={magnitude:.6f}, phase={phase_deg:.1f} deg -- "
          f"compare to the generic default x0=1.0, x1=0.0, "
          f"magnitude=1.0, phase=0.0 deg)")
    print()
    print("Paste into the avc: block of params_scao_avc_full_demo.yml:")
    print(f"  x0:                {x0:.6f}")
    print(f"  x1:                {x1:.6f}")


if __name__ == '__main__':
    main()
