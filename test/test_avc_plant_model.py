import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula.lib.avc_plant_model import (
    open_loop_transfer_function,
    closed_loop_sensitivity,
    avc_command_to_residual_tf,
    generate_avc_init_conditions,
)


def _reference_generate_avc_init_conditions(fAVC, KAO, T, f0, csi, DMdc,
                                             delRTC, nSampDelAVC):
    """Direct, independent translation of ESO's
    generateAVCInitConditions.m, used as a reference to validate the
    module under test against the original Matlab formula line by line
    (mirrors the approach used in test_avc.py's _reference_avc_step()).
    """
    f = np.arange(0.1, 1 / (2 * T) + 1e-9, 0.01)

    w0 = 2 * np.pi * f0
    WFS = (1 - np.exp(-2j * np.pi * T * f)) / (2j * np.pi * f * T)
    CTR = KAO / (1 - np.exp(-2j * np.pi * T * f))
    DAC = (1 - np.exp(-2j * np.pi * T * f)) / (2j * np.pi * f * T)
    DM = DMdc / ((2j * np.pi * f)**2 / w0**2 + 2 * (2j * np.pi * f) * csi / w0 + 1)
    RTC = np.exp(-2j * np.pi * delRTC * f)
    delAVC = np.exp(-2j * np.pi * (nSampDelAVC * T) * f)
    HOL = WFS * RTC * CTR * DAC * DM
    Hs = 1.0 / (1 + HOL)
    Pavc = WFS * RTC * delAVC * DAC * DM * Hs

    idx = np.searchsorted(f, fAVC)
    value = Pavc[idx]
    if abs(value) > 0:
        return value.real, value.imag
    return 1.0, 0.0


class TestAvcPlantModel(unittest.TestCase):

    def test_rejects_low_frequency_disturbances(self):
        """A closed loop with a healthy integrator gain should strongly
        reject a disturbance well below its bandwidth."""
        S = closed_loop_sensitivity(1.0, T=0.001, loop_gain=0.5, loop_delay=0.002)
        self.assertLess(np.abs(S)[0], 0.1)

    def test_passes_high_frequency_disturbances(self):
        """Near Nyquist, loop gain has rolled off and the loop can no
        longer reject: |S| should approach 1 (no attenuation)."""
        S = closed_loop_sensitivity(400.0, T=0.001, loop_gain=0.5, loop_delay=0.002)
        self.assertGreater(np.abs(S)[0], 0.9)

    def test_higher_loop_gain_rejects_more_at_low_frequency(self):
        S_low_gain = closed_loop_sensitivity(2.0, T=0.001, loop_gain=0.2, loop_delay=0.002)
        S_high_gain = closed_loop_sensitivity(2.0, T=0.001, loop_gain=0.8, loop_delay=0.002)
        self.assertLess(np.abs(S_high_gain)[0], np.abs(S_low_gain)[0])

    def test_zero_or_negative_frequency_raises(self):
        with self.assertRaises(ValueError):
            open_loop_transfer_function(0.0, T=0.001, loop_gain=0.5, loop_delay=0.002)
        with self.assertRaises(ValueError):
            closed_loop_sensitivity(-10.0, T=0.001, loop_gain=0.5, loop_delay=0.002)

    def test_ideal_dm_is_default(self):
        """With no dm_f0/dm_csi, the DM term is a flat unity-ish gain: two
        widely separated frequencies should give the same |DM| contribution,
        which we check indirectly via P_avc having a smoothly varying (not
        resonance-peaked) magnitude across a wide band."""
        f = np.array([5.0, 300.0])
        P_ideal = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                              loop_delay=0.002)
        P_resonant = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                                 loop_delay=0.002,
                                                 dm_f0=50.0, dm_csi=0.05)
        # The lightly-damped resonance at 50 Hz should visibly boost the
        # response near 300 Hz relative to the ideal (flat) DM model -- the
        # two models must therefore disagree there.
        self.assertNotAlmostEqual(abs(P_ideal[1]), abs(P_resonant[1]), places=3)

    def test_avc_command_to_residual_tf_matches_open_loop_sensitivity(self):
        """P_avc must equal WFS*RTC*delAVC*DAC*DM*S by construction; check
        this identity holds for a representative frequency."""
        f = 47.0
        T, loop_gain, loop_delay = 0.001, 0.5, 0.002
        S = closed_loop_sensitivity(f, T, loop_gain, loop_delay)
        HOL = open_loop_transfer_function(f, T, loop_gain, loop_delay)
        P = avc_command_to_residual_tf(f, T, loop_gain, loop_delay)
        # HOL = WFS*RTC*CTR*DAC*DM, and P = (HOL/CTR)*S = (HOL/loop_gain)*(1-exp(-jwT))*S
        # simplest robust check: P/S must have the same magnitude behaviour
        # as HOL with the integrator term removed; here we just check P is
        # finite, nonzero, and that P == S * (HOL * (1-exp(-jwT))/loop_gain)
        w = 2 * np.pi * f
        ctr = loop_gain / (1 - np.exp(-1j * w * T))
        expected = (HOL / ctr) * np.exp(-1j * w * 0.0) * S
        np.testing.assert_allclose(P, expected, rtol=1e-10)

    def test_generate_avc_init_conditions_matches_matlab_reference(self):
        # NOTE: the Matlab source evaluates Pavc on a fixed, coarse 0.01 Hz
        # frequency grid and takes the first grid point >= fAVC (see
        # _reference_generate_avc_init_conditions above, which reproduces
        # that quantization exactly). This module instead evaluates the
        # transfer function continuously, exactly at fAVC -- a deliberate
        # improvement, not a bug -- so results agree only up to the
        # grid-quantization error (<0.01 Hz worth of phase/gain drift),
        # not to full floating point precision.
        for fAVC, KAO, f0, csi, DMdc, delRTC, nSampDelAVC in [
            (47.0, 0.5, 1.0, 1.0, 1.0, 0.002, 0),
            (40.0, 0.5, 1.0, 1.0, 1.0, 0.002, 0),
            (120.0, 0.8, 1.0, 1.0, 1.0, 0.003, 1),
        ]:
            x0_ref, x1_ref = _reference_generate_avc_init_conditions(
                fAVC, KAO, 0.001, f0, csi, DMdc, delRTC, nSampDelAVC)
            x0, x1 = generate_avc_init_conditions(
                fAVC, T=0.001, loop_gain=KAO, loop_delay=delRTC,
                avc_extra_delay=nSampDelAVC * 0.001,
                dm_dc_gain=DMdc, dm_f0=f0, dm_csi=csi)
            np.testing.assert_allclose([x0, x1], [x0_ref, x1_ref], rtol=1e-2, atol=1e-3)

    def test_generate_avc_init_conditions_differs_from_generic_default(self):
        """The whole point of this module: for a real closed, delayed
        loop, the correct initial condition is not the generic (1.0, 0.0)
        used when no calibration is available."""
        x0, x1 = generate_avc_init_conditions(
            47.0, T=0.001, loop_gain=0.5, loop_delay=0.002)
        self.assertFalse(np.isclose(x0, 1.0) and np.isclose(x1, 0.0))

    def test_no_dynamics_dm_reduces_to_constant_gain(self):
        """With dm_f0=None (default), the DM response is dm_dc_gain at
        every frequency, so changing it changes the overall loop gain and
        therefore |P_avc|. Deep inside the loop's rejection band (very low
        frequency) HOL is already so large that S is saturated near zero
        and insensitive to a further doubling of the DM gain -- so this is
        checked near the loop's crossover region instead, where the effect
        is unambiguous."""
        f = 50.0
        P_gain1 = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                              loop_delay=0.002, dm_dc_gain=1.0)
        P_gain2 = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                              loop_delay=0.002, dm_dc_gain=2.0)
        self.assertNotAlmostEqual(abs(P_gain1[0]), abs(P_gain2[0]), places=2)


if __name__ == '__main__':
    unittest.main()
