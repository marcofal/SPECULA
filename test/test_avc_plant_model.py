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
    """Line by line translation of ESO's generateAVCInitConditions.m,
    used as a reference."""
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
        S = closed_loop_sensitivity(1.0, T=0.001, loop_gain=0.5, loop_delay=0.002)
        self.assertLess(np.abs(S)[0], 0.1)

    def test_passes_high_frequency_disturbances(self):
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
        """A DM resonance must change the response w.r.t. the ideal DM."""
        f = np.array([5.0, 300.0])
        P_ideal = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                              loop_delay=0.002)
        P_resonant = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                                 loop_delay=0.002,
                                                 dm_f0=50.0, dm_csi=0.05)
        self.assertNotAlmostEqual(abs(P_ideal[1]), abs(P_resonant[1]), places=3)

    def test_avc_command_to_residual_tf_matches_open_loop_sensitivity(self):
        """P_avc = -(HOL / CTR) * S."""
        f = 47.0
        T, loop_gain, loop_delay = 0.001, 0.5, 0.002
        S = closed_loop_sensitivity(f, T, loop_gain, loop_delay)
        HOL = open_loop_transfer_function(f, T, loop_gain, loop_delay)
        P = avc_command_to_residual_tf(f, T, loop_gain, loop_delay)
        w = 2 * np.pi * f
        ctr = loop_gain / (1 - np.exp(-1j * w * T))
        expected = -(HOL / ctr) * S
        np.testing.assert_allclose(P, expected, rtol=1e-10)

    def test_generate_avc_init_conditions_matches_matlab_reference(self):
        # The Matlab source evaluates Pavc on a 0.01 Hz grid, this module
        # exactly at fAVC: they agree only up to the grid quantization.
        # The sign is opposite, see the module docstring.
        for fAVC, KAO, f0, csi, DMdc, delRTC, nSampDelAVC in [
            (47.0, 0.5, 1.0, 1.0, 1.0, 0.002, 0),
            (40.0, 0.5, 1.0, 1.0, 1.0, 0.002, 0),
            (120.0, 0.8, 1.0, 1.0, 1.0, 0.003, 1),
        ]:
            re_ref, im_ref = _reference_generate_avc_init_conditions(
                fAVC, KAO, 0.001, f0, csi, DMdc, delRTC, nSampDelAVC)
            p_re, p_im = generate_avc_init_conditions(
                fAVC, T=0.001, loop_gain=KAO, loop_delay=delRTC,
                avc_extra_delay=nSampDelAVC * 0.001,
                dm_dc_gain=DMdc, dm_f0=f0, dm_csi=csi)
            np.testing.assert_allclose([p_re, p_im], [-re_ref, -im_ref], rtol=1e-2, atol=1e-3)

    def test_matches_specula_scao_loop(self):
        """Response measured in the params_scao_sh_test.yml loop (int_gain
        0.5, delay 2) by injecting a 47 Hz tone on the tip command,
        downstream of the integrator: 0.80 at 255 degrees."""
        T = 0.001
        p = avc_command_to_residual_tf(47.0, T, loop_gain=0.5, loop_delay=2 * T,
                                        avc_extra_delay=-2 * T)[0]
        self.assertAlmostEqual(abs(p), 0.80, delta=0.02)
        self.assertAlmostEqual(np.degrees(np.angle(p)) % 360, 255.0, delta=10.0)

    def test_generate_avc_init_conditions_differs_from_generic_default(self):
        """A closed, delayed loop is not the default (1.0, 0.0) plant."""
        p_re, p_im = generate_avc_init_conditions(
            47.0, T=0.001, loop_gain=0.5, loop_delay=0.002)
        self.assertFalse(np.isclose(p_re, 1.0) and np.isclose(p_im, 0.0))

    def test_no_dynamics_dm_reduces_to_constant_gain(self):
        """Without DM dynamics, dm_dc_gain scales the loop gain. Checked
        near crossover, where |P_avc| is sensitive to it."""
        f = 50.0
        P_gain1 = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                              loop_delay=0.002, dm_dc_gain=1.0)
        P_gain2 = avc_command_to_residual_tf(f, T=0.001, loop_gain=0.5,
                                              loop_delay=0.002, dm_dc_gain=2.0)
        self.assertNotAlmostEqual(abs(P_gain1[0]), abs(P_gain2[0]), places=2)


if __name__ == '__main__':
    unittest.main()
