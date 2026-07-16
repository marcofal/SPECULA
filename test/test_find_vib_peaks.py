import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula.lib.find_vib_peaks import find_vib_peaks, fit_ar2, VibPeaks, _ar2_psd_model


def _make_synthetic_psd(freq, peaks, floor=0.05, noise_std=0.0, seed=None):
    """Build a synthetic PSD as a noise floor plus AR2-shaped peaks with
    given (freq, damping, power), using the same model the fitter assumes.
    """
    df = freq[1] - freq[0]
    psd = np.full_like(freq, floor)
    for p in peaks:
        model = _ar2_psd_model(freq, 1.0 / (freq[-1] * 2), p['f'], p['k'])
        model = model / (np.sum(model) * df) * p['sigma']
        psd += model
    if noise_std > 0:
        rng = np.random.RandomState(seed)
        psd = psd * (1 + noise_std * rng.randn(*psd.shape))
    return np.clip(psd, 1e-8, None)


class TestFindVibPeaks(unittest.TestCase):

    def test_single_peak_detected(self):
        freq = np.linspace(0.1, 250.0, 4096)
        peaks = [dict(f=63.0, k=0.004, sigma=30.0)]
        psd = _make_synthetic_psd(freq, peaks, noise_std=0.03, seed=1)

        result = find_vib_peaks(psd, freq, fmin=1.0, fmax=200.0)

        self.assertIsInstance(result, VibPeaks)
        self.assertEqual(result.freq.size, 1)
        self.assertAlmostEqual(result.freq[0], 63.0, delta=1.0)
        self.assertGreater(result.power[0], 0)
        self.assertGreater(result.damping[0], 0)

    def test_multiple_peaks_detected_and_sorted_by_frequency(self):
        freq = np.linspace(0.1, 250.0, 4096)
        peaks = [
            dict(f=23.4, k=0.003, sigma=50.0),
            dict(f=87.1, k=0.006, sigma=20.0),
            dict(f=150.0, k=0.002, sigma=10.0),
        ]
        psd = _make_synthetic_psd(freq, peaks, noise_std=0.05, seed=7)

        result = find_vib_peaks(psd, freq, fmin=1.0, fmax=200.0)

        self.assertGreaterEqual(result.freq.size, len(peaks))
        # Result must be sorted by ascending frequency
        np.testing.assert_array_equal(result.freq, np.sort(result.freq))

        df_bin = freq[1] - freq[0]
        for p in peaks:
            closest = result.freq[np.argmin(np.abs(result.freq - p['f']))]
            self.assertLess(abs(closest - p['f']), 5 * df_bin)

    def test_flat_psd_returns_no_peaks(self):
        freq = np.linspace(0.1, 250.0, 2048)
        rng = np.random.RandomState(3)
        psd = np.full_like(freq, 0.1) * (1 + 0.02 * rng.randn(freq.size))

        result = find_vib_peaks(psd, freq, fmin=1.0, fmax=200.0)

        self.assertEqual(result.freq.size, 0)
        self.assertEqual(result.damping.size, 0)
        self.assertEqual(result.power.size, 0)

    def test_band_too_narrow_returns_no_peaks(self):
        freq = np.linspace(0.1, 250.0, 2048)
        psd = np.full_like(freq, 0.1)

        result = find_vib_peaks(psd, freq, fmin=1.0, fmax=2.0, num_sampl_init=50)

        self.assertEqual(result.freq.size, 0)

    def test_mismatched_shapes_raise(self):
        freq = np.linspace(0.1, 250.0, 2048)
        psd = np.ones(10)

        with self.assertRaises(ValueError):
            find_vib_peaks(psd, freq, fmin=1.0, fmax=200.0)

    def test_low_power_peak_is_rejected(self):
        """A peak that carries a negligible fraction of the in-band power
        must not be reported (power_ratio_threshold gating)."""
        freq = np.linspace(0.1, 250.0, 4096)
        # Very high floor compared to peak power
        peaks = [dict(f=100.0, k=0.005, sigma=0.01)]
        psd = _make_synthetic_psd(freq, peaks, floor=5.0)

        result = find_vib_peaks(psd, freq, fmin=1.0, fmax=200.0,
                                 power_ratio_threshold=0.01)

        self.assertEqual(result.freq.size, 0)


class TestFitAr2(unittest.TestCase):

    def test_recovers_known_single_peak(self):
        freq = np.linspace(40.0, 60.0, 512)
        true_f, true_k = 50.0, 0.005
        fs = 500.0
        model = _ar2_psd_model(freq, 1.0 / fs, true_f, true_k)
        df = freq[1] - freq[0]
        power = model / (np.sum(model) * df) * 10.0

        sigma, k, f_vib = fit_ar2(power, freq, 1.0 / fs, threshold=0.5, df=df)

        self.assertEqual(f_vib.size, 1)
        self.assertAlmostEqual(f_vib[0], true_f, delta=0.5)
        self.assertGreater(sigma[0], 0)
        self.assertGreater(k[0], 0)

    def test_output_sorted_by_frequency(self):
        freq = np.linspace(1.0, 100.0, 1024)
        # Two well-separated peaks packed in one segment to exercise the
        # secondary-peak branch.
        peaks = [dict(f=30.0, k=0.004, sigma=20.0), dict(f=70.0, k=0.004, sigma=15.0)]
        power = _make_synthetic_psd(freq, peaks, floor=0.01)
        df = freq[1] - freq[0]

        sigma, k, f_vib = fit_ar2(power, freq, 1.0 / (freq[-1] * 2), threshold=0.3, df=df)

        np.testing.assert_array_equal(f_vib, np.sort(f_vib))


if __name__ == '__main__':
    unittest.main()
