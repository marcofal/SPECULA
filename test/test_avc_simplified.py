import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula import cpuArray
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula.processing_objects.avc import AVC
from specula.processing_objects.avc_simplified import AVCSimplified

from test.specula_testlib import cpu_and_gpu


def _run_closed_loop(obj, n_steps, p_star, lam_star, target_device_idx, xp):
    """Drive an AVC-like object with a synthetic single-tone loop.

    The residual seen by the object is the projection of the phasor
    ``Z = theta * conj(P_star) + Lambda_star`` on the instantaneous phase,
    which is the model the report's analysis is built on. The object's own
    command is fed back through the true plant ``P_star``, so the loop is
    closed exactly as in the derivation.

    Returns the recorded residual history.
    """
    measurement = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
    obj.inputs['in_measurement'].set(measurement)
    obj.setup()

    ts = float(obj._T)
    omega_true = 2 * np.pi * 47.0
    phase = 0.0
    residuals = np.zeros(n_steps)
    comm = 0.0

    for i in range(n_steps):
        # true disturbance plus the effect of the previous command, both
        # projected on the true instantaneous phase
        z = comm * np.conj(p_star) + lam_star
        y = z.real * np.cos(phase) + z.imag * np.sin(phase)
        residuals[i] = y

        t = obj.seconds_to_t((i + 1) * ts)
        measurement.value[:] = xp.asarray([y])
        measurement.generation_time = t
        obj.check_ready(t)
        obj.trigger()
        obj.post_trigger()

        # the emitted command, as a phasor in the object's own frame
        state = cpuArray(obj.outputs['out_state'].value)[0]
        x1, x2, x3, x4 = state
        denom = x1 * x1 + x2 * x2
        comm = -(x1 + 1j * x2) * (x3 + 1j * x4) / denom

        phase += omega_true * ts

    return residuals


class TestAVCSimplified(unittest.TestCase):

    @cpu_and_gpu
    def test_construction_and_shapes(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        n_avc = 3
        avc = AVCSimplified(simul_params=simul_params, n_avc=n_avc,
                            freq=[10.0, 20.0, 30.0], gx=0.05, gomega=0.001, k=1.0,
                            plant_re=[1.0, 2.0, 3.0], plant_im=[0.0, 1.0, -1.0],
                            target_device_idx=target_device_idx)

        self.assertEqual(avc.outputs['out_comm'].value.shape, (n_avc,))
        self.assertEqual(avc.outputs['out_freq'].value.shape, (n_avc,))
        self.assertEqual(avc.outputs['out_state'].value.shape, (n_avc, 4))
        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_avc))

    @cpu_and_gpu
    def test_invalid_n_avc_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVCSimplified(simul_params=simul_params, n_avc=0,
                          freq=10.0, gx=0.05, gomega=0.001, k=1.0,
                          target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_mismatched_parameter_length_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVCSimplified(simul_params=simul_params, n_avc=3,
                          freq=[10.0, 20.0], gx=0.05, gomega=0.001, k=1.0,
                          target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_zero_plant_raises(self, target_device_idx, xp):
        """The calibration sits in the denominator of the fixed map."""
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                          gx=3.0, gomega=0.2, k=1.0,
                          plant_re=0.0, plant_im=0.0,
                          target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_cartesian_and_polar_calibration_agree(self, target_device_idx, xp):
        """The same plant given both ways must give the same fixed map."""
        simul_params = SimulParams(time_step=0.001)
        rho, psi_deg = 0.555, 119.0
        p_re = rho * np.cos(np.radians(psi_deg))
        p_im = rho * np.sin(np.radians(psi_deg))

        a = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                          gx=3.0, gomega=0.2, k=1.0,
                          plant_re=p_re, plant_im=p_im,
                          target_device_idx=target_device_idx)
        b = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                          gx=3.0, gomega=0.2, k=1.0,
                          plant_gain=rho, plant_phase=psi_deg,
                          target_device_idx=target_device_idx)

        for name in ('_m11', '_m12', '_m21', '_m22'):
            np.testing.assert_allclose(cpuArray(getattr(a, name)),
                                       cpuArray(getattr(b, name)), rtol=1e-6)

    @cpu_and_gpu
    def test_both_calibration_forms_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                          gx=3.0, gomega=0.2, k=1.0,
                          plant_re=1.0, plant_gain=1.0,
                          target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_first_trigger_outputs_zero_correction(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        n_avc = 2
        avc = AVCSimplified(simul_params=simul_params, n_avc=n_avc, freq=[25.0, 55.0],
                            gx=0.05, gomega=0.001, k=1.0,
                            target_device_idx=target_device_idx)

        measurement = BaseValue(value=xp.array([0.3, -0.2]), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(measurement)
        avc.setup()

        t = avc.seconds_to_t(0.001)
        measurement.generation_time = t
        avc.check_ready(t)
        avc.trigger()
        avc.post_trigger()

        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_avc))

    @cpu_and_gpu
    def test_plant_state_never_changes(self, target_device_idx, xp):
        """x1, x2 are a calibration, not a state: they must be bit-stable."""
        simul_params = SimulParams(time_step=0.001)
        p_star = 0.555 * np.exp(1j * np.radians(119.0))
        avc = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                            gx=3.0, gomega=0.2, k=1.0,
                            plant_re=p_star.real, plant_im=p_star.imag,
                            target_device_idx=target_device_idx)
        _run_closed_loop(avc, 2000, p_star, 120.0 * np.exp(0.7j),
                         target_device_idx, xp)

        state = cpuArray(avc.outputs['out_state'].value)[0]
        self.assertEqual(state[0], cpuArray(avc._plant_re)[0])
        self.assertEqual(state[1], cpuArray(avc._plant_im)[0])

    @cpu_and_gpu
    def test_matches_full_avc_with_plant_frozen(self, target_device_idx, xp):
        """The reduction is exact.

        With ``adapt_plant=False`` and ``theta_min_energy=0`` (soft clamp,
        i.e. epsilon = 0) the full AVC object *is* this algorithm. Driving
        both with the same residual stream must give the same states.
        """
        simul_params = SimulParams(time_step=0.001)
        # deliberately mis-calibrated plant: 10% in magnitude, 31 deg in phase
        p_hat = 0.500 * np.exp(1j * np.radians(150.0))

        common = dict(simul_params=simul_params, n_avc=1, freq=45.0,
                      gx=3.0, gomega=0.2, k=1.0,
                      target_device_idx=target_device_idx)
        full = AVC(c=1.0, x0=p_hat.real, x1=p_hat.imag,
                   theta_min_energy=0.0, soft_clamp=True, adapt_plant=False,
                   **common)
        simple = AVCSimplified(plant_re=p_hat.real, plant_im=p_hat.imag, **common)

        m_full = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        m_simple = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        full.inputs['in_measurement'].set(m_full)
        simple.inputs['in_measurement'].set(m_simple)
        full.setup()
        simple.setup()

        rng = np.random.default_rng(1234)
        ts = 0.001
        for i in range(20000):
            # an arbitrary, non-degenerate residual stream
            y = float(np.sin(2 * np.pi * 47.0 * i * ts) + 0.1 * rng.normal())
            t = full.seconds_to_t((i + 1) * ts)
            for obj, m in ((full, m_full), (simple, m_simple)):
                m.value[:] = xp.asarray([y])
                m.generation_time = t
                obj.check_ready(t)
                obj.trigger()
                obj.post_trigger()

        s_full = cpuArray(full.outputs['out_state'].value)[0]
        s_simple = cpuArray(simple.outputs['out_state'].value)[0]
        np.testing.assert_allclose(s_simple, s_full, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(cpuArray(simple.outputs['out_comm'].value),
                                   cpuArray(full.outputs['out_comm'].value),
                                   rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(cpuArray(simple.outputs['out_freq'].value),
                                   cpuArray(full.outputs['out_freq'].value),
                                   rtol=1e-5, atol=1e-5)

    @cpu_and_gpu
    def test_cancels_tone_with_exact_calibration(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        p_star = 0.555 * np.exp(1j * np.radians(119.0))
        avc = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                            gx=3.0, gomega=0.0, k=1.0,
                            plant_re=p_star.real, plant_im=p_star.imag,
                            target_device_idx=target_device_idx)
        res = _run_closed_loop(avc, 20000, p_star, 120.0 * np.exp(0.7j),
                               target_device_idx, xp)

        start = res[:500].std()
        end = res[-500:].std()
        self.assertLess(end, 0.01 * start)

    @cpu_and_gpu
    def test_converges_inside_the_margin_and_diverges_outside(self, target_device_idx, xp):
        """The 90 degree condition, which is the object's whole contract.

        Only the *phase* of the calibration is varied; the magnitude is
        left exact, so kappa has unit modulus and the sign of Re(kappa) is
        decided by the phase error alone.
        """
        simul_params = SimulParams(time_step=0.001)
        p_star = 0.555 * np.exp(1j * np.radians(119.0))
        lam_star = 120.0 * np.exp(0.7j)

        def final_over_initial(dphi_deg):
            p_hat = abs(p_star) * np.exp(1j * (np.angle(p_star) + np.radians(dphi_deg)))
            avc = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                                gx=3.0, gomega=0.0, k=1.0,
                                plant_re=p_hat.real, plant_im=p_hat.imag,
                                target_device_idx=target_device_idx)
            res = _run_closed_loop(avc, 20000, p_star, lam_star,
                                   target_device_idx, xp)
            return res[-500:].std() / res[:500].std()

        # comfortably inside the cone: strong rejection
        self.assertLess(final_over_initial(0.0), 0.01)
        self.assertLess(final_over_initial(60.0), 0.05)
        # just inside: still decaying, if slowly
        self.assertLess(final_over_initial(85.0), 1.0)
        # just outside: growing
        self.assertGreater(final_over_initial(95.0), 1.0)
        # well outside: divergence
        self.assertGreater(final_over_initial(120.0), 100.0)

    @cpu_and_gpu
    def test_magnitude_error_only_slows_convergence(self, target_device_idx, xp):
        """|kappa| is absent from the stability condition: a badly scaled
        calibration must still converge, only more slowly."""
        simul_params = SimulParams(time_step=0.001)
        p_star = 0.555 * np.exp(1j * np.radians(119.0))
        lam_star = 120.0 * np.exp(0.7j)

        ratios = []
        for scale in (1.0, 5.0):
            p_hat = scale * p_star
            avc = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                                gx=3.0, gomega=0.0, k=1.0,
                                plant_re=p_hat.real, plant_im=p_hat.imag,
                                target_device_idx=target_device_idx)
            res = _run_closed_loop(avc, 20000, p_star, lam_star,
                                   target_device_idx, xp)
            ratios.append(res[-500:].std() / res[:500].std())

        # both converge ...
        self.assertLess(ratios[0], 0.01)
        self.assertLess(ratios[1], 0.5)
        # ... but the mis-scaled one is slower
        self.assertGreater(ratios[1], ratios[0])

    @cpu_and_gpu
    def test_frequency_tracking_converges_on_pure_tone(self, target_device_idx, xp):
        """Open-loop PLL check, mirroring the equivalent test for AVC.

        The frequency loop is unchanged from the original algorithm, so it
        is exercised the same way: feed a pure sinusoid and check the
        estimate is pulled onto it from a wrong initial guess. Gains are
        larger than a real deployment would use, purely to keep the
        iteration count down.
        """
        T = 0.001
        true_freq = 50.0
        simul_params = SimulParams(time_step=T)
        avc = AVCSimplified(simul_params=simul_params, n_avc=1, freq=48.0,
                            gx=0.5, gomega=5.0, k=1.0,
                            target_device_idx=target_device_idx)

        meas = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(meas)
        avc.setup()

        for it in range(20000):
            t_sec = (it + 1) * T
            t = avc.seconds_to_t(t_sec)
            meas.value[:] = xp.asarray([0.7 * np.sin(2 * np.pi * true_freq * t_sec)])
            meas.generation_time = t
            avc.check_ready(t)
            avc.trigger()
            avc.post_trigger()

        final_freq = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(final_freq, true_freq, delta=1.0)

    @cpu_and_gpu
    def test_gomega_zero_holds_frequency(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        avc = AVCSimplified(simul_params=simul_params, n_avc=1, freq=47.0,
                            gx=3.0, gomega=0.0, k=1.0,
                            target_device_idx=target_device_idx)
        p_star = 0.555 * np.exp(1j * np.radians(119.0))
        _run_closed_loop(avc, 5000, p_star, 120.0 * np.exp(0.7j),
                         target_device_idx, xp)

        f_est = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(f_est, 47.0, places=4)


if __name__ == '__main__':
    unittest.main()
