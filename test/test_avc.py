import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula import cpuArray
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula.processing_objects.avc import AVC

from test.specula_testlib import cpu_and_gpu


def _reference_avc_step(state, measurement):
    """
    Direct, scalar, 1:1 Python translation of the Matlab ``updateAVC.m`` /
    ``initAVC.m`` algorithm, used as an independent ground truth to check
    the vectorized :class:`~specula.processing_objects.avc.AVC` object
    against. ``state`` is a dict with keys
    ``x`` (length-4 array), ``omega``, ``alpha``, ``alpha_c``, ``T``,
    ``gx``, ``gomega``, ``k``, ``c``, ``n``, and is updated in place.
    Returns the correction u[k] that was output *before* this update
    (matching AVC's "apply previous correction, then update" semantics).
    """
    x = state['x']
    omega = state['omega']
    alpha = state['alpha']
    alpha_c = state['alpha_c']
    T = state['T']
    gx = state['gx']
    gomega = state['gomega']
    k = state['k']
    c = state['c']
    n = state['n']

    out = state['correction']

    if abs(x[0] ** 2 + x[1] ** 2) < 1e-2:
        theta = np.array([1.0, 0.0])
    else:
        f = -(1.0 / (x[0] ** 2 + x[1] ** 2))
        theta = np.array([f * (x[0] * x[2] - x[1] * x[3]),
                           f * (x[0] * x[3] + x[1] * x[2])])

    thetac, thetas = theta
    alphaold = alpha

    w = np.array([
        thetac * np.cos(alpha) + thetas * np.sin(alpha),
        thetas * np.cos(alpha) - thetac * np.sin(alpha),
        np.cos(alpha),
        np.sin(alpha),
    ])

    err = float(np.dot(w, x)) - measurement

    x[0] = x[0] - gx * T * w[0] * err * c
    x[1] = x[1] - gx * T * w[1] * err * c
    x[2] = x[2] - gx * T * w[2] * err
    x[3] = x[3] - gx * T * w[3] * err

    alpha_input = T * omega / n - 2 * k * T / n * gomega * np.sin(alpha) * (-err)
    alpha_y = alpha_input - alpha_c
    alpha_t = alpha + alpha_y
    alpha_c = (alpha_t - alpha) - alpha_y
    alpha = alpha_t
    if alpha > np.pi:
        alpha -= 2 * np.pi
    elif alpha < -np.pi:
        alpha += 2 * np.pi

    omega = omega - 2 * T * gomega * np.sin(alphaold) * (-err)

    if abs(x[0] ** 2 + x[1] ** 2) < 1e-2:
        theta = np.array([1.0, 0.0])
    else:
        f = -(1.0 / (x[0] ** 2 + x[1] ** 2))
        theta = np.array([f * (x[0] * x[2] - x[1] * x[3]),
                           f * (x[0] * x[3] + x[1] * x[2])])

    state['x'] = x
    state['omega'] = omega
    state['alpha'] = alpha
    state['alpha_c'] = alpha_c
    state['correction'] = np.cos(alpha) * theta[0] + np.sin(alpha) * theta[1]

    return out


class TestAVC(unittest.TestCase):

    @cpu_and_gpu
    def test_construction_and_shapes(self, target_device_idx, xp):
        """Basic construction with array and scalar (broadcast) parameters."""
        simul_params = SimulParams(time_step=0.001)
        n_avc = 3
        avc = AVC(simul_params=simul_params, n_avc=n_avc,
                  freq=[10.0, 20.0, 30.0], gx=0.05, gomega=0.001, k=1.0, c=1.0,
                  target_device_idx=target_device_idx)

        self.assertEqual(avc.outputs['out_comm'].value.shape, (n_avc,))
        self.assertEqual(avc.outputs['out_freq'].value.shape, (n_avc,))
        self.assertEqual(avc.outputs['out_state'].value.shape, (n_avc, 4))
        # Initial correction is zero (x3 = x4 = 0 at init)
        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_avc))
        np.testing.assert_allclose(cpuArray(avc.outputs['out_freq'].value), np.zeros(n_avc))
        # out_state is only populated after the first trigger (diagnostic
        # output, not part of the constructor's initial condition)
        np.testing.assert_allclose(cpuArray(avc.outputs['out_state'].value), np.zeros((n_avc, 4)))

    @cpu_and_gpu
    def test_invalid_n_avc_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVC(simul_params=simul_params, n_avc=0,
                freq=10.0, gx=0.05, gomega=0.001, k=1.0, c=1.0,
                target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_mismatched_parameter_length_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVC(simul_params=simul_params, n_avc=3,
                freq=[10.0, 20.0], gx=0.05, gomega=0.001, k=1.0, c=1.0,
                target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        avc = AVC(simul_params=simul_params, n_avc=2, freq=[30.0, 60.0],
                  gx=0.05, gomega=0.001, k=1.0, c=1.0,
                  target_device_idx=target_device_idx)
        avc.sanity_check()

    @cpu_and_gpu
    def test_raises_on_missing_input(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        avc = AVC(simul_params=simul_params, n_avc=1, freq=30.0,
                  gx=0.05, gomega=0.001, k=1.0, c=1.0,
                  target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            avc.setup()

        measurement = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(measurement)
        avc.setup()  # does not raise anymore

    @cpu_and_gpu
    def test_first_trigger_outputs_zero_correction(self, target_device_idx, xp):
        """u[0] must be zero, since x3=x4=0 initially (Step 1 outputs the
        correction computed at the *previous* iteration)."""
        simul_params = SimulParams(time_step=0.001)
        n_avc = 2
        avc = AVC(simul_params=simul_params, n_avc=n_avc, freq=[25.0, 55.0],
                  gx=0.05, gomega=0.001, k=1.0, c=1.0,
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
    def test_check_ready_skips_stale_input(self, target_device_idx, xp):
        """The object must not trigger again if the input has not been
        refreshed since the last trigger."""
        simul_params = SimulParams(time_step=0.001)
        avc = AVC(simul_params=simul_params, n_avc=1, freq=30.0,
                  gx=0.05, gomega=0.001, k=1.0, c=1.0,
                  target_device_idx=target_device_idx)

        measurement = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(measurement)
        avc.setup()

        t1 = avc.seconds_to_t(0.001)
        measurement.generation_time = t1
        self.assertTrue(avc.check_ready(t1))
        avc.trigger()
        avc.post_trigger()

        # Advance time without refreshing the input
        t2 = avc.seconds_to_t(0.002)
        self.assertFalse(avc.check_ready(t2))

    @cpu_and_gpu
    def test_matches_matlab_reference_implementation(self, target_device_idx, xp):
        """Cross-check the vectorized object against a direct scalar
        translation of updateAVC.m / initAVC.m for several independent
        AVCs with different tuning parameters, over many iterations."""
        rng = np.random.RandomState(42)

        T = 0.002
        n_avc = 3
        freq = np.array([37.0, 82.5, 140.0])
        gx = np.array([0.05, 0.03, 0.08])
        gomega = np.array([0.0015, 0.0008, 0.002])
        k = np.array([1.0, 2.0, 0.5])
        c = np.array([1.0, 1.5, 0.7])
        n_over = np.array([1.0, 1.0, 1.0])
        x0 = np.array([1.0, 0.8, 1.2])
        x1 = np.array([0.0, 0.3, -0.4])
        alpha0 = np.array([0.0, 0.5, -1.0])

        n_iter = 300
        measurements = rng.randn(n_iter, n_avc) * 0.5

        # Independent scalar reference, one state dict per AVC
        ref_states = [
            dict(x=np.array([x0[i], x1[i], 0.0, 0.0]), omega=2 * np.pi * freq[i],
                 alpha=alpha0[i], alpha_c=0.0, T=T, gx=gx[i], gomega=gomega[i],
                 k=k[i], c=c[i], n=n_over[i], correction=0.0)
            for i in range(n_avc)
        ]
        ref_out = np.zeros((n_iter, n_avc))
        ref_state = np.zeros((n_iter, n_avc, 4))
        for it in range(n_iter):
            for i in range(n_avc):
                ref_out[it, i] = _reference_avc_step(ref_states[i], measurements[it, i])
                ref_state[it, i, :] = ref_states[i]['x']

        # SPECULA object
        simul_params = SimulParams(time_step=T, total_time=T * n_iter)
        avc = AVC(simul_params=simul_params, n_avc=n_avc,
                  freq=freq, gx=gx, gomega=gomega, k=k, c=c,
                  n_oversample=n_over, x0=x0, x1=x1, alpha0=alpha0,
                  target_device_idx=target_device_idx)

        meas = BaseValue(value=xp.zeros(n_avc), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(meas)
        avc.setup()

        spec_out = np.zeros((n_iter, n_avc))
        spec_state = np.zeros((n_iter, n_avc, 4))
        for it in range(n_iter):
            t = avc.seconds_to_t((it + 1) * T)
            meas.value[:] = xp.asarray(measurements[it, :])
            meas.generation_time = t
            avc.check_ready(t)
            avc.trigger()
            avc.post_trigger()
            spec_out[it, :] = cpuArray(avc.outputs['out_comm'].value)
            spec_state[it, :, :] = cpuArray(avc.outputs['out_state'].value)

        np.testing.assert_allclose(spec_out, ref_out, rtol=1e-6, atol=1e-9)

        # out_state must match the reference's internal x = (x1,x2,x3,x4)
        # trajectory (after this iteration's update), for every AVC
        # instance and every iteration -- this is the diagnostic output
        # used to investigate the plant-estimate (x1,x2) stability.
        np.testing.assert_allclose(spec_state, ref_state, rtol=1e-6, atol=1e-9)

    @cpu_and_gpu
    def test_frequency_tracking_converges_on_pure_tone(self, target_device_idx, xp):
        """Feeding a pure sinusoid at a known frequency should make the
        estimated frequency converge close to the true one."""
        T = 0.001
        true_freq = 50.0
        simul_params = SimulParams(time_step=T)

        # Note: AVC's frequency tracking is a slow gradient-type estimator
        # by design (see the discussion of gomega's role) - with the small
        # gains typical of a real deployment, closing a 2 Hz initial error
        # takes many seconds of simulated time. Here we use larger gx/gomega
        # purely to keep the test's iteration count reasonable; the point
        # is to check the *direction and mechanism* of convergence, not to
        # validate production tuning.
        avc = AVC(simul_params=simul_params, n_avc=1, freq=48.0,
                  gx=0.5, gomega=5.0, k=1.0, c=1.0,
                  target_device_idx=target_device_idx)

        meas = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(meas)
        avc.setup()

        n_iter = 20000
        for it in range(n_iter):
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
    def test_gomega_zero_disables_frequency_tracking(self, target_device_idx, xp):
        """With gomega=0, the frequency estimate must stay exactly constant
        regardless of the measurement."""
        T = 0.001
        simul_params = SimulParams(time_step=T)
        avc = AVC(simul_params=simul_params, n_avc=1, freq=42.0,
                  gx=0.05, gomega=0.0, k=1.0, c=1.0,
                  target_device_idx=target_device_idx)

        meas = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(meas)
        avc.setup()

        rng = np.random.RandomState(0)
        for it in range(200):
            t = avc.seconds_to_t((it + 1) * T)
            meas.value[:] = xp.asarray([rng.randn()])
            meas.generation_time = t
            avc.check_ready(t)
            avc.trigger()
            avc.post_trigger()

        final_freq = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(final_freq, 42.0, places=9)


if __name__ == '__main__':
    unittest.main()
