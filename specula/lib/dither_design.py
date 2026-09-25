"""Suggested dither amplitude for the data-driven LQG (DdLqg, DataDrivenLqg).

The plant of each mode is identified by instrumental variables on the dither r (white,
std sigma_d), after an AR prefilter that whitens the turbulence. Each of the n_g taps
is then estimated with a standard deviation of about

    std(gamma_j) ~ sigma_e / (sigma_d sqrt(N))

where sigma_e is the innovation std of the turbulence of that mode (what the AR prefilter
leaves) and N the length of the identification window in frames. The modal gain
Gamma(1) = sum_j gamma_j has a relative error of about

    rel_std ~ sqrt(n_g) sigma_e / (sigma_d |Gamma(1)| sqrt(N)),

and the smallest dither that meets a target rel_std is

    sigma_d = sqrt(n_g) sigma_e / (rel_std |Gamma(1)| sqrt(N)).

The designs are checked for plant gains x0.5 .. x1.5, so rel_std <= 0.15 keeps an error of
three standard deviations inside the checked range. The dither is injected on the command,
so it adds about |Gamma(1)| sigma_d to the residual of each mode (sigma_d sqrt(m) over m
modes): a trade-off between identifiability and performance.

The expression is optimistic: it ignores the part of the command that the controller
correlates with the turbulence and the changes of the optical gain with the residual. On
the RAMA LEO runs (tip defect, 2 and 3 nm dither, 2 s windows) the measured scatter of
|Gamma(1)| was 1.3-1.4 times the prediction (median over modes 0-19) and up to 1.8 times on
modes 0-4, so suggest_dither_std multiplies it by margin = 1.5.

sigma_e is estimated from a stretch of closed-loop data (for example the warm-up with the
integrator): the pseudo open loop y - Gamma_guess(q) u with the nominal plant
-g q^-delay, and an AR(p) fit on it.
"""
import numpy as np

__all__ = ["innovation_std", "gain_rel_std", "suggest_dither_std"]


def innovation_std(y, u, gain_guess=1.0, delay=2, p=8):
    """Innovation std of the turbulence of each mode, from closed-loop y and u (T, m).

    u[k] is the command applied at frame k; the nominal plant is -gain_guess q^-delay.
    """
    y = np.atleast_2d(np.asarray(y, float).T).T
    u = np.atleast_2d(np.asarray(u, float).T).T
    T, m = y.shape
    g = np.broadcast_to(np.asarray(gain_guess, float), (m,))
    phi = y.copy()
    phi[delay:] += g * u[:-delay]                       # y - (-g q^-delay) u
    phi = phi[delay:] - phi[delay:].mean(axis=0)
    out = np.zeros(m)
    for i in range(m):
        x = phi[:, i]
        X = np.column_stack([x[p - l - 1:len(x) - l - 1] for l in range(p)])
        a, *_ = np.linalg.lstsq(X, x[p:], rcond=None)
        out[i] = np.std(x[p:] - X @ a)
    return out


def gain_rel_std(sigma_e, dither_std, window, gain=1.0, n_g=3):
    """Expected relative std of the identified |Gamma(1)| for a given dither."""
    return np.sqrt(n_g) * np.asarray(sigma_e, float) / (
        np.asarray(dither_std, float) * np.abs(np.asarray(gain, float)) * np.sqrt(window))


def suggest_dither_std(y, u, window, gain_guess=1.0, rel_std=0.15, n_g=3, p=8, delay=2,
                       margin=1.5, d_min=0.5, d_max=None):
    """Dither std per mode [units of y, u] for a relative error rel_std on |Gamma(1)|.

    y, u: closed-loop data (T, m); window: identification window [frames];
    gain_guess: expected |Gamma(1)| per mode (optical gain; 1 if unknown, which
    underestimates the dither where the optical gain is low). Returns (dither, sigma_e).
    """
    sigma_e = innovation_std(y, u, gain_guess, delay, p)
    d = margin * np.sqrt(n_g) * sigma_e / (rel_std * np.abs(np.broadcast_to(gain_guess, sigma_e.shape))
                                           * np.sqrt(window))
    d = np.maximum(d, d_min)
    if d_max is not None:
        d = np.minimum(d, d_max)
    return d, sigma_e
