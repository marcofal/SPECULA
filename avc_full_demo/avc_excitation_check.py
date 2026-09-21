#!/usr/bin/env python3
"""
Does adding excitation make the AVC plant estimate (x1,x2) observable, so
that online adaptation (adapt_plant=True) actually converges to the true
plant? Tests three excitation conditions, all in the idealized setting
(no loop/delay/noise, y = W^T x_true identity, known fixed frequency).

Key question behind the user's intuition: a single STEADY tone gives a
rank-2 regressor (plant unobservable). Does richer excitation fix it?
"""
import numpy as np

PR, PI = 0.127, -0.596          # true plant response at the tone frequency
T, f, gx, theta_min = 1e-3, 47.0, 0.05, 1e-2

def theta_of(x):
    x1, x2, x3, x4 = x
    d = x1*x1 + x2*x2 + theta_min
    return -(x1*x3 - x2*x4)/d, -(x1*x4 + x2*x3)/d

def W_of(alpha, thc, ths):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([thc*c + ths*s, ths*c - thc*s, c, s])

def run(lam_of_t, label, N=150000):
    """lam_of_t(t) -> (lam_c, lam_s), the (possibly time-varying) true
    disturbance quadratures. Plant PR,PI is always fixed."""
    x = np.array([0.8*PR, 0.8*PI, 0.0, 0.0])   # plant seeded a bit off; adapt ALL 4
    alpha = 0.0
    WWT = np.zeros((4, 4))
    for k in range(N):
        t = k*T
        lc, ls = lam_of_t(t)
        x_true = np.array([PR, PI, lc, ls])
        thc, ths = theta_of(x)
        W = W_of(alpha, thc, ths)
        y = W @ x_true
        x = x - gx*T*W*(W @ x - y)
        alpha += 2*np.pi*f*T
        if alpha > np.pi: alpha -= 2*np.pi
        if k >= N-40000: WWT += np.outer(W, W)
    WWT /= 40000
    eig = np.linalg.eigvalsh(WWT)[::-1]
    perr = np.hypot(x[0]-PR, x[1]-PI)
    print(f"{label}")
    print(f"    final plant est (x1,x2) = ({x[0]:+.3f},{x[1]:+.3f})  "
          f"true = ({PR:+.3f},{PI:+.3f})   |error| = {perr:.4f}")
    print(f"    eig<W W^T> = [{eig[0]:.3f}, {eig[1]:.3f}, {eig[2]:.2e}, {eig[3]:.2e}]"
          f"   rank {'4 (observable)' if eig[3] > 1e-4 else '2 (UNobservable)'}")
    print()

# small amplitude so the plant-update gain (~ (A/|P|)^2) stays in the
# stable range -- isolates observability from step-size instability
A0 = 1.0
# 1) single steady tone: constant amplitude
run(lambda t: (A0, 0.0),
    "1) STEADY single tone (constant amplitude):")
# 2) amplitude-modulated tone (a real vibration is rarely constant): the
#    disturbance envelope varies slowly -> the optimal theta must vary ->
#    the regressor sweeps a richer set of directions
run(lambda t: (A0*(1 + 0.6*np.sin(2*np.pi*2.0*t)), 0.0),
    "2) AMPLITUDE-MODULATED tone (envelope varies at 2 Hz):")
# 3) rotating-phase tone (amplitude constant but the cos/sin split drifts)
run(lambda t: (A0*np.cos(2*np.pi*1.0*t), A0*np.sin(2*np.pi*1.0*t)),
    "3) PHASE-ROTATING disturbance (quadrature split drifts):")

print("Interpretation (the hypothesis FAILED -- and here is why):")
print("  Making the DISTURBANCE time-varying does NOT make the plant")
print("  observable in this algorithm. In all three cases the two extra")
print("  eigenvalues of <W W^T> stay ~1e-5 (vs ~1 for the observable pair,")
print("  a 1e5 ratio = effectively still rank 2), and the plant estimate")
print("  never converges to the truth.")
print()
print("  The reason is structural: in W = [thc*c+ths*s, ths*c-thc*s, c, s],")
print("  the first two entries are ALWAYS a linear combination of the last")
print("  two (rows 1,2 = thc*row3 +/- ths*row4). So W is pinned to the plane")
print("  spanned by (cos,sin) no matter what theta does. Time-varying theta")
print("  tilts that plane only slightly, giving a negligible O(mod-depth^2)")
print("  observability that is useless on any practical timescale.")
print()
print("  Consequence: you cannot self-identify the plant by enriching the")
print("  DISTURBANCE. The plant must be CALIBRATED/MEASURED and frozen")
print("  (adapt_plant=False), which is what actually works. To track a")
print("  plant online you'd need a known probe injected into the COMMAND")
print("  (FxLMS-style secondary-path ID) -- a different algorithm than this.")
