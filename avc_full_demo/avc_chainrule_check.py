#!/usr/bin/env python3
"""
Does restoring the dropped chain-rule term dW/dx (through theta) fix the
plant-estimate drift?

The paper minimises eps^2 = (W(theta(x))^T x - y)^2 but updates with only
x -= gx*W*e (eq. 10), treating W as independent of x. The exact gradient
of the error is  g = W + J^T x,  J[j,i] = dW_j/dx_i via theta(x).

We compare the two update directions in the idealised setting (no loop,
no delay, no noise, known fixed frequency, measurement = the algorithm's
own identity y = W^T x_true) and check (a) whether the plant estimate
converges to the truth and (b) the rank of the time-averaged update-
direction covariance.
"""
import numpy as np

PR, PI = 0.127, -0.596
T, f, theta_min = 1e-3, 47.0, 1e-2
A0 = 1.0                                   # small amplitude -> stable step size
gx = 0.05

def theta_of(x):
    x1, x2, x3, x4 = x
    d = x1*x1 + x2*x2 + theta_min
    return -(x1*x3 - x2*x4)/d, -(x1*x4 + x2*x3)/d

def W_of(x, alpha):
    thc, ths = theta_of(x)
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([thc*c + ths*s, ths*c - thc*s, c, s])

def jacobian(x, alpha, h=1e-7):
    """J[j,i] = dW_j/dx_i by finite differences (through theta(x))."""
    J = np.zeros((4, 4))
    for i in range(4):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        J[:, i] = (W_of(xp, alpha) - W_of(xm, alpha)) / (2*h)
    return J

def run(full_gradient, N=150000):
    x = np.array([PR, PI, 0.0, 0.0]) + np.array([0.05, -0.05, 0.0, 0.0])
    x_true = np.array([PR, PI, A0, 0.0])
    alpha = 0.0
    GG = np.zeros((4, 4))
    for k in range(N):
        W = W_of(x, alpha)
        y = W @ x_true
        e = W @ x - y
        if full_gradient:
            J = jacobian(x, alpha)
            g = W + J.T @ x                # exact gradient direction
        else:
            g = W                          # paper's pseudo-gradient
        x = x - gx*T*g*e
        alpha += 2*np.pi*f*T
        if alpha > np.pi: alpha -= 2*np.pi
        if k >= N-30000: GG += np.outer(g, g)
    GG /= 30000
    eig = np.linalg.eigvalsh(GG)[::-1]
    perr = np.hypot(x[0]-PR, x[1]-PI)
    return x, perr, eig

for full, name in [(False, "PAPER   (g = W, chain term DROPPED)"),
                   (True,  "EXACT   (g = W + J^T x, chain term KEPT)")]:
    x, perr, eig = run(full)
    rank = "4 (observable)" if eig[3] > 1e-4*eig[0] else "2 (UNobservable)"
    print(name)
    print(f"   final plant est (x1,x2) = ({x[0]:+.3f},{x[1]:+.3f})   true = ({PR:+.3f},{PI:+.3f})")
    print(f"   |plant error| = {perr:.4f}   (started at 0.0707)")
    print(f"   eig<g g^T> = [{eig[0]:.3f}, {eig[1]:.3f}, {eig[2]:.2e}, {eig[3]:.2e}]  -> rank {rank}")
    print()

print("Conclusion: keeping the exact chain-rule term changes WHICH 2-D")
print("subspace the update lives in, but g = c*v1 + s*v2 is still rank-2")
print("per cycle (v1,v2 are phase-independent), so the plant axes remain")
print("unobservable and the estimate still fails to converge. The dropped")
print("term is a real approximation, but it is NOT the cause of the drift.")
