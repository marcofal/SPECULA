# AVC full closed-loop demo (personal scratch folder, not part of the repo)

> **Just want to run/calibrate it?** See **[HOW_TO.md](HOW_TO.md)** — the
> practical recipe, the hyperparameter guide, and a symptom→cause
> debugging table. This README is the investigation log that produced it.
>
> **Want the theory + the original-vs-corrected comparison?** See
> **`avc_exact_gradient.pdf`** (source `avc_exact_gradient.tex`): the exact
> differentiation of the update law with a step-by-step Jacobian appendix,
> the two structural consequences (rank-2 unobservability; the
> disturbance-drive attenuation identity), and a controlled six-run
> comparison of the published pseudo-gradient against the corrected hybrid
> (figures `avc_gradient_comparison.pdf`,
> `avc_convergence_timescales.pdf`).

Self-contained SCAO/single-NGS closed-loop simulation with a manually
injected narrow-band mechanical vibration and AVC active in the loop.
Structured like `params_morfeo_single_ngs_mini.yml` (full pipeline in one
file: atmosphere, propagation, WFS chain, temporal control, DM, PSF, plus
a dedicated vibration source), but self-contained: it reuses the small,
already-validated SCAO calibration set bundled here in `calib/` (64px
pupil, 8x8 subaperture SH, 40 Zernike modes), instead of MORFEO's
production-scale calibration data (petal masks, IM/REC, M2 windshake time
history, etc.), which isn't available in this environment.

## Files

- `params_scao_avc_full_demo.yml` — full pipeline, AVC active as the
  *sole* controller on mode 0 (tip). **There is no main integrator in
  this file.** All other modes get zero command (uncorrected).
- `params_scao_vibration_baseline.yml` — the "free system": same
  atmosphere + vibration, but **no correction at all** (no integrator, no
  DM). `rec.out_modes` is the vibration + atmosphere as seen through the
  WFS, completely uncorrected. Use this as the "before" run.
- `calibrate_avc.py` — computes AVC's closed-loop-consistent initial
  state (`x0`/`x1`), using `specula.lib.avc_plant_model` (ported from
  ESO's `generateAVCInitConditions.m`). Its output is already pasted into
  `params_scao_avc_full_demo.yml`'s `avc:` block; re-run it and update the
  YAML whenever the loop parameters it depends on change.
- `analyze_results.py` — compares the latest baseline/AVC run pair and
  plots frequency convergence + residuals vs. time (see below).
- `calib/` — bundled calibration files (SH subaperture map, reconstruction
  matrix) copied from SPECULA's own test suite. Self-contained, no
  external downloads needed.
- `output/` — `DataStore` output, one timestamped subfolder per run.

**Why no integrator?** Earlier versions of this demo ran AVC *alongside*
an active main integrator on the same mode. That confounds two different
questions — "does AVC's own adaptive loop work" and "how does AVC
interact with a second, independently-adapting controller on the same
DOF" — and this test setup (a single processing object closing a loop
against a fixed physical simulation) isn't well suited to disentangling
two simultaneously active, independently adapting controllers anyway (the
integrator belongs in a separate, parallel module in a real deployment).
Removing it isolates the more basic question: with nothing else acting on
mode 0, does AVC's own correction reduce the vibration or not? See
"Final finding" below — the answer turned out to be no, and instrumenting
AVC's internal state shows why.

## Running

From this folder:

```python
import specula
specula.init(0, precision=1)
from specula.simul import Simul

Simul('params_scao_vibration_baseline.yml').run()   # "before": vibration, no AVC
Simul('params_scao_avc_full_demo.yml').run()          # "after": vibration + AVC
```

Each run creates a new `output/<timestamp>/` folder containing
`res_modes.fits` (all 40 reconstructed residual modes per frame) and,
AVC run only, `avc_freq.fits` (frequency estimate, Hz), `avc_comm.fits`
(AVC's correction command), and `avc_state.fits` (internal state
`[x1, x2, x3, x4]`, diagnostic). (The PSF/Strehl block is commented out
in both files to keep runs fast; uncomment it in both if you want SR
too.)

Then, to compare the two:

```bash
python analyze_results.py
```

This automatically picks the most recent baseline run and the most recent
AVC run in `output/`, prints a numeric summary (frequency lock error, RMS
residual on mode 0 and overall, PSD at the vibration frequency, and AVC's
internal-state statistics), and saves `avc_analysis.pdf` — five panels:
AVC frequency convergence vs. the true vibration frequency, mode 0 (tip)
residual vs. time, overall residual (RMS across all 40 modes) vs. time
(baseline overlaid with AVC, faint = raw signal, bold = 100 ms moving-RMS
trend), input-vs-residual PSD (narrowband rejection check), and AVC's
internal state `|x1+i x2|` (plant estimate) / `|x3+i x4|` (disturbance
estimate) vs. time.

## What this demo actually shows

The vibration is 150 nm / 47 Hz injected on tip (mode 0), independent of
atmosphere and of the loop's own correction.

**Frequency/phase tracking works correctly in closed loop**, in every
configuration tried: starting from a 40 Hz guess, AVC locks onto the true
47.0 Hz within a couple of seconds and holds there, despite photon/readout
noise and the loop's 2-frame delay.

**The residual (mode 0 and overall) does not improve with AVC active** —
it comes out slightly *worse* than the no-AVC baseline. The rest of this
section documents the investigation into why, and what fixed most (not
all) of the gap.

### First finding: an unbounded amplitude drift, and its cause

With the generic AVC defaults (`x0=1.0, x1=0.0`, i.e. "AVC's own
correction reaches the measurement with unit gain and zero phase lag"),
`gx` (the amplitude-state gain) above ~0.08 causes the correction
amplitude to grow without bound over a few seconds instead of settling —
it *adds* to the residual rather than cancelling it. Reducing the main
integrator's gain on the same mode doesn't fix this. Even the
conservative `gx = 0.03` used before only grows more slowly: a 10 s run at
full gain showed the correction's amplitude climbing the entire time
(1.6 nm at t=0-1s up to 27.6 nm at t=9-10s, still rising, essentially
uncorrelated with the actual residual, `corr ≈ 0.002`) while the residual
degraded from matching baseline to ~9% worse.

That "unit gain, zero phase" assumption is exactly what's wrong: once AVC
closes its own loop around a system with the main integrator's own 2-frame
delay and WFS/DAC sample-and-hold in between, "1 unit of AVC's own
correction" does *not* show up in the next measurement as "1 unit,
in phase" — it shows up scaled and phase-rotated by the loop's own closed-
loop transfer function. Seeding the adaptive filter with the wrong
gain/phase for that relationship is a classic cause of instability in
adaptive feedback cancellers (the same reason FxLMS-style adaptive filters
filter their reference by the estimated "secondary path" instead of using
it directly).

### The fix: `specula/lib/avc_plant_model.py` + `calibrate_avc.py`

This ports ESO's own calibration step (`generateAVCInitConditions.m`,
called from `computeAVCparamsCIAO.m`/`computeAVCparamsModeCIAO.m`): an
analytic model of the loop (WFS/DAC zero-order-hold, the RTC delay, the
integrator, and — since SPECULA's `DM` object has no internal dynamics —
an ideal-gain DM) gives the actual closed-loop transfer function from
"AVC's own command" to "measured residual", `P_avc(f)`. Its real/imaginary
parts at the target frequency become AVC's `x0`/`x1`, instead of the
generic `(1.0, 0.0)`. `calibrate_avc.py` computes this for the demo's
current loop parameters (`T=0.001s`, `int_gain=0.5`, `delay=2` frames,
freq guess `40 Hz`) — it prints `x0=0.373400, x1=0.490409`, quite different
in both magnitude (0.62 vs. 1.0) and phase (53° vs. 0°) from the generic
default, confirming the loop really does rotate/scale the correction
non-trivially at 47 Hz. Those values are already in `params_scao_avc_full_demo.yml`.
Covered by `test/test_avc_plant_model.py` (9 tests, including a
line-by-line reference translation check against the Matlab formula).

**Effect, measured (5 s runs, everything else unchanged):**

| Config | mode 0 residual vs. baseline | overall residual vs. baseline |
|---|---|---|
| generic `x0=1, x1=0` | +8.9 % worse | +8.6 % worse |
| calibrated `x0/x1` (current file) | **+2.9 % worse** | **+2.8 % worse** |
| calibrated `x0/x1` + `int_gain[0]` reduced 0.5→0.1 (recalibrated for the new gain) | +0.5 % worse | +0.5 % worse |

Each step roughly halves the gap. The amplitude drift is also visibly
slower and its bucket-to-bucket growth ratio is decelerating (rather than
reaccelerating, as it did with the generic seed at longer run times) —
consistent with a system that's now close to marginally stable rather than
cleanly unstable.

The plant-identification step is implemented, tested, and demonstrably
closed most of the gap in that (integrator + AVC) configuration,
confirming the diagnosis of *that* problem — but it never crossed over
into a net improvement, and a natural question remained: is the residual
instability caused by AVC fighting the integrator (two independently
adapting controllers on the same mode), or is it intrinsic to AVC's own
adaptive law regardless of what else is in the loop?

### Final finding: removing the integrator doesn't fix it — the instability is intrinsic to AVC's own estimator

To answer that cleanly, the main integrator was removed entirely from
both files. `params_scao_vibration_baseline.yml` is now a true "free
system" (vibration + atmosphere, zero correction of any kind), and
`params_scao_avc_full_demo.yml` gives AVC sole, uncontested control of
mode 0 — no competing controller, so the earlier confound is gone.
`calibrate_avc.py` was re-run for this simpler plant (`loop_gain=0`, one
frame of delay instead of an integrator's own delay), giving a new seed
(`x0=0.871704, x1=-0.479223`, magnitude 0.995 — much closer to the
generic unit-gain default than before, as expected once the integrator's
own contribution to the loop's transfer function is gone).

**Result: still no improvement.** A 5 s run (`gx=0.1`) gives:

| Metric | free system (baseline) | AVC-only | change |
|---|---|---|---|
| mode 0 residual RMS | 621.4 nm | 622.3 nm | +0.1 % (worse) |
| overall residual RMS | 105.7 nm | 105.8 nm | +0.1 % (worse) |
| PSD at 47 Hz | 3.84e5 nm²/Hz (input) | 2.78e5 nm²/Hz both | 0 dB vs. baseline |

AVC's frequency tracking still works (locks to 46.9 Hz from a 40 Hz
guess). But the PSD panel shows *baseline already sits ~1.4 dB below the
raw injected-vibration PSD* purely from WFS/detector transfer-function
rolloff and noise — and AVC's residual PSD at 47 Hz is identical to
baseline's, i.e. AVC contributes essentially zero narrowband rejection of
its own.

The internal-state diagnostic (`avc.out_state`, new: `x1..x4` exported
per frame) shows why directly: the plant-response estimate
`|x1 + i x2|` — which should converge to a fixed value if AVC has
correctly identified how its own correction propagates back into the
measurement — instead **swings between 0.20 and 46** over the run,
never settling. Because the correction's phase/gain
(`hatTheta = -(x3+i x4)/conj(x1+i x2)`) is a *ratio*, and the denominator
keeps wandering close to zero, the correction it computes is not tracking
a stable target. Consistently, the disturbance-quadrature numerator
`|x3 + i x4|` grows monotonically the entire run (68 → 208, never
plateauing) — a direct symptom of the prediction error never shrinking,
which is exactly what happens when the ratio estimator that's supposed to
be converging is instead oscillating.

**Bottom line:** removing the integrator ruled out "AVC fights a second
controller" as the explanation. The real cause is structural: AVC's
adaptation law estimates the correction gain/phase as a *ratio* of two
independently gradient-updated quantities, with a constant (non-decaying)
step size (`gx`) and no filtering of the regressor by the estimated plant
response (unlike FxLMS-style adaptive cancellers, which filter the
reference through the identified secondary path specifically to keep the
adaptation well-conditioned). When the denominator of that ratio wanders
near zero — which nothing in the update law prevents, `theta_min_energy`
is only a late safety clamp, active just 0.1% of this run — the estimated
plant response, and hence the correction it computes, become
disconnected from the physical relationship they're supposed to track.
This is a classic failure mode of division-based/ratio adaptive
estimators, not an artifact of the SPECULA loop or of this test harness.
A fix would need to live in the AVC algorithm itself (e.g. a
decaying/normalized step size, or regularizing the denominator away from
zero more proactively than a hard floor), which is out of scope for this
demo.

### Does a smaller step size (`gx`) fix it? No — it just relocates the instability

Tried `gx = 0.1, 0.01, 0.001` (all other params, including the
integrator-free architecture and calibrated seed, unchanged; 5 s runs):

| `gx` | mode 0 residual vs. baseline | overall residual vs. baseline | PSD at 47 Hz vs. baseline | plant estimate `\|x1+i x2\|` | clamp active |
|---|---|---|---|---|---|
| 0.1 | +0.1 % | +0.1 % | -0.0 dB | mean 11.3 (range 0.20–46) | 0.1 % |
| 0.01 | +0.1 % | +0.0 % | -0.0 dB | mean 0.30 (range 0.002–1.77) | 55.8 % |
| 0.001 | -0.0 % | -0.0 % | +0.0 dB | mean 0.19 (range 0.08–0.72) | 49.7 % |

None of the three produces a net improvement — every difference is within
run-to-run noise. What changes is *where* the instability settles, not
whether it goes away: at `gx=0.1` the plant estimate swings over nearly
three orders of magnitude; at `gx=0.001` it stops swinging as widely but
instead converges onto a limit cycle sitting right at the
`theta_min_energy` safety-clamp floor (visible in the state plot as a
flat line pinned at 0.1 from ~2.5 s onward) — i.e. the clamp, not the
adaptation law, is what's holding it up from collapsing to zero. The
disturbance-estimate numerator still grows monotonically the whole run at
every `gx` tried (just proportionally smaller/slower for smaller `gx`).
At `gx=0.001` the AVC residual PSD also picks up extra broadband power in
the 20–40 Hz band relative to baseline — a small non-cancelling
correction is being injected without addressing the target peak at all.

This is the expected signature for a constant (non-decaying) step size in
a division-based estimator: shrinking `gx` shrinks the size of the
persistent "jitter" around wherever the estimator settles (consistent
with the classical adaptive-filtering result that excess error scales
with step size), but it doesn't add a restoring force that pulls
`x1, x2` toward a well-conditioned, non-zero, physically-correct plant
estimate — because nothing in the update law depends on how far the
denominator is from zero except the after-the-fact clamp. Confirms the
diagnosis above: this needs an algorithmic fix (normalization,
regularization, or a differently-structured estimator), not a tuning
fix.

### Checking against the original paper, and a soft-floor experiment

A line-by-line comparison of `avc.py`'s `trigger_code` against Muradore,
Pettazzi, Fedrigo & Clare, *"On the rejection of vibrations in Adaptive
Optics Systems"* (SPIE 8447-38) found no discrepancy in the core adaptive
law: the regressor (their eq. 9), the output equation (eq. 11), the state
update (eq. 10/15) and the frequency/phase update (eq. 15) all match the
paper exactly, including the one-frame delay bookkeeping (the regressor
built at the start of iteration `k+1` uses exactly the state/phase that
produced the correction actually applied at iteration `k`).

What is *not* in the paper is `theta_min_energy`: a safeguard carried
over from the Matlab reference (`updateAVC.m`) that hard-resets
`hatTheta` to `(1, 0)` whenever `x1²+x2² < theta_min_energy`, instead of
smoothly regularizing. Since `hatTheta ∝ 1/(x1²+x2²)` is already, by the
paper's own eq. 11, prone to arbitrarily large gain as the denominator
shrinks, a discontinuous switch sitting right on a boundary the state
keeps crossing is a natural trigger for limit-cycle chattering — matching
what the state plots above show.

`avc.py` now has an opt-in `soft_clamp: true` parameter that replaces the
hard branch with an unconditional soft floor,
`safe_denom = x1²+x2² + theta_min_energy` (default remains the exact
Matlab-parity hard clamp; `test_matches_matlab_reference_implementation`
still validates against it). Re-running `gx=0.001` with `soft_clamp: true`
removes the chattering completely — but reveals the real underlying
dynamics instead of masking them: `|x1+i x2|` no longer oscillates near
the clamp boundary, it decays **smoothly and monotonically straight
through zero**, dropping over 21 orders of magnitude within about 2
seconds of simulated time (a clean, unbroken straight line on a log
plot). The chattering seen with the hard clamp wasn't hiding a
recoverable estimate underneath — it was literally the clamp repeatedly
catching this same collapse and resetting it to `(1, 0)`, only for it to
immediately start falling again. Residual/PSD numbers are, unsurprisingly,
unchanged (~0% vs. baseline): a plant estimate that has collapsed to zero
can't drive a meaningful correction.

**Conclusion:** the origin is a genuine *attracting* fixed point of the
`(x1, x2)` sub-dynamics under this loop's actual error signal — not a
clamp artifact, not a chattering side-effect, and not something a
different `gx` fixes (a smaller `gx` only slows the same collapse; a
larger one only makes the pre-collapse jitter noisier). This is a real
mismatch between the algorithm's implicit assumptions (the paper's
stability proof, credited to Pigg & Bodson but not reproduced in this
SPIE paper, presumably requires conditions — e.g. a persistently
informative regressor separating the `x1,x2` and `x3,x4` directions, or a
plant response bounded away from zero — that don't hold robustly for this
particular loop/signal combination) and this specific application, rather
than a bug in this port or a tuning issue.

### Ruled out: is a near-allpass ("pure delay") plant the culprit?

One remaining hypothesis: with no integrator and an ideal, dynamics-free
DM, the loop's only element between "AVC command" and "measurement" is a
single-frame zero-order-hold delay — at 47 Hz and `T=1ms` that's close to
unit magnitude and only a small phase shift (calibrated
`|P|≈0.995, phase≈-28.8°`), i.e. almost an allpass. An estimation problem
where the true target sits that close to the trivial "no dynamics at all"
answer could plausibly be poorly conditioned for a ratio-based estimator.

Tested by replacing the ideal DM with a genuine first-order actuator
low-pass (`lowpass:` block, SPECULA `LowPassFilter(n_ord=1)`, inserted
between `avc_inserter` and `dm`; `specula/lib/avc_plant_model.py` gained a
matching `lpf_fc` parameter — an exact analytic port of
`IirFilterData.lpf_from_fc(n_ord=1)`'s difference equation, verified
against `IirFilter.trigger_code` line by line). At `cutoff_freq=80Hz`
(~1.7x the vibration frequency) the plant at 47 Hz becomes genuinely
dynamic: `|P|≈0.61`, phase `≈-78°` — clearly distinguishable from a
trivial pass-through, recalibrated `x0=0.233956, x1=-0.628879`.

**Result: no change.** At `gx=0.1`: mode 0 residual -0.1%, overall -0.1%,
PSD at 47 Hz +0.0 dB — all within noise, identical to the pure-delay
case. At `gx=0.001` with `soft_clamp: true`: `|x1+i x2|` shows the *same*
clean, monotonic, ~21-decade exponential collapse straight through zero,
starting slightly earlier (~2.0s vs. ~2.5s) but otherwise indistinguishable
from the pure-delay run.

**This rules out the near-allpass/pure-delay hypothesis.** The collapse
to zero happens identically whether the plant is a near-unit-gain,
near-zero-phase delay or a genuinely dynamic low-pass with substantial
attenuation and phase lag. Whatever pulls `x1,x2` to the origin isn't
about how "trivial" or well-separated-from-unity the true `P_R,P_I` is —
it's a property of the coupled adaptation law itself (the same mechanism
described above: `w1,w2` depend on `x1,x2` through the `hatTheta` ratio,
feeding back into `x1,x2`'s own update), independent of the specific
plant being identified.

### Root cause (proven): the plant estimate is *unobservable* from a single sinusoid

`avc_observability_check.py` isolates the plant-ID sub-problem from
everything else — no AO loop, no delay, no noise, frequency known and
fixed, measurement generated by the algorithm's *own* identity
`y = W^T x_true` (eq. 8, zero model mismatch) — the best possible case.
It settles the question definitively.

The state is `x = [P_R, P_I, λ_c, λ_s]` (2 plant + 2 disturbance
parameters). The regressor (eq. 9) is
`W = [θc·c+θs·s, θs·c−θc·s, c, s]` with `c=cos α, s=sin α`. Every entry is
a linear combination of just `c` and `s`, so — writing `W = M·[c; s]` —
the whole 4-vector lives in the 2-D column space of a fixed `4×2` matrix
`M`. Averaging over a cycle, `⟨W Wᵀ⟩ = ½ M Mᵀ`, and since `MᵀM = (|θ|²+1)I₂`,
this has **exactly two nonzero eigenvalues and two zeros**. Experiment A
confirms it numerically to machine precision: eigenvalues
`(1.777, 1.777, 2e-14, 2e-14)`, and the gradient drives the *observable*
part of the estimate error to `~1e-13` while the **unobservable part stays
frozen at its initial value forever** — it is never corrected, by
construction.

Crucially, the two null (unobservable) directions are
`n_A = [1, 0, −θc, −θs]` and `n_B = [0, 1, −θs, θc]` — each carries unit
weight on a *plant* axis (`x1` or `x2`). In words: **you can change the
plant estimate `(x1,x2)` and compensate with the disturbance estimate
`(x3,x4)` without changing the prediction error at all.** A single
sinusoid simply does not carry enough information to separate "how the
plant scales/rotates my correction" from "how big the disturbance is" —
both act on the same 2-D `(cos, sin)` measurement. This is a textbook
persistent-excitation / observability deficiency, not a bug: the SPECULA
code faithfully matches eqs. 9–11 and 15–16 line by line (re-verified this
pass).

Two immediate consequences, both matching what the full sim shows:

- **Frequency tracking still works** because the PLL (eq. 15, `ω`/`α`
  updates) is driven by a *different* signal — the quadrature phase error
  `sin(α)·e` — which does not require identifying the plant. That is why
  frequency locks cleanly while plant ID fails.
- **The estimate doesn't just drift, it runs away.** Because `θ = f(x)`
  (eq. 11) is recomputed from the running estimate and contains
  `1/(x1²+x2²)`, the free drift along the unobservable directions feeds
  back through `θ → W → x`-update. Experiment B seeds `x1,x2` at the
  *true* plant value and the estimate still diverges (plant magnitude
  0.55 → thousands, disturbance growing monotonically), with no noise, no
  delay, exact frequency — exactly the collapse/growth seen in the full
  loop.

### The fix this points to

Since `(x1,x2)` is unobservable anyway, adapting it online can only hurt.
Experiment C freezes the plant estimate at its calibrated value (which we
already compute offline via `calibrate_avc.py`) and adapts **only** the
disturbance components `x3,x4`. Result: the disturbance estimate converges
to the truth exactly (`(1000.000, −0.000)` vs. true `(1000, 0)`) and the
prediction error nulls to `~2e-6`. Freezing the unobservable axes removes
the drift entirely and recovers a working canceller.

This is also, in hindsight, *why the original paper works*: it uses
"approximate values of the closed loop plant" as initial conditions and
relies on fast error-nulling to freeze `x` near that (correct) seed before
the unobservable drift matters — the algorithm effectively *retains* a
good plant calibration rather than truly identifying the plant online. The
paper's own §3.2 even removes the MPLL's amplitude estimation to "remove
the principal source of instability" — the very same over-parameterization
(estimating the disturbance amplitude in two places at once) that the
`x1,x2 ↔ x3,x4` degeneracy is another instance of.

Concretely, the recommended change to `avc.py` is an opt-in "freeze plant"
mode: keep `x1,x2` fixed at the `x0,x1` calibration and update only
`x3,x4` (and `ω,α`). This is a small, well-motivated addition rather than
a reworking of the adaptation law.

### Resolution: AVC finally cancels (−9.5% on mode 0)

Two fixes together make the AVC work in this loop.

**Fix 1 — freeze the unobservable plant estimate** (`adapt_plant: false`,
new opt-in flag on `avc.py`, default `true` = original behaviour). With it
on, `x1,x2` stay pinned at their calibration (the state plot shows a
perfectly flat red line at `|x1+i x2| = 0.555`) instead of drifting/
collapsing. Only the observable disturbance quadrature `x3,x4` (and the
frequency/phase) adapt. This removes the drift entirely — but on its own
it made the residual *worse* (+8.5%), which exposed a second, independent
bug.

**Fix 2 — correct the plant-calibration sign.** With the plant frozen, a
wrong frozen value can only ever mis-phase the correction, so the sign/
phase of `x0,x1` now matters absolutely. `params_measure_plant.yml`
measures the true command→residual transfer directly from a SPECULA probe
run (inject a known 47 Hz sinusoid into the DM command path with the
disturbance off, lock-in on the response): it comes out **|P| = 0.555,
phase = +119°**. The analytic `calibrate_avc.py` predicted **|P| = 0.61,
phase = −78°** — essentially **180° apart**. The generic/ESO-MATLAB plant
model (which `avc_plant_model.py` faithfully reproduces, verified in
`test_avc_plant_model.py`) does not capture the sign of SPECULA's specific
DM/loop wiring: a DM correction here enters the residual with the opposite
sign to what that model assumes. Every earlier calibration was therefore
~180° mis-phased, so the "correction" was adding to the vibration — which
is exactly why *nothing* helped for so many iterations, independent of
gx, clamp, or plant conditioning. (Confirmed quantitatively: negating the
zero-extra-delay analytic value gives phase +118.9° vs. the measured
+119.0°, an essentially exact match.)

**With both fixes** (`adapt_plant: false`, `x0,x1` = measured
`−0.268871, +0.485866`, `gx=0.1`, 5 s, frequency guess a wrong 40 Hz):

| Metric | free system | AVC | change |
|---|---|---|---|
| mode 0 residual RMS | 621.4 nm | 562.1 nm | **−9.5 %** |
| overall residual RMS | 105.7 nm | 97.1 nm | **−8.2 %** |
| PSD at 47 Hz vs. baseline | — | — | **−0.5 dB** (a real notch) |

The AVC residual now sits *below* baseline at the vibration frequency for
the first time, the plant estimate is rock-stable, and the disturbance
estimate bends over toward a plateau (converging) instead of running away.
Cancellation only switches on once the frequency locks (~2.5 s from the
40 Hz guess), so this 5 s figure understates the steady-state ceiling —
a longer run, a closer frequency seed, or a larger `gx` all improve it
further.

**Takeaways for anyone re-using this AVC in SPECULA:**

1. Run it with `adapt_plant: false` — online plant adaptation is
   unobservable from a single tone and destabilises the loop.
2. Calibrate `x0,x1` by *measuring* the command→residual transfer from a
   probe run (`params_measure_plant.yml`), not from the generic analytic
   model, whose sign convention does not match SPECULA's loop.

### Pushing the rejection deeper, and a negative result on "add excitation"

With the loop confirmed working, the residual `-9.5 %` above was throttled
by two things unrelated to the plant: the frequency needs ~2.5 s to lock
(half of a 5 s run), and `gx=0.1` lets the disturbance integrator wind up
only slowly. Freezing the plant at the correct measured value lets `gx` be
pushed hard. Seeding the frequency near truth (in practice from
`find_vib_peaks`) and sweeping `gx` (5 s runs, steady-state window):

| `gx` | mode 0 RMS vs. baseline | PSD notch at 47 Hz | disturbance estimate |
|---|---|---|---|
| 0.1 | −9.5 % | −0.5 dB | still ramping |
| 0.5 | −37.9 % | −4.4 dB | still ramping |
| **1.0** | **−42.8 %** | **−9.4 dB** | converging → ~960 (≈ 1000 nm tone) |
| 2.0 | −4.8 % (worse RMS!) | −8.4 dB | overshoots to 1439, under-damped |

`gx=1.0` is the sweet spot: near-complete tonal cancellation (`|x3+i x4|`
approaches the true ~1000 nm amplitude), a real `−9.4 dB` notch, and
`−42.8 %` mode-0 RMS. `gx=2.0` deepens the *notch* but over-corrects and
injects broadband power, so the total RMS is worse — the classic
step-size / bandwidth trade-off. Longer runs, a closer frequency seed, or
a slightly decaying `gx` would push the notch deeper still.

**Would adding a second excitation let the plant self-identify?** Tempting
idea (make the state observable, then trust `adapt_plant: true`), but it
**does not work for this algorithm**, and `avc_excitation_check.py` shows
why. Enriching the *disturbance* — amplitude-modulating the tone, or
rotating its phase — leaves the two extra eigenvalues of `⟨W Wᵀ⟩` at
`~1e-5` (vs. `~1` for the observable pair), i.e. still effectively rank 2,
and the plant estimate never converges. The reason is structural: in
`W = [θc·c+θs·s, θs·c−θc·s, c, s]` the first two entries are *always* a
linear combination of the last two (`row1,2 = θc·row3 ± θs·row4`), so `W`
is pinned to the 2-D plane spanned by `(cos, sin)` no matter what `θ`
does; a time-varying `θ` tilts that plane only by an `O(mod-depth²)`
amount — useless in practice. So you cannot make the plant observable by
enriching the *disturbance*. Observability has to come from the *command*
side — a known probe injected into `u` and correlated against the
response (exactly what `params_measure_plant.yml` does offline, and what
FxLMS-style adaptive cancellers do online with a pilot tone). That is a
different algorithm than this AVC; for this one, **measure-and-freeze is
the right architecture**, and it already gives real rejection.

### Frequency tracking gets captured by the atmosphere at low vibration SNR

Symptom: with the vibration amplitude lowered from 1000 nm to **30 nm**
(and `seeing` 1 → 0.6), the frequency estimate stopped converging — seeded
at the true 47 Hz it slid *down* to ~39–42 Hz instead of staying put.

The tone is not buried (it is still a sharp peak, ~1500× its local
background). The problem is the *low-frequency* end: mode-0 residual PSD
at 5 Hz is `2.35e2` versus `2.49e2` at 47 Hz — the atmospheric tip power
now **rivals the tone**, where at 1000 nm the tone dominated by ~1000×.
The PLL (eq. 15) is a plain gradient frequency tracker with nothing
anchoring it near its initial guess, so it walks downhill toward the
dominant low-frequency atmospheric power.

`gomega` controls how fast it walks, and the sweep is unambiguous — a
*faster* PLL is captured *harder*:

| `gomega` | frequency behaviour (seeded at 47 Hz) | mode 0 RMS | notch at 47 Hz |
|---|---|---|---|
| 10.0 | collapses to **6.6 Hz** (onto the ~5 Hz atmospheric peak) | — | — |
| 0.5 | drifts to 39–42 Hz | 99.0 nm | none |
| **0.05** | **stays locked at 47.06 Hz** | 78.6 nm | −4.7 dB |
| 0 (tracking off) | held at 47.00 Hz | 78.0 nm | **−7.2 dB** |

So the cancellation machinery is perfectly healthy at 30 nm — with the
frequency held fixed it gives a −7.2 dB notch and takes mode-0 RMS from
99 → 78 nm. *Only* the frequency tracker was broken.

**The rule this establishes:** `gomega` must be re-tuned whenever the
vibration SNR changes. It is not a dimensionless constant — the frequency
update `2·T·gomega·sin(α)·e` is driven by whatever dominates `e`, so a
value tuned for a strong tone becomes far too aggressive (i.e. noise-
driven) for a weak one. Three options, in increasing order of robustness:

1. **Lower `gomega`** proportionally to the tone's SNR (0.5 → 0.05 here).
2. **Set `gomega: 0`** and hold the frequency from `find_vib_peaks` —
   best notch here, and legitimate when the vibration frequency is stable
   (the paper measures ±0.1 Hz on real NACO data).
3. **Band-pass the AVC input** around the vibration so the PLL never sees
   the atmospheric low-frequency power at all. This is the principled fix
   and would restore robust tracking at a high `gomega`; it needs a filter
   object between `mode_slicer` and `avc` (not yet wired in).

Unrelated regression found while investigating: `adapt_plant` had been set
back to `true` while its comment still read "freeze", reintroducing the
plant runaway (`|x1+i x2|` wandering 0.013 → 4.31). Restored to `false`.

---

## The realistic closed-loop demo (current configuration)

The demo now runs the architecture of Fig. 1 of the paper: **the main
integrator and AVC in parallel**, the integrator rejecting the atmosphere
and AVC handling the narrow-band peak. `params_scao_vibration_baseline.yml`
is the same closed loop *without* AVC, so the difference between the two
files is exactly and only the AVC module.

### Why 47 Hz is the right place to put the vibration

Measured rejection transfer of this loop (integrator gain 0.5, delay 2
frames + 1 propagation frame, 80 Hz actuator low-pass, 1 kHz):

| freq | 1 Hz | 5 Hz | 10 Hz | 20 Hz | **47 Hz** | 100 Hz |
|---|---|---|---|---|---|---|
| \|S\| | −38 dB | −24 dB | −17 dB | −9 dB | **+19 dB** | ≈0 dB |

The integrator crushes the low frequencies but **amplifies by ~19 dB at
47 Hz** — the vibration sits right on the waterbed peak (~45 Hz). A 50 nm
injected tone becomes a **449 nm** residual. This is precisely the
situation the paper describes for the 48 Hz NACO vibration, and it is what
makes AVC worth having: the main controller actively makes this
disturbance ~10x worse, and cannot be retuned to fix it without giving up
low-frequency performance.

The analytic model agrees with the measurement here (predicted +12.4 dB,
measured +19 dB at 47 Hz; peak location 45 Hz), which is a useful
cross-check of `avc_plant_model.py` — its *magnitudes* are trustworthy
even though its *sign convention* is not (see above).

### Results

4 s runs, vibration 50 nm @ 47 Hz, AVC frequency guess deliberately wrong
(45 Hz), `gx=3.0`, `gomega=0.2`, `adapt_plant: false`, `x0,x1` measured
closed-loop:

| Metric | baseline (integrator only) | + AVC | change |
|---|---|---|---|
| mode 0 (tip) residual RMS | 315.8 nm | **26.6 nm** | **−91.6 %** |
| overall 40-mode residual RMS | 50.2 nm | **6.5 nm** | **−87.0 %** |
| 47 Hz tone amplitude | 448.8 nm | **13 nm** | **−31 dB** |
| PSD at 47 Hz | +19.1 dB vs. input | +9.6 dB | **−9.4 dB** |

AVC removes essentially the entire tone — the 47 Hz residual (13 nm) ends
up *below* what it would be with no correction at all (≈44 nm open loop),
i.e. AVC undoes the integrator's amplification and then some. Frequency
tracking now works cleanly too: locked from the wrong 45 Hz guess to
46.94 Hz with 0.033 Hz jitter, because with the integrator suppressing the
low-frequency atmosphere the tone is the dominant feature of the residual
spectrum and the PLL cannot be captured (contrast the failure above).

### Two calibration traps worth remembering

1. **The plant must be re-measured with the loop closed.** Closing the
   integrator changes AVC's plant from `|P|=0.555 ∠+119°` to
   `|P|=7.223 ∠+169.5°` — a factor 13 in gain and 50° in phase, because
   AVC's command now propagates through the closed-loop sensitivity.
   `params_measure_plant.yml` contains the same integrator for this reason.
2. **Keep the probe (and the vibration) inside the WFS linear range.**
   Because the loop amplifies ~10x at 47 Hz, a 500 nm probe drives the
   residual to ~1300 nm RMS, the Shack-Hartmann saturates, and the
   measured transfer is garbage (it read `|P|=3.64` instead of `7.22`, and
   the apparent value drifted with probe amplitude — the tell-tale sign).
   A 30 nm probe giving a ~200 nm response is both linear and well above
   the noise. The same applies to the injected vibration: 500 nm in closed
   loop saturates the WFS and invalidates the whole run.

---

## Verdict: is the plant drift the algorithm's, or ours?

Worth settling explicitly, since the whole investigation hinged on it:
**is the plant-estimate drift an expected property of the published
algorithm, or a bug in this port / in SPECULA?**

### It is the algorithm's. Four independent lines of evidence

1. **Line-by-line agreement** with eqs. 9, 10/15, 11 and 16 of the paper.
2. **`test_matches_matlab_reference_implementation` passes** — the port
   reproduces an independent implementation of the same algorithm to
   `rtol=1e-6`, state vector included.
3. **The divergence reproduces with no SPECULA at all.**
   `avc_observability_check.py` implements only the paper's equations in
   plain NumPy — no AO loop, no delay, no noise, exact known frequency,
   and the measurement generated from the algorithm's *own* identity
   `y = W^T x_true` (zero model mismatch). The rank-2 deficiency shows up
   to machine precision (eigenvalues of `<W W^T>`:
   `1.777, 1.777, 2e-14, 2e-14`) and the estimate still fails to converge.
   Nothing in that harness is ours to get wrong.
4. **The definitive in-loop test.** With the *correct* measured seed
   (`|P|=7.223 ∠+169.5°`), a *stable* plant-update step size
   (`c=0.01`, see below), and a loop that reaches −31 dB when the plant is
   frozen, turning `adapt_plant: true` back on still lets the estimate
   wander over the whole complex plane:

   | | seed | t=1 s | t=2 s | t=3 s | end |
   |---|---|---|---|---|---|
   | `\|x1+i x2\|` | 7.22 | 5.97 | 1.45 | 16.59 | 3.85 |
   | phase | +169.5° | +86.7° | −123.1° | +99.2° | +124.4° |

   and performance collapses with it (47 Hz residual 417 nm vs. 13 nm
   frozen; mode-0 RMS 361 nm vs. 27 nm). So it is neither the bad seed
   nor a step-size instability.

The mechanism is forced by the paper's own eqs. 8–9: the state is 4-D, but
in `W = [θc·c+θs·s, θs·c−θc·s, c, s]` the first two entries are *always* a
linear combination of the last two (`row1,2 = θc·row3 ± θs·row4`), so
`<W W^T>` is rank 2 and the two null directions are exactly the ones
carrying the plant axes.

*Fairness note:* unobservable is not the same as divergent. Drift along a
null space is "free", not necessarily explosive. What converts it into
divergence is the feedback `θ = f(x)` containing `1/(x1²+x2²)`: the free
drift changes the correction, which changes the error, which drives more
drift.

### Traces of it in the paper

The authors do not discuss this failure, but three things point at it:

1. **§3.1, the initialisation condition** — *"The only condition is that
   the first and the second component of x(0) cannot be both equal to
   zero."* They know `x1=x2=0` is singular (it is the denominator of
   eq. 11), but they guard only the *initial* value and say nothing about
   the estimate *staying* away from zero — which is precisely the failure
   mode.
2. **§3.2, an explicit admission of this class of instability** — they
   drop the MPLL's amplitude estimation because *"both the AVC and the
   MPLL estimate the amplitude of the vibration … the transients of the
   two dynamical systems are badly affected by this coupling"*, and say
   this *"removed the principal source of instability in the AVC."* That
   is a redundant-parameterisation instability: two subsystems estimating
   the same quantity. The `x1,x2 ↔ x3,x4` degeneracy is the same disease
   one level down, *inside* the AVC state itself.
3. **§4.3, how they actually run it** — *"using as initial conditions
   approximate values of the closed loop plant."* They seed with a good
   plant estimate rather than identifying it from scratch. Their `gx` also
   varies from 1 to 30 across test cases, consistent with the
   amplitude-dependent gain scaling below. And the stability proof is
   explicitly not reproduced (*"even though it is not reported here"*),
   deferred to Pigg & Bodson — so the paper never demonstrates that the
   identifiability conditions hold in their setting.

### The timescale conflict, and what `c` is really for

The plant update carries an extra factor `(|θ|²+1)` that the disturbance
update does not, and `θ ≈ |x3,x4|/|P|` grows with the disturbance. In the
working demo `|θ| ≈ 82`, so `(|θ|²+1) ≈ 6673`:

| requirement | implied `gx` |
|---|---|
| disturbance loop converges in ~2 s | `gx ≈ 3` |
| plant update stays stable (gain < 2, `c=1`) | `gx < 0.3` |

A single `gx` cannot satisfy both — they are ~20x apart here, and the gap
widens with disturbance amplitude. Tellingly, **`c` — the tuning constant
that scales *only* the `x1,x2` update, exactly the knob needed to decouple
these two timescales — exists in ESO's `updateAVC.m` but appears nowhere
in the paper.** That looks like practitioners hitting this in the lab and
patching it downstream of publication. (`c=0.01` does make the plant
update numerically stable at `gx=3` — it just does not make the estimate
*correct*, per the test above.)

### What genuinely was ours

Worth separating, because the two were conflated for a long stretch of
this investigation:

- **The 180° sign error** in the analytic calibration was ours
  (`avc_plant_model.py` vs. SPECULA's wiring). That is what made AVC
  *amplify* rather than cancel, and it is fixed by measuring the plant
  instead (§ "Two calibration traps"). Unrelated to the drift.
- **Not ruled out:** the code builds the regressor from the *current* `θ`,
  while the measurement reflects a correction applied 1–2 frames earlier,
  and the paper's derivation assumes no delay. This cannot explain the
  drift (which reproduces at zero delay in pure NumPy), but it is a real
  deviation between the paper's assumptions and any delayed
  implementation. It would be the first thing to examine if online plant
  tracking were ever wanted.

### A dropped chain-rule term in the update (the gradient is only a pseudo-gradient)

The paper sets up `eps^2 = (W(theta)^T x - y)^2` and does gradient descent
(eq. 10). But `W` depends on `theta`, and `theta` depends on `x` (eq. 11),
so the *exact* derivative of the error is

```
de/dx_i = W_i + sum_j (dW_j/dx_i) x_j       (chain rule through theta(x))
```

The paper's update keeps only `W_i` and **drops the second term**. This is
the standard equation-error / pseudo-gradient approximation (inherited from
Pigg & Bodson), but it is a genuine approximation, not the true gradient.
`avc_chainrule_check.py` restores the exact term and compares:

| update | final plant error | eig `<g g^T>` | plant identified? |
|---|---|---|---|
| paper (chain term dropped) | 0.265 (drifts *up* from 0.071) | [1.73, 1.73, ~0, ~0] | no — actively wanders |
| exact (chain term kept)     | 0.063 (~ stays put)          | [~0, ~0, 4e-8, ~0]    | no — settles at wrong value |

Two separate effects, and it is worth keeping them apart:

- **Identifiability: unchanged.** Both directions are rank 2. Analytically
  `g_exact = W + J^T x = c*v1 + s*v2` with `v1, v2` phase-independent, so —
  exactly like `W` — it is trapped in a 2-D plane per cycle and the plant
  axes stay unobservable. The exact gradient merely converges to a
  *different* point on the same 2-D zero-error manifold (0.063, not 0),
  fixed by the initial condition. It does **not** recover the true plant.
  The single-tone rank deficiency is intrinsic to the regressor and
  survives the correction.
- **Stability: it helps.** The pseudo-gradient does not vanish where the
  true cost is stationary, so it keeps injecting motion and the estimate
  *actively wanders* (0.071 -> 0.265, and in the real feedback loop,
  diverges). The exact gradient is a true gradient — it vanishes on the
  zero-error manifold (`g -> 0`, hence the ~0 eigenvalues) and therefore
  *stops* instead of wandering.

So the dropped term is a real imperfection in the paper's derivation and is
**not benign for stability** — restoring it would tame the active
drift/runaway. But it is **not** the cause of "the plant cannot be
identified": that is the observability barrier above, which is invariant
to whether the chain-rule term is kept. The two symptoms this
investigation chased — unobservability (fundamental) and active divergence
(partly the dropped term) — are distinct.

#### Restoring it in the full loop reveals why the pseudo-gradient is load-bearing

`avc.py` now has an opt-in `exact_gradient: true` that adds the chain-rule
term back analytically (verified against finite differences to 1e-9, and
the SPECULA path against an independent NumPy reference to 1e-17). Running
the closed-loop demo with `adapt_plant: true` at the working operating
point (`gx=3`, `c=1`):

| config | plant `\|x1+i x2\|` | disturbance est. | 47 Hz residual |
|---|---|---|---|
| frozen plant (`adapt_plant:false`) | flat at 7.22 | → 592 | **13 nm** |
| adapt, **pseudo**-grad (paper) | wanders 1 → 620 | → 2545 | 444 nm |
| adapt, **exact**-grad (chain term) | **pinned at 7.22** | **→ 0** | 448 nm |

The exact gradient does exactly what the theory promised on the plant: the
estimate stops drifting completely and sits at its seed. But cancellation
is *lost anyway* — the disturbance estimate collapses to zero.

The reason is the real punchline of this whole investigation: **the
pseudo-gradient is load-bearing.** The algorithm minimises the *equation
error* `e = W^T x − y` (model self-consistency), which is not the
cancellation objective. With the pseudo-gradient, the disturbance update
`x3 -= gx·T·cos·e` is effectively **integral action on the residual** —
that is the mechanism that ramps the correction up until the tone
cancels. The exact gradient adds `cr3, cr4` to those components, turning
that integral action into genuine equation-error minimisation, whose
trivial solution is `x3 = x4 = 0` (zero correction). So the two dropped
terms play *opposite* roles: dropping the chain rule on the **plant**
components causes the drift (harmful), but dropping it on the
**disturbance** components is what makes the loop cancel at all
(essential). The paper's pseudo-gradient is "wrong" in a way that is
exactly what makes it work.

#### The hybrid works: exact gradient on the plant, pseudo-gradient on the disturbance

That diagnosis points at a fix, now implemented as `exact_gradient: plant`
(exact chain-rule term on `x1,x2` only; `x3,x4` keep the integral-action
pseudo-gradient). It resolves the tension above — and, unlike the other
two adaptive modes, actually works, *provided the plant step is damped*.

The plant update carries a `(|theta|^2 + 1)` gain (here `~6673`), so it
needs `gx*T*(|theta|^2+1)*c <~ 2` for stability. At `gx=3` that means
`c <~ 0.1`. Sweeping `c` (adapt_plant true, `exact_gradient: plant`):

| `c` | plant-update gain | plant `\|x1+i x2\|` | 47 Hz residual |
|---|---|---|---|
| 1.0 | ~20 | diverges (→ 309) | 470 nm |
| 0.1 | ~2 | 7.22 → 6.36, then **flat** | **2 nm** |
| 0.01 | ~0.2 | 7.22 → 7.13 | 11 nm |
| 0.001 | ~0.02 | ~ frozen | 12 nm |

At `c=0.1` (6 s run) the plant estimate settles cleanly — 7.22 → 6.36 by
t=3 s then flat to 6 s (6.362, 6.353, 6.359, 6.355), phase held at
+169.5°, disturbance plateauing at ~523 — and the steady 47 Hz residual is
**2 nm vs. 13 nm frozen** (mode-0 RMS 24.2 vs. 27 nm). So the plant makes a
small *bounded, settling* excursion that self-optimises the cancellation
point, instead of wandering (pseudo) or collapsing (full exact).

Three things worth being precise about:

- This still does **not** identify the plant — it is unobservable from one
  tone, and 6.36 is not "the true plant", it is a nearby value that
  cancels marginally better at this operating point. What the exact
  gradient buys is a descent direction that does not run away, so the
  bounded excursion is safe.
- It needs the damping (`c<~0.1`); with the paper's `c=1` it diverges, for
  exactly the `(|theta|^2+1)`-gain reason quantified above.
- Its practical advantage over a well-calibrated frozen plant is modest
  (both cancel well). Where it could matter is a plant that *drifts slowly*
  over a night: the hybrid can track slow changes around the seed without
  the runaway, whereas a frozen plant cannot track at all. That case is
  not tested here.

Bottom line: `adapt_plant: false` (frozen, measured plant) remains the
simplest robust choice, but `exact_gradient: plant` with a damped `c` is a
genuine, stable online-adapting alternative — and it is the one
configuration in this whole investigation where adapting the plant online
neither diverges nor degrades cancellation.

### Reading

The algorithm is best understood as an **adaptive amplitude/phase
canceller with a plant prior**, not a joint plant + disturbance
identifier. The paper's own practice — seeding with approximate
closed-loop plant values — matches that reading. So `adapt_plant: false`
is not a workaround for a broken port: it is using the algorithm the way
its authors actually used it, with the plant supplied by calibration
(here, by direct measurement) rather than discovered online.
