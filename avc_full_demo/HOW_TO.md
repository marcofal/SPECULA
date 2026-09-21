# How to calibrate and run a working AVC demo

A practical guide to getting SPECULA's `AVC` (Adaptive Vibration
Cancellation) object actually rejecting a vibration, plus the reasoning
behind each choice. Everything here was learned the hard way while getting
`avc_full_demo` from "makes things worse" to "−91 % residual"; the
investigation itself is in `README.md`.

**TL;DR — the five things that matter, in order:**

1. Put the vibration where the main loop *cannot* help (see §2). If it is
   inside the integrator's bandwidth, AVC is pointless.
2. **Measure** the plant `x0,x1` with the loop **closed** (§3). Do not use
   an analytic model — its sign convention will not match your wiring, and
   a 180° error makes AVC *amplify* the vibration.
3. Set `adapt_plant: false` (§4.1). Online plant adaptation is
   mathematically unobservable from a single tone and will diverge.
4. Set `gx` large enough to converge within your run (§4.2).
5. Set `gomega` small enough that the PLL is not captured by the
   atmosphere (§4.3).

---

## 1. Architecture

```
                                    ┌──────────────┐
   atmosphere ──┐                   │  Integrator  │◄─── rec.out_modes
                ├──► WFS ──► slopec ─┴──► rec ──┐   │
   vibration ───┤                              │   ▼
                │                    mode_slicer│  comm_combiner ──► lowpass ──► DM ──┐
                │                              ▼   ▲                                 │
                └──────────────────────────── AVC ─┘                                 │
                              (pupil) ◄────────────────────────────────────────────── ┘
                                              (dm.out_layer:-1, 1-frame delay)
```

The integrator handles **all 40 modes** and rejects the low-frequency
atmosphere. AVC acts on **one mode only** (here mode 0 = tip), targets one
narrow-band peak, and its command is *summed* with the integrator's by
`comm_combiner` before the actuator low-pass and DM. This is the parallel
architecture of Fig. 1 in Muradore et al. (SPIE 8447-38) — AVC can be
switched on and off without touching the main controller.

Files:

| File | Purpose |
|---|---|
| `params_scao_avc_full_demo.yml` | the working closed-loop demo, AVC active |
| `params_scao_vibration_baseline.yml` | identical loop, **no AVC** — the "before" run |
| `params_measure_plant.yml` | probe harness that measures `x0,x1` (§3) |
| `analyze_results.py` | compares the latest baseline/AVC pair, writes `avc_analysis.pdf` |
| `calibrate_avc.py` | analytic plant model — **reference only**, see §3.4 |

---

## 2. Designing a loop where AVC is actually useful

This is the step people skip, and it invalidates everything downstream.

**AVC is only worth running where the main controller does not already
reject the disturbance.** A standard integrator rejects everything well
inside its bandwidth. Rough rule for an integrator of gain `g` at sample
period `T`:

```
0 dB crossover  f_c  ≈  g / (2π T)
```

For `g=0.5`, `T=1 ms` → `f_c ≈ 80 Hz`. Anything well below that is already
crushed (we measure −38 dB at 1 Hz, −24 dB at 5 Hz), and adding AVC there
buys nothing.

Above the crossover the **waterbed effect** takes over: the rejection
transfer `|S| = |1/(1+HOL)|` rises above 1 and the loop *amplifies*. That
is where a vibration hurts most and where AVC pays off. In the current
demo:

| freq | 1 Hz | 5 Hz | 10 Hz | 20 Hz | **47 Hz** | 100 Hz |
|---|---|---|---|---|---|---|
| \|S\| | −38 dB | −24 dB | −17 dB | −9 dB | **+19 dB** | ≈0 dB |

A 50 nm tone at 47 Hz becomes a **449 nm** residual — the integrator makes
it ~10× worse. That is a compelling case for AVC, and it mirrors the
paper's 48 Hz NACO vibration sitting "close to the amplification range".

**How to find the amplification region for your loop.** Either:

- *Analytically*, with `specula.lib.avc_plant_model`:

  ```python
  import numpy as np
  from specula.lib.avc_plant_model import closed_loop_sensitivity
  f = np.linspace(1, 400, 800)
  S = np.abs(closed_loop_sensitivity(f, T=0.001, loop_gain=0.5,
                                     loop_delay=3*0.001, lpf_fc=80.0))
  print(f[S.argmax()], 20*np.log10(S.max()))   # peak location and height
  ```

  Use `loop_delay` = integrator `delay` + 1 frame (the `dm.out_layer:-1`
  reference in `prop`). The model's **magnitudes** are reliable (it
  predicted the 45 Hz peak location and +12 dB where we measured +19 dB);
  its **phase/sign** is not (§3.4).

- *Empirically*, which is definitive: inject a tone at the frequency of
  interest with `seeing` ≈ 0.01 (so only the tone is present), run once
  with `int_gain: 0` and once with the real gain, and lock-in on both.
  `|S| = closed / open`. This is what produced the table above.

**Tuning knobs if the peak is in the wrong place:** raising the actuator
low-pass cutoff or lowering the integrator gain moves and flattens the
waterbed peak. `lpf_fc=80 Hz` costs a lot of phase margin and puts a tall
peak at ~45 Hz; `lpf_fc=200 Hz` moves it to ~58 Hz; `g=0.2` reduces the
peak to +7.7 dB.

---

## 3. Calibrating the plant (`x0`, `x1`) — the critical step

`x0,x1` are the real and imaginary parts of the transfer function **from
AVC's own command to the residual AVC measures**, evaluated at the
vibration frequency. Get this wrong and AVC confidently drives the loop in
the wrong direction.

### 3.1 Why it must be measured with the loop closed

Closing the integrator multiplies AVC's plant by the closed-loop
sensitivity `S`. In this demo that is not a detail:

| | \|P\| | phase |
|---|---|---|
| integrator **open** | 0.555 | +119° |
| integrator **closed** | **7.223** | **+169.5°** |

A factor 13 in gain and 50° in phase. `params_measure_plant.yml`
therefore contains the *same integrator* as the demo.

### 3.2 The procedure

`params_measure_plant.yml` injects a known sinusoid at the vibration
frequency into exactly the point where AVC's command enters
(`comm_combiner`), with the real vibration switched off (`vib_wave.amp: 0`)
and the atmosphere effectively off (`seeing: 0.01`), then records the probe
and the mode-0 residual. Lock-in gives `P` directly:

```bash
python -c "
import specula; specula.init(0, precision=1)
from specula.simul import Simul; Simul('params_measure_plant.yml').run()"
```

then

```python
import numpy as np, glob, os
from astropy.io import fits
d = sorted(glob.glob('output_probe/*/'))[-1]
rm = np.array(fits.getdata(os.path.join(d,'res_modes.fits')))
if rm.shape[0] < rm.shape[1]: rm = rm.T
pr = np.array(fits.getdata(os.path.join(d,'probe.fits'))).ravel()
n = rm.shape[0]; sl = slice(1500, n)              # drop the transient
t = np.arange(n)[sl]*0.001
ref = np.exp(-1j*2*np.pi*47.0*t)                  # vibration frequency
P = (2*np.mean(rm[sl,0]*ref)) / (2*np.mean(pr[sl]*ref))
print(f"x0 = {P.real:.6f}\nx1 = {P.imag:.6f}")
print(f"|P| = {abs(P):.4f}  phase = {np.degrees(np.angle(P)):+.2f} deg")
```

Paste `x0`/`x1` into the `avc:` block.

### 3.3 Keep the probe inside the WFS linear range — this bites

Because the loop amplifies ~10× at 47 Hz, a large probe drives the
Shack-Hartmann out of its linear regime and the measured transfer becomes
meaningless:

| probe | closed-loop residual RMS | measured \|P\| |
|---|---|---|
| 500 nm | 1287 nm | 3.64 ← saturated, wrong |
| 30 nm | 153 nm | **7.22** ← correct |
| 20 nm | 102 nm | 7.23 ← consistent |

**The tell-tale sign of saturation is that the measured gain changes when
you change the probe amplitude.** Always measure at two amplitudes; if
they disagree, you are saturating (go smaller) or noise-limited (go
bigger). Aim for a response of ~100–200 nm.

The same applies to the **injected vibration**: 500 nm in this closed loop
gives ~2000 nm of residual and saturates everything. The demo uses 50 nm.

### 3.4 Why not the analytic model?

`calibrate_avc.py` / `specula/lib/avc_plant_model.py` faithfully port
ESO's `generateAVCInitConditions.m` (verified in
`test/test_avc_plant_model.py`), but that model does **not** capture the
sign convention of SPECULA's DM/loop wiring. At 47 Hz it gives phase
−78° where the truth is +119° — essentially **180° out**. Feeding that to
AVC makes the "correction" add to the vibration. Its magnitudes are fine,
so it remains useful for loop design (§2), but always take `x0,x1` from
the measurement.

### 3.5 Re-measure whenever anything changes

Integrator gain or delay, actuator low-pass cutoff, sampling period,
target frequency, or where the command is injected — any of these changes
`P`. Re-run §3.2.

---

## 4. Hyperparameters

Current working values (`params_scao_avc_full_demo.yml`):

```yaml
avc:
  n_avc:        1
  freq:         [45]          # initial guess; true tone is 47 Hz
  gx:           3.0
  gomega:       0.2
  k:            1.0
  c:            1.0
  x0:           -7.102731     # MEASURED, closed loop
  x1:            1.313658
  soft_clamp:   true
  adapt_plant:  false
```

### 4.1 `adapt_plant` — set it to `false`

Default is `true` (faithful to the paper). **Use `false`.**

From a single sinusoid the regressor `W = [θc·c+θs·s, θs·c−θc·s, c, s]`
has its first two entries always a linear combination of the last two, so
the averaged regressor covariance `⟨W Wᵀ⟩` has exactly **two nonzero and
two zero eigenvalues** — the 4-component state is only rank-2 observable,
and the unobservable directions are precisely the plant axes. Adapting
`x1,x2` online therefore cannot identify them; the estimate drifts, and
because it sits in the `1/(x1²+x2²)` denominator of the correction, the
drift feeds back and diverges. Proof: `avc_observability_check.py`.

Enriching the *disturbance* does not fix this (`avc_excitation_check.py`);
observability would have to come from a known probe injected into the
*command*, which is a different algorithm (FxLMS-style). Since we measure
the plant offline anyway (§3), freezing it is the right architecture.

*Symptom if you get this wrong:* `|x1+i·x2|` in `avc_state.fits` wanders
or collapses instead of sitting flat.

This is a property of the published algorithm, not of this port: it
reproduces in a pure-NumPy implementation of the paper's equations with no
SPECULA, no delay and no noise, and it persists in-loop even with a
correct measured seed and a stable step size. See *"Verdict: is the plant
drift the algorithm's, or ours?"* in `README.md` for the evidence and for
the traces of it in the paper. Practical upshot: the plant is meant to
come from calibration (§3), not to be discovered online.

### 4.2 `gx` — disturbance adaptation rate

Controls how fast `x3,x4` (the disturbance quadratures) converge. The
time constant is roughly `2/(gx·T)` samples, i.e. **independent of the
disturbance amplitude**, so scale it to your run length, not your signal.

| `gx` | behaviour in a 4 s run |
|---|---|
| 1.0 | still ramping at t=4 s; notch only −4 dB |
| **3.0** | **converges by ~2 s; notch −31 dB** |
| too high | overshoots, injects broadband power — the *narrowband* notch may still deepen while the *total* RMS gets worse |

Diagnose with the `|x3+i·x4|` trace in `analyze_results.py`'s last panel:
it should rise and **plateau**. Still climbing at the end of the run →
raise `gx`. Oscillating/overshooting → lower it.

Watch the trade-off: in an earlier open-loop configuration `gx=2.0` gave a
deeper 47 Hz notch but a *worse* total RMS than `gx=1.0`, because the
over-correction added broadband power. Judge on both metrics.

### 4.3 `gomega` — frequency (PLL) tracking rate

**Not a dimensionless constant — it must be re-tuned with the tone's
signal-to-noise ratio.** The update is `2·T·gomega·sin(α)·e`, driven by
whatever dominates the error `e`. If the tone does not dominate, the PLL
walks downhill toward whatever does — normally the low-frequency
atmosphere, which is a red spectrum.

Measured with a *weak* (30 nm, no integrator) tone, where atmospheric
power at 5 Hz rivalled the tone:

| `gomega` | result |
|---|---|
| 10.0 | collapses to **6.6 Hz** — captured by the ~5 Hz atmospheric peak |
| 0.5 | drifts 47 → 39–42 Hz |
| 0.05 | stays locked at 47.06 Hz |
| 0 | tracking disabled, frequency held |

In the **current closed-loop demo** the integrator suppresses the
low-frequency atmosphere, so the tone dominates the residual spectrum and
the PLL is safe: `gomega=0.2` locks from a wrong 45 Hz guess to 46.94 Hz
with 0.033 Hz jitter. *Rejecting the atmosphere with the integrator is
itself what makes frequency tracking robust.*

If your tone is weak, either lower `gomega`, or set `gomega: 0` and hold a
fixed frequency from `find_vib_peaks` (`specula.lib.find_vib_peaks`) —
legitimate when the vibration frequency is stable (the paper measures
±0.1 Hz on real NACO data).

*Symptom if you get this wrong:* `avc_freq.fits` slides monotonically away
from the true frequency, typically downward.

### 4.4 `freq` — initial guess

Comes from `find_vib_peaks` on a baseline PSD in practice. It need not be
exact: the demo locks from 45 Hz onto a 47 Hz tone in ~1.5 s. But the
frequency must be acquired before cancellation can build, so allow for the
transient (or seed it accurately and use `gomega: 0`).

### 4.5 `soft_clamp` and `theta_min_energy`

`theta_min_energy` (default `1e-2`) guards the `1/(x1²+x2²)` division.
With `soft_clamp: false` (the Matlab-reference behaviour) it *hard-resets*
the correction whenever the denominator dips below the threshold — a
discontinuity that causes limit-cycle chattering if the state sits near it.
`soft_clamp: true` instead applies an unconditional soft floor
`x1²+x2²+theta_min_energy`. **With `adapt_plant: false` this is largely
moot** (the denominator is frozen at a healthy value), but `true` is the
safer default.

### 4.6 `k` and `c`

`k` is the paper's `gα`, the phase-update coupling; `1.0` works and we
never needed to touch it. `c` scales the `x1,x2` update only, so with
`adapt_plant: false` it has **no effect at all**. Leave both at `1.0`.

### 4.7 `n_avc` and multiple vibrations

One AVC instance per vibration peak. `n_avc > 1` vectorises independent
instances; feed them the same measurement and sum their corrections. Each
needs **its own** `x0,x1` measured at **its own** frequency, since `P(f)`
varies strongly across the waterbed peak.

---

## 5. Step-by-step recipe

1. **Build the closed loop without AVC** (`params_scao_vibration_baseline.yml`):
   atmosphere → WFS chain → integrator → actuator low-pass → DM, with
   `prop` referencing `dm.out_layer:-1`. Verify it is **stable** (mode-0
   RMS settles rather than growing).
2. **Find the amplification region** (§2) and place the vibration there.
   Confirm the loop rejects at low frequency and amplifies at your target.
3. **Choose the vibration amplitude** so the *amplified* closed-loop
   residual stays in the WFS linear range (here 50 nm in → ~450 nm out).
4. **Measure the plant** (§3) with the integrator closed and a small probe.
   Cross-check at two probe amplitudes.
5. **Fill in the `avc:` block**: measured `x0,x1`, `adapt_plant: false`,
   `soft_clamp: true`, `freq` from `find_vib_peaks`, and starting values
   `gx=1`, `gomega=0.2`.
6. **Run both** and compare:
   ```bash
   python -c "
   import specula; specula.init(0, precision=1)
   from specula.simul import Simul
   Simul('params_scao_vibration_baseline.yml').run()
   Simul('params_scao_avc_full_demo.yml').run()"
   python analyze_results.py
   ```
7. **Tune** `gx` up until `|x3+i·x4|` plateaus well before the end of the
   run, watching that total RMS improves and not just the notch.

Runs take ~25 s per simulated second on CPU; keep `total_time` around 4 s
while iterating.

---

## 6. Diagnosing a broken run

| Symptom | Likely cause | Fix |
|---|---|---|
| Residual **worse** with AVC, disturbance estimate grows without bound | `x0,x1` mis-phased (often 180°) | re-measure closed-loop (§3); do not use the analytic model |
| `\|x1+i·x2\|` wanders or collapses | `adapt_plant: true` | set `false` (§4.1) |
| Frequency slides away, usually downward | `gomega` too high for the tone's SNR | lower `gomega`, or `gomega: 0` (§4.3) |
| `\|x3+i·x4\|` still climbing at end of run | `gx` too low | raise `gx` (§4.2) |
| Deeper notch but worse total RMS | `gx` too high, broadband over-correction | lower `gx` |
| Measured `\|P\|` changes with probe amplitude | WFS saturation | shrink the probe (§3.3) |
| AVC changes nothing at all | vibration is inside the integrator's bandwidth | move the tone into the amplification region (§2) |
| Everything looks wrong after edits | analysing a stale `output/` directory | clear `output/` between experiments — this cost us a full debugging cycle |

## 7. Sanity checks before believing a result

- Baseline and AVC configs differ **only** by the AVC module.
- Closed-loop residual RMS is well inside the WFS linear range (≲ 300 nm
  for this 64 px / 8×8 setup).
- `|x1+i·x2|` is a flat line at the calibrated `|P|`.
- `|x3+i·x4|` plateaus.
- `avc_freq.fits` converges and stays.
- The improvement appears **at the vibration frequency** in the PSD panel,
  not as a broadband shift.
