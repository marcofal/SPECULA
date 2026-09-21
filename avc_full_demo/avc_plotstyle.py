"""Shared plotting style for the report figures.

Colours come from matplotlib's default ``tab10`` cycle, which is what makes
the figures pleasant to look at on screen. That cycle is qualitative, so its
colours are *not* separated in lightness: ``tab:blue`` and ``tab:red`` differ
by about one CIELAB L* unit and print as the same grey. Colour alone is
therefore not enough for a report that may be printed in black and white.

The fix here is not to give up the colours but to stop relying on them:

* every series gets a **distinct dash pattern**, which survives grey scale
  and photocopying untouched, and does the actual work of telling curves
  apart;
* markers vary as well, for plots with few points;
* the ``tab10`` colours are re-ordered so that neighbouring series are as far
  apart in lightness as that palette allows, which costs nothing and helps a
  little.

So colour is the pleasant redundant cue, and dash pattern is the reliable
one. Run this file as a script to print the lightness of the ordering.
"""
import matplotlib as mpl
import numpy as np

# tab10, reordered for the largest lightness contrast between neighbours
# (L* = 48, 67, 47, 74, 52, 64, 42, 71, 58, 53)
COLORS = ['tab:blue', 'tab:orange', 'tab:red', 'tab:olive', 'tab:purple',
          'tab:pink', 'tab:brown', 'tab:cyan', 'tab:green', 'tab:gray']

# Distinct dash patterns: this is what actually separates the series.
DASHES = ['-',
          (0, (5, 1.6)),                          # dashed
          (0, (1.3, 1.3)),                        # dotted
          (0, (6, 1.6, 1.3, 1.6)),                # dash-dot
          (0, (3, 1.3, 1.3, 1.3, 1.3, 1.3)),      # dash-dot-dot
          (0, (9, 2.2)),                          # long dash
          (0, (2.5, 1.3, 1.3, 1.3))]              # short dash-dot

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']

HATCHES = ['', '///', '...', 'xxx', '\\\\\\', '+++']


def lightness(color):
    """CIELAB L* of a colour: what a grey-scale printer will produce."""
    c = np.array(mpl.colors.to_rgb(color))
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = float(lin @ np.array([0.2126, 0.7152, 0.0722]))
    return 116 * np.cbrt(y) - 16 if y > 0.008856 else 903.3 * y


def palette(n):
    """n colours from the reordered tab10 cycle."""
    return [COLORS[k % len(COLORS)] for k in range(n)]


def styles(n, markers=True):
    """Colour, dash pattern and marker for each of n series."""
    out = []
    for k in range(n):
        s = {'color': COLORS[k % len(COLORS)],
             'linestyle': DASHES[k % len(DASHES)]}
        if markers:
            s['marker'] = MARKERS[k % len(MARKERS)]
        out.append(s)
    return out


def series(i, n=5, **kw):
    """Style for the i-th series, with a marker."""
    s = styles(max(n, i + 1))[i]
    s.update(kw)
    return s


def line(i, n=5, **kw):
    """Style for the i-th series, without a marker."""
    s = styles(max(n, i + 1), markers=False)[i]
    s.update(kw)
    return s


def greys(n, lo=0.0, hi=0.72):
    """A pure lightness ramp, for backgrounds and reference lines."""
    if n == 1:
        return [str(lo)]
    return [str(lo + (hi - lo) * k / (n - 1)) for k in range(n)]


def dashes(n):
    return [DASHES[k % len(DASHES)] for k in range(n)]


def apply_rc(plt):
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.4,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.7',
        'axes.prop_cycle': (plt.cycler(color=palette(7))
                            + plt.cycler(linestyle=DASHES)),
    })


if __name__ == '__main__':
    for k, c in enumerate(COLORS):
        print(f'{k}  {c:11s} {mpl.colors.to_hex(c)}  L*={lightness(c):5.1f}')
