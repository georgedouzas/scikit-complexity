"""
# Fractional damped oscillator

With no friction at all, `gamma = 0`, the oscillator of order 0.8 still loses its amplitude: the memory of the
fractional derivatives damps it, and the orbit spirals into the origin of the phase plane. The displacement follows
the Mittag-Leffler function `E_(2 alpha)(-omega^2 t^(2 alpha))`, the solution of the fractional oscillation equation.
"""

# %%
# The system logs `x` and `v` as time series. After the run, its orbit in the phase plane of `x` and `v` is logged once
# as a line, next to the closed orbit of order 1, and the blueprint puts this view first, as the primary view.

import matplotlib.pyplot as plt
import mpmath
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.fractional_dynamics import FractionalOscillatorSystem

SYSTEM = 'fractional_oscillator'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(
            origin='/phase_plane',
            name='phase plane of x and v',
            visual_bounds=rrb.VisualBounds2D(x_range=[-110, 110], y_range=[-110, 110]),
        ),
        rrb.TimeSeriesView(origin='/oscillator', name='x, v', axis_x=whole_run),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=2)
trajectory = simulate(FractionalOscillatorSystem(order=0.8), observers=[observer])
harmonic = simulate(FractionalOscillatorSystem(order=1.0), until=2 * np.pi)
strips = [
    np.column_stack([100 * run.values('oscillator.x'), -100 * run.values('oscillator.v')])
    for run in (harmonic, trajectory)
]
observer.recording_.log(
    'phase_plane',
    rr.LineStrips2D(strips, radii=0.8, colors=[[90, 170, 255], [255, 170, 40]]),
    static=True,
)
observer.recording_.flush()

# %%
# Mittag-Leffler oscillation
# --------------------------
# The Laplace transform of the two equations with `v(0) = 0` gives `x(t) = x(0) E_(2 alpha)(-omega^2 t^(2 alpha))`.
# The simulated displacement of every order should lie on it, oscillating with an amplitude that decays as a power
# law for `alpha < 1` and keeping its amplitude for order 1. The Mittag-Leffler function is the power series
# `E_a(z) = sum_k z^k / Gamma(a k + 1)`, whose terms reach `exp(30)` before they cancel at `t = 30`, so it is summed
# with 60 digits.


def mittag_leffler_series(alpha: float, values: np.ndarray, terms: int = 400) -> np.ndarray:
    """Return the Mittag-Leffler function `E_alpha(z)` at every value by its power series summed with 60 digits."""
    with mpmath.workdps(60):
        coefficients = [1 / mpmath.gamma(alpha * k + 1) for k in reversed(range(terms))]
        return np.array([float(mpmath.polyval(coefficients, mpmath.mpf(z))) for z in values])


fig, ax = plt.subplots(figsize=(7, 4.5))
for order, color in ((1.0, 'tab:blue'), (0.9, 'tab:green'), (0.8, 'tab:orange'), (0.6, 'tab:red')):
    run = simulate(FractionalOscillatorSystem(order=order))
    exact = mittag_leffler_series(2 * order, -(run.times ** (2 * order)))
    ax.plot(run.times, run.values('oscillator.x'), 'o', markersize=1.5, color=color, label=f'order {order}')
    ax.plot(run.times, exact, '-', linewidth=0.8, color='black')
ax.set_xlabel('time t')
ax.set_ylabel('x(t)')
ax.set_title('Simulated oscillation on E_(2 alpha)(-t^(2 alpha)) (black)')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
