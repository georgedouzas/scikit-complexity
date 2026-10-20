"""
# Pendulum

A bob on a rigid rod swings back and forth under gravity. Small swings keep the period `2π sqrt(l / g)` whatever their
amplitude, but large swings take longer, and the period diverges as the amplitude approaches the upside-down position.
"""

# %%
# The pendulum is released from rest at 2.5 radians, a large swing. The model logs the angle as a time series only,
# so the observer below also draws the rod and the bob, with the trace of the bob. The 2D views of Rerun point their
# vertical axis down, so the bob hangs at `(l sin θ, l cos θ)`.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.special import ellipk

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'pendulum'


class PendulumObserver(RerunObserver):
    """Observer that also draws the rod, the bob, and its trace."""

    def on_start(self, problem):
        super().on_start(problem)
        self.trace_ = []

    def on_frame(self, frame):
        theta = frame.value('bob.theta')
        bob = [np.sin(theta), np.cos(theta)]
        self.trace_.append(bob)
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('swing/trace', rr.LineStrips2D([self.trace_], radii=0.01, colors=[90, 170, 230]))
            self.recording_.log('swing/rod', rr.LineStrips2D([[[0, 0], bob]], radii=0.02, colors=[220, 220, 220]))
            self.recording_.log('swing/bob', rr.Points2D([bob], radii=0.08, colors=[250, 180, 60]))


blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/swing',
        name='swing',
        visual_bounds=rrb.VisualBounds2D(x_range=(-1.5, 1.5), y_range=(-1.0, 1.2)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, theta=2.5, omega=0.0)
observer = PendulumObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=5)

# %%
# Period against amplitude
# ------------------------
# The exact period of a pendulum released from rest at `θ0` is `T = 4 K(sin^2(θ0 / 2)) sqrt(l / g)`, with `K` the
# complete elliptic integral of the first kind. The periods measured from the upward zero crossings of the angular
# velocity should lie on this curve, which starts at `2π sqrt(l / g)` and diverges at `θ0 = π`.


def measured_period(amplitude):
    """Return the mean period of a pendulum released from rest at an amplitude."""
    run = Simulation(create(MODEL, theta=amplitude, omega=0.0), dt=0.005).run(until=30)
    times, velocity = np.asarray(run.trajectory_.times), np.ravel(run.trajectory_['bob.theta_d1'])
    crossings = times[1:][(velocity[:-1] < 0) & (velocity[1:] >= 0)]
    return np.diff(crossings).mean()


amplitudes = np.linspace(0.2, 3.0, 12)
grid = np.linspace(0.01, 3.1, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, 4 * ellipk(np.sin(grid / 2) ** 2) / np.sqrt(9.81), color='black', label='exact period')
ax.plot(amplitudes, [measured_period(amplitude) for amplitude in amplitudes], 'o', label='simulated')
ax.axhline(2 * np.pi / np.sqrt(9.81), linestyle='--', color='grey', label='small swings')
ax.set_xlabel('amplitude θ0 (radians)')
ax.set_ylabel('period')
ax.set_title('Large swings take longer')
ax.legend()
plt.show()
