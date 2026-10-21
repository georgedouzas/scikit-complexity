"""
# Coupled pendulums

Two identical pendulums joined by a weak spring pass their energy back and forth: the one set swinging slowly comes to
rest while the other takes over, and then the motion flows back, in beats at the difference of the normal frequencies.
"""

# %%
# The left pendulum starts at 0.2 radians and the right one at rest, with the spring `k = 0.5`. The angles of the model
# lie in `[0, 2π)`, so a swing to the left jumps by `2π`. The observer below also logs the horizontal displacements
# `l sin θ` of the bobs, and the blueprint shows them as time series.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'coupled_pendulums'


# Workaround: remove once the core draws angles wrapped to (-π, π] in time series.
class DisplacementObserver(RerunObserver):
    """Observer that also logs the horizontal displacements of the bobs."""

    def on_start(self, problem):
        super().on_start(problem)
        for side, color in (('left', [90, 170, 230]), ('right', [250, 150, 60])):
            self.recording_.log(f'displacement/{side}', rr.SeriesLines(colors=color, names=side, widths=2), static=True)

    def on_frame(self, frame):
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            for side in ('left', 'right'):
                self.recording_.log(f'displacement/{side}', rr.Scalars(np.sin(frame.value(f'{side}.theta'))))


blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/displacement', name='displacement'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, theta1=0.2, theta2=0.0, k=0.5)
observer = DisplacementObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=50)

# %%
# Normal modes and beats
# ----------------------
# For small swings the in-phase mode has the frequency `ω1 = sqrt(g / l)`, which does not stretch the spring, and the
# anti-phase mode `ω2 = sqrt(g / l + 2 k / m)`. Starting one pendulum alone excites both equally, so the left
# displacement should follow `θ0 cos((ω2 - ω1) t / 2) cos((ω1 + ω2) t / 2)`, whose envelope, dashed, beats with the
# period `2π / (ω2 - ω1)`.

times = np.asarray(simulation.trajectory_.times)
left = np.sin(np.ravel(simulation.trajectory_['left.theta']))
right = np.sin(np.ravel(simulation.trajectory_['right.theta']))
slow, fast = np.sqrt(9.81), np.sqrt(9.81 + 2 * 0.5)
envelope = 0.2 * np.abs(np.cos((fast - slow) * times / 2))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(times, left, linewidth=1, label='left')
ax.plot(times, right, linewidth=1, label='right')
ax.plot(times, envelope, '--', color='black', label='beat envelope of the left pendulum')
ax.plot(times, -envelope, '--', color='black')
ax.set_xlabel('time')
ax.set_ylabel('horizontal displacement')
ax.set_title('Energy passes back and forth between the pendulums')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
fig.tight_layout()
plt.show()
