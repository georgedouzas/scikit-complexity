"""
# Circular billiard

A particle bouncing inside a circle hits the wall at the same angle every time, so its path is a regular star of chords
that never enters an inner circle, the caustic. The circular billiard is integrable: no chaos, only rotation.
"""

# %%
# The model logs the position as time series only, so the observer below also logs the walls of the table and the
# path of the particle, sampled at every step of the simulation, with `y` negated because the 2D views of Rerun point
# their vertical axis down. The blueprint shows the table alone.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'circular_billiard'


# Workaround: remove once the core draws billiard tables and trajectories in a y-up 2D view.
class TableObserver(RerunObserver):
    """Observer that also logs the wall of the unit circle and the path of the particle."""

    def on_start(self, problem):
        super().on_start(problem)
        angles = np.linspace(0, 2 * np.pi, 200)
        wall = np.column_stack([np.cos(angles), np.sin(angles)])
        self.recording_.log('table/wall', rr.LineStrips2D([wall], colors=[220, 220, 220]), static=True)
        self.path_ = []

    def on_frame(self, frame):
        self.path_.append([frame.value('particle.x'), -frame.value('particle.y')])
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('table/path', rr.LineStrips2D([self.path_], radii=0.006, colors=[90, 170, 230]))
            self.recording_.log('table/particle', rr.Points2D([self.path_[-1]], radii=0.04, colors=[250, 180, 60]))


blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/table',
        name='table',
        visual_bounds=rrb.VisualBounds2D(x_range=(-1.05, 1.05), y_range=(-1.05, 1.05)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, x=0.3, y=0.0, vx=0.4, vy=1.0)
observer = TableObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=20)

# %%
# The caustic
# -----------
# Angular momentum `L = x v_y - y v_x` is conserved between bounces and by every reflection, so each chord passes at
# the distance `|L| / |v|` from the centre. The distance of the particle from the centre should never drop below
# that radius, and its histogram should pile up at the caustic, where the chords are tangent.

x, y = np.ravel(simulation.trajectory_['particle.x']), np.ravel(simulation.trajectory_['particle.y'])
vx, vy = np.ravel(simulation.trajectory_['particle.vx']), np.ravel(simulation.trajectory_['particle.vy'])
momentum = x * vy - y * vx
caustic = abs(momentum[0]) / np.hypot(0.4, 1.0)
fig, (conserved, radii) = plt.subplots(1, 2, figsize=(10, 4))
conserved.plot(simulation.trajectory_.times, momentum)
conserved.set_ylim(0, 0.5)
conserved.set_xlabel('time')
conserved.set_ylabel('angular momentum L')
conserved.set_title('L is conserved by every bounce')
radii.hist(np.hypot(x, y), bins=40, density=True, alpha=0.6)
radii.axvline(caustic, color='black', linestyle='--', label=f'caustic |L| / |v| = {caustic:.3f}')
radii.set_xlabel('distance from the centre')
radii.set_ylabel('probability density')
radii.set_title('The path never enters the caustic')
radii.legend()
fig.tight_layout()
plt.show()
