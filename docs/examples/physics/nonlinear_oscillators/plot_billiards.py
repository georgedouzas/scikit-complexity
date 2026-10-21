"""
# Dynamical billiards of Sinai and mushroom shapes

A particle bounces around a square table with a round obstacle in the middle. The convex disc scatters nearby paths
apart at every hit, so the motion of the Sinai billiard is chaotic and its path soon criss-crosses the whole table.
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

MODEL = 'billiards'


# Workaround: remove once the core draws billiard tables and trajectories in a y-up 2D view.
class TableObserver(RerunObserver):
    """Observer that also logs the walls of the Sinai table and the path of the particle."""

    def on_start(self, problem):
        super().on_start(problem)
        square = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
        angles = np.linspace(0, 2 * np.pi, 100)
        disc = np.column_stack([0.4 * np.cos(angles), 0.4 * np.sin(angles)])
        self.recording_.log('table/walls', rr.LineStrips2D([square, disc], colors=[220, 220, 220]), static=True)
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
        visual_bounds=rrb.VisualBounds2D(x_range=(-1.5, 1.5), y_range=(-1.1, 1.1)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, shape='sinai', x=0.1, y=0.6, vx=0.6, vy=0.8)
observer = TableObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=4)
simulation = Simulation(system, dt=0.04, observers=[observer]).run(until=40)

# %%
# Separation of nearby paths
# --------------------------
# A second particle starts with its velocity turned by `10^-8` radians. In a chaotic billiard the distance between
# the two particles should grow exponentially, a straight line on the logarithmic axis, until it saturates at the size
# of the table.

turn = 1e-8
vx, vy = 0.6 * np.cos(turn) - 0.8 * np.sin(turn), 0.6 * np.sin(turn) + 0.8 * np.cos(turn)
paths = [
    Simulation(create(MODEL, shape='sinai', x=0.1, y=0.6, vx=velocity[0], vy=velocity[1]), dt=0.01).run(until=60)
    for velocity in [(0.6, 0.8), (vx, vy)]
]
times = np.asarray(paths[0].trajectory_.times)
distance = np.hypot(
    np.ravel(paths[0].trajectory_['particle.x']) - np.ravel(paths[1].trajectory_['particle.x']),
    np.ravel(paths[0].trajectory_['particle.y']) - np.ravel(paths[1].trajectory_['particle.y']),
)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(times, distance)
ax.set_xlabel('time')
ax.set_ylabel('distance between the particles')
ax.set_title('Exponential divergence in the Sinai billiard')
plt.show()
