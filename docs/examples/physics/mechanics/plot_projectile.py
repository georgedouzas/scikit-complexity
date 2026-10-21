"""
# Projectile

A ball thrown at 45 degrees flies along a parabola and lands at the range `v^2 / g`, the farthest any launch angle
reaches. Air drag bends the path into a lopsided arc that falls more steeply than it rose and lands short.
"""

# %%
# The ball is launched at speed 10 without drag and lands after `2 v sin(45°) / g`, about 1.44 time units. The model
# logs the ball as a point, so the observer below also logs its path, with the height negated because the 2D views of
# Rerun point their vertical axis down. The blueprint shows the flight alone.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'projectile'


# Workaround: remove once the core draws planar trajectories y-up.
class FlightObserver(RerunObserver):
    """Observer that also logs the path of the ball and the ground."""

    def on_start(self, problem):
        super().on_start(problem)
        self.recording_.log(
            'flight/ground',
            rr.LineStrips2D([[[-0.5, 0.0], [10.7, 0.0]]], colors=[150, 150, 150]),
            static=True,
        )
        self.path_ = []

    def on_frame(self, frame):
        x, y = frame.value('ball.x')
        self.path_.append([x, -y])
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('flight/path', rr.LineStrips2D([self.path_], radii=0.04, colors=[90, 170, 230]))
            self.recording_.log('flight/ball', rr.Points2D([self.path_[-1]], radii=0.2, colors=[250, 180, 60]))


blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/flight',
        name='flight',
        visual_bounds=rrb.VisualBounds2D(x_range=(-0.5, 10.7), y_range=(-3.2, 0.4)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, speed=10.0, angle=45.0)
observer = FlightObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=1.44)

# %%
# Parabola and drag
# -----------------
# Without drag the path should lie on the parabola `y = x - g x^2 / v^2` of a 45 degree launch, landing at
# `v^2 / g`. Linear drag shortens the flight, and the paths with larger drag should land closer and fall more
# steeply.

g, speed = 9.81, 10.0
fig, ax = plt.subplots(figsize=(6, 4))
grid = np.linspace(0, speed**2 / g, 200)
ax.plot(grid, grid - g * grid**2 / speed**2, color='black', linewidth=3, alpha=0.3, label='parabola')
for drag in [0.0, 0.3, 1.0]:
    run = Simulation(create(MODEL, speed=speed, angle=45.0, drag=drag), dt=0.005).run(until=1.5)
    position = np.asarray(run.trajectory_['ball.x'])
    above = position[:, 1] >= 0
    ax.plot(position[above, 0], position[above, 1], label=f'drag = {drag}')
ax.set_xlabel('horizontal distance')
ax.set_ylabel('height')
ax.set_title('Projectile paths with and without drag')
ax.legend()
plt.show()
