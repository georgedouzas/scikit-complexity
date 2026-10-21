"""
# Double pendulum

A pendulum hanging from the bob of another, released from high up, tumbles chaotically: the lower bob traces a tangled,
never repeating path, and two releases that differ by a hair soon move in completely different ways.
"""

# %%
# The rods start high, at 2 and 2.5 radians from the downward vertical, and at rest. The model logs the angles as time
# series only, so the observer below also draws the rods, the bobs, and the trace of the lower bob. The 2D views of
# Rerun point their vertical axis down, so a bob at angle `θ` from the downward vertical sits at `(l sin θ, l cos θ)`
# from its pivot.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'double_pendulum'


def bobs(theta1, theta2):
    """Return the positions of the two bobs of unit rods, with the vertical axis pointing down."""
    first = np.array([np.sin(theta1), np.cos(theta1)])
    return first, first + np.array([np.sin(theta2), np.cos(theta2)])


# Workaround: remove once the core draws pendulums as bodies (rod and bob) in a y-up 2D view.
class DoublePendulumObserver(RerunObserver):
    """Observer that also draws the rods, the bobs, and the trace of the lower bob."""

    def on_start(self, problem):
        super().on_start(problem)
        self.trace_ = []

    def on_frame(self, frame):
        first, second = bobs(frame.value('pendulum.theta1'), frame.value('pendulum.theta2'))
        self.trace_.append(second)
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            self.recording_.log('swing/trace', rr.LineStrips2D([self.trace_], radii=0.008, colors=[90, 170, 230]))
            rods = rr.LineStrips2D([[[0, 0], first, second]], radii=0.025, colors=[220, 220, 220])
            self.recording_.log('swing/rods', rods)
            self.recording_.log('swing/bobs', rr.Points2D([first, second], radii=0.09, colors=[250, 180, 60]))


blueprint = rrb.Blueprint(
    rrb.Spatial2DView(
        origin='/swing',
        name='swing',
        visual_bounds=rrb.VisualBounds2D(x_range=(-2.2, 2.2), y_range=(-2.1, 2.1)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, theta1=2.0, theta2=2.5)
observer = DoublePendulumObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=10)
simulation = Simulation(system, dt=0.005, observers=[observer]).run(until=12)

# %%
# Sensitive dependence on the initial angle
# -----------------------------------------
# A second double pendulum starts with its upper angle larger by `10^-8` radians. The distance between the lower bobs
# of the two pendulums should grow exponentially, a straight line on the logarithmic axis, until it is as large as the
# pendulum itself.

runs = [
    Simulation(create(MODEL, theta1=np.pi / 2 + shift, theta2=np.pi / 2), dt=0.005).run(until=20)
    for shift in [0.0, 1e-8]
]
lower = [
    bobs(np.ravel(run.trajectory_['pendulum.theta1']), np.ravel(run.trajectory_['pendulum.theta2']))[1] for run in runs
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(runs[0].trajectory_.times, np.linalg.norm(lower[0] - lower[1], axis=0))
ax.set_xlabel('time')
ax.set_ylabel('distance between the lower bobs')
ax.set_title('Two releases 1e-8 radians apart')
plt.show()
