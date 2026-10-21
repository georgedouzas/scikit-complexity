"""
# Ballistic pendulum

A bullet buries itself in a block hanging on a rod, and the block swings up. The collision loses most of the energy,
but it conserves momentum, so the height of the swing measures the speed of the bullet.
"""

# %%
# A bullet of mass 0.01 at speed 200 hits a block of mass 1 after 1 unit of flight. The default blueprint plots the
# bullet's position and speed on one axis, where the speed of 200 dwarfs the swing, so the observer below logs the
# angle of the pendulum alone, wrapped to (-π, π].

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'ballistic_pendulum'


# Workaround: remove once the core draws angles wrapped to (-π, π] in time series; it logs them in [0, 2π), so a
# swing to the left jumps by 2π.
class AngleObserver(RerunObserver):
    """Observer that also logs the angle of the pendulum wrapped to (-π, π]."""

    def on_frame(self, frame):
        super().on_frame(frame)
        if frame.step % self.interval == 0:
            angle = np.angle(np.exp(1j * frame.value('pendulum.theta')))
            self.recording_.log('angle', rr.Scalars(angle))


blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/angle', name='angle'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, speed=200.0, distance=1.0, m=0.01, M=1.0)
observer = AngleObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=4.0)

# %%
# Height of the swing against the speed of the bullet
# ---------------------------------------------------
# Momentum conservation gives the block the speed `m v / (m + M)` after the impact, so it rises to the height
# `h = (m v / (m + M))^2 / (2 g)`. The highest point `l (1 - cos θ_max)` of runs over a range of bullet speeds should
# lie on that parabola.

speeds = np.linspace(50.0, 400.0, 8)
heights = []
for speed in speeds:
    run = Simulation(create(MODEL, speed=speed, distance=1.0, m=0.01, M=1.0), dt=0.005).run(until=2.0)
    angle = np.angle(np.exp(1j * np.ravel(run.trajectory_['pendulum.theta'])))
    heights.append(1 - np.cos(np.abs(angle).max()))
grid = np.linspace(0.0, 400.0, 200)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, (0.01 * grid / 1.01) ** 2 / (2 * 9.81), color='black', label='momentum conservation')
ax.plot(speeds, heights, 'o', label='simulated')
ax.set_xlabel('bullet speed v')
ax.set_ylabel('height of the swing')
ax.set_title('The swing measures the bullet speed')
ax.legend()
plt.show()
