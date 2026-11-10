"""
# Ballistic pendulum

A bullet buries itself in a block hanging on a rod, and the block swings up. The collision loses most of the energy,
but it conserves momentum, so the height of the swing measures the speed of the bullet.
"""

# %%
# A bullet of mass 0.01 at speed 200 hits a block of mass 1 after 1 unit of flight. The default blueprint plots the
# bullet's position and speed on one axis, where the speed of 200 dwarfs the swing, so the blueprint below shows the
# angle of the pendulum alone.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.mechanics import BallisticPendulumSystem

SYSTEM = 'ballistic_pendulum'

blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/', contents=['+ /pendulum/theta'], name='angle'),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = BallisticPendulumSystem(speed=200.0, distance=1.0, m=0.01, M=1.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=2)
trajectory = simulate(system, until=4.0, observers=[observer])

# %%
# Height of the swing against the speed of the bullet
# ---------------------------------------------------
# Momentum conservation gives the block the speed `m v / (m + M)` after the impact, so it rises to the height
# `h = (m v / (m + M))^2 / (2 g)`. The highest point `l (1 - cos θ_max)` of runs over a range of bullet speeds should
# lie on that parabola.

speeds = np.linspace(50.0, 400.0, 8)
heights = []
for speed in speeds:
    run = simulate(BallisticPendulumSystem(speed=speed, distance=1.0, m=0.01, M=1.0), until=2.0, dt=0.005)
    angle = np.ravel(run.get_values('pendulum.theta'))
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
