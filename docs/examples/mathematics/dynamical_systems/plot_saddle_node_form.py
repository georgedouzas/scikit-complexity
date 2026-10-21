"""
# Saddle-node normal form

Just past the saddle-node bifurcation, `mu` slightly below 0, the two equilibria have vanished, yet the state still
lingers for a long time near `x = 0`, where they collided: the ghost of the saddle-node, a bottleneck whose passage
time diverges as `π / √|mu|`.
"""

# %%
# The run starts at `x = 1` with `mu = -0.004`. The state slides down to the bottleneck in a few time units, crawls
# through it for about 40 time units, and then escapes towards negative values. The default blueprint shows `x` as a
# time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'saddle_node_form'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(create(MODEL, x=1.0, mu=-0.004), dt=0.1, observers=[observer]).run(until=48)

# %%
# Passage time through the bottleneck
# -----------------------------------
# The time to go from `x = 1` to `x = -1` should be dominated by the bottleneck and approach `π / √|mu|` as `mu`
# approaches 0 from below. Above 0 the same start should settle on the stable equilibrium `√mu`.

distances = np.geomspace(1e-4, 1e-1, 10)
passages = []
for distance in distances:
    width = np.sqrt(distance)
    until = 2 * np.arctan(1 / width) / width + 0.5
    trajectory = Simulation(create(MODEL, x=1.0, mu=-distance), dt=0.01).run(until=until).trajectory_
    passages.append(trajectory.times[np.argmax(trajectory['form.x'] < -1)])
fig, (series, passage) = plt.subplots(1, 2, figsize=(11, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_['form.x'], label='mu = -0.004')
for mu in (0.01, 0.1):
    settled = Simulation(create(MODEL, x=1.0, mu=mu), dt=0.1).run(until=48).trajectory_
    series.plot(settled.times, settled['form.x'], label=f'mu = {mu}')
series.set_ylim(-2, 1.05)
series.set_xlabel('time')
series.set_ylabel('x')
series.set_title('The ghost holds the state near 0')
series.legend()
passage.loglog(distances, passages, 'o', label='simulated')
passage.loglog(distances, np.pi / np.sqrt(distances), color='black', label=r'$\pi / \sqrt{|\mu|}$')
passage.set_xlabel(r'distance past the bifurcation $|\mu|$')
passage.set_ylabel('passage time from 1 to -1')
passage.set_title('Passage time diverges at the bifurcation')
passage.legend()
fig.tight_layout()
plt.show()
