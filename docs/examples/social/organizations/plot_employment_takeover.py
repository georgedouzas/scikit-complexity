"""
# Employment takeover

On a checkerboard of employers and workers, every worker copies the colour of the employer to its right, and every
employer follows a majority of its four workers. From a random start the colours organize into two large regions
whose border creeps for a while, then freezes after about forty steps.
"""

# %%
# The system is created by name from the catalog on a periodic 32 by 32 checkerboard, with half of the employers and
# half of the workers coloured at random, and runs for 50 steps. Naming the colour field as primary puts its view first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'employment_takeover'

system = create_system(SYSTEM, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='cells/colour')
simulation = Simulation(system, dt=1, observers=[observer]).prepare().run(until=50)

# %%
# A frozen state
# --------------
# The rules are deterministic, so the run should reach a fixed point or a short cycle: the number of cells that change
# colour should drop quickly and then reach zero, and the shares of coloured employers and workers should freeze.

trajectory = simulation.trajectory_
colours = trajectory.lattice('cells.colour')
employers = trajectory.lattice('cells.employer')[0] > 0
changes = np.sum(colours[1:] != colours[:-1], axis=(1, 2))
fig, (activity, shares) = plt.subplots(1, 2, figsize=(10, 4))
activity.plot(trajectory.times[1:], changes, 'o-', markersize=3)
activity.set_xlabel('step')
activity.set_ylabel('cells that change colour')
activity.set_title('The colours freeze')
shares.plot(trajectory.times, colours[:, employers].mean(axis=1), label='employers coloured 1')
shares.plot(trajectory.times, colours[:, ~employers].mean(axis=1), label='workers coloured 1')
shares.set_ylim(0, 1)
shares.set_xlabel('step')
shares.set_ylabel('share')
shares.set_title('Shares of the two colours')
shares.legend()
fig.tight_layout()
plt.show()
