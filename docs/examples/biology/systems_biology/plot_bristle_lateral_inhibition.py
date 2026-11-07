"""
# Bristle lateral inhibition

A ring of epithelial cells that inhibit the neural fate of their neighbours starts uniform and almost epidermal. The
cells rise together until small differences break the symmetry: the leading cells turn neural and silence their
neighbours, leaving sparse sensory bristles several cell diameters apart.
"""

# %%
# The system is created by name from the catalog with forty cells, and the observer saves every frame with the default
# blueprint, whose primary view is the time series of the neural states.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'bristle_lateral_inhibition'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).prepare().run(until=60)

# %%
# Spacing of the bristles
# -----------------------
# The trajectories should peel off a common envelope near the saddle on the diagonal, and the neural cells should end
# at least four cells apart.

trajectory = simulation.trajectory_
states = trajectory.values('cells.a')
neural = np.flatnonzero(states[-1] > 0.5)
gaps = np.diff(np.r_[neural, neural[0] + states.shape[1]])
fig, (series, ring) = plt.subplots(1, 2, figsize=(11, 4.5))
series.plot(trajectory.times, states, lw=0.7)
series.set_xlabel('time')
series.set_ylabel('neural state a')
series.set_title('Trajectories peel off a common envelope')
ring.bar(np.arange(states.shape[1]), states[-1])
ring.set_xlabel('cell on the ring')
ring.set_ylabel('final neural state')
ring.set_title(f'Gaps between bristles: {gaps.tolist()}')
fig.tight_layout()
plt.show()
