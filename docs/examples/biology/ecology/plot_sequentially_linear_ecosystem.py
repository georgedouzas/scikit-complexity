"""
# Sequentially linear ecosystem

Between extinctions the abundances follow linear dynamics on the network of influences, and approach its dominant
eigenvector. When a species falls below the extinction threshold it is replaced by a species with random influences,
which changes the network and starts a new linear stretch. Bursts of extinctions give way to long quiet periods.
"""

# %%
# The model is created by name from the catalog with 60 species that read 4 species each. The observer saves every
# step with the default blueprint, whose primary view is the network of influences colored by the abundances.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sequentially_linear_ecosystem'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=1).run(until=299)

# %%
# Extinctions and the turnover of the network
# -------------------------------------------
# The species of age 0 count the extinctions of the previous step. Their cumulative number should rise in steps,
# steep early and flatter later, while the abundances stay normalised and dominated by a few species.

trajectory = simulation.trajectory_
extinctions = np.nansum(trajectory['species.age'][1:] == 0, axis=1)
abundances = np.sort(np.nan_to_num(trajectory['species.x'][-1]))[::-1]
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.step(trajectory.times[1:], np.cumsum(extinctions))
left.set_xlabel('step')
left.set_ylabel('cumulative extinctions')
left.set_title(f'{extinctions[:100].sum()} early and {extinctions[200:].sum()} late extinctions')
right.semilogy(abundances[abundances > 0], 'o')
right.set_xlabel('rank')
right.set_ylabel('abundance')
right.set_title(f'Abundances sum to {abundances.sum():.3f}')
plt.show()
