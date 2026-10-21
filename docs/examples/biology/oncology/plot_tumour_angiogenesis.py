"""
# Tumour angiogenesis

A small tumour far from the parent vessel runs short of oxygen. Its hypoxic cells secrete VEGF, which draws new
sprouts out of the vessel towards the tumour, and once the new vessels arrive and bring oxygen the tumour grows again.
"""

# %%
# The model is created by name from the catalog with five sprouts on a 40 by 25 grid, the parent vessel along the
# left edge. The observer saves every step with the default blueprint, whose primary view shows the cells and the
# fields.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tumour_angiogenesis'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=120)

# %%
# Growth follows the vessels
# --------------------------
# The vessels should grow first, and the tumour should stall with a high share of hypoxic cells until they arrive,
# then grow while the hypoxic share falls.

trajectory = simulation.trajectory_
frames = range(len(trajectory.times))
hypoxic = [np.mean(trajectory.agents('cells/tumour', frame).values['hypoxic']) for frame in frames]
fig, (counts, share) = plt.subplots(1, 2, figsize=(11, 4))
counts.plot(trajectory.times, trajectory['cells/tumour.count'], label='tumour cells')
counts.plot(trajectory.times, trajectory['cells/vessels.count'], label='vessel cells')
counts.set_xlabel('step')
counts.set_ylabel('cells')
counts.set_title('New vessels feed the tumour')
counts.legend()
share.plot(trajectory.times, hypoxic)
share.set_xlabel('step')
share.set_ylabel('share of hypoxic tumour cells')
share.set_title(f'Hypoxic share from {max(hypoxic):.2f} at most to {hypoxic[-1]:.2f}')
fig.tight_layout()
plt.show()
