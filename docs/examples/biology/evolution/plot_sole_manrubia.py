"""
# Solé-Manrubia extinctions

Every species depends on a few others through couplings of random sign. When the couplings drift and the field of a
species turns negative it goes extinct, the species that depend on it lose a coupling, and extinctions spread in
avalanches. New species fill the empty niches, so the number of species stays constant while the network turns over.
"""

# %%
# The model is created by name from the catalog with 100 species of 5 dependencies each. The observer saves every
# step with the default blueprint, whose primary view is the network of the species, colored by their age, so recent
# replacements are dark.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'sole_manrubia'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, observers=[observer], random_state=1).run(until=299)

# %%
# Sizes of the extinction avalanches
# ----------------------------------
# A new species has age 0, so the species of age 0 count the extinctions of the previous step. After the first
# collapse of the random initial ecosystem, quiet steps should alternate with avalanches of several species, with a
# broad distribution of sizes.

trajectory = simulation.trajectory_
sizes = np.nansum(trajectory['species.age'][1:] == 0, axis=1)
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.bar(trajectory.times[1:], sizes, width=1.0)
left.set_xlabel('step')
left.set_ylabel('extinctions')
left.set_title('Quiet steps and avalanches')
values, counts = np.unique(sizes[50:][sizes[50:] > 0], return_counts=True)
right.loglog(values, counts / counts.sum(), 'o')
right.set_xlabel('avalanche size')
right.set_ylabel('frequency')
right.set_title(f'{np.count_nonzero(sizes[50:] == 0)} quiet steps after step 50')
plt.show()
