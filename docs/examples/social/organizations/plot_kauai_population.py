"""
# Kauai population

One agent stands for each resident of the census regions of a map, here a small synthetic island. Residents commute
between their home, a facility of work, and a favourite place. Their sentiment follows the rules of Zanbaka et al. in
Carmichael et al. (2019 ch. 3 §4.3): it responds to events on the topics a resident weighs, and each resident chooses a
partner from its grid cell of 1 km, whose sentiment and its own converge at rates set by how strong each opinion is. The
census, the topics, and the demographics here are synthetic, and the security interactions and the adverse agents of
the chapter are left out.
"""

# %%
# The default island runs for thirty steps. The default blueprint shows the residents over the outlines of the regions.

import matplotlib.pyplot as plt

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'kauai_population'

system = create(MODEL)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = (
    Simulation(system, observers=[observer], observables=observables, random_state=0).check().prepare().run(until=30)
)

# %%
# Where the residents stand
# -------------------------
# The residents gather in their homes, facilities, and favourite places in turn, a cycle of three steps. The mean
# sentiment shows how the population responds as a whole.

trajectory = simulation.trajectory_
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for frame, label in zip((0, 1, 2), ('home', 'work', 'favourite'), strict=True):
    locations = trajectory.agents('residents', frame).locations
    axes[0].scatter(locations[:, 0], locations[:, 1], s=6, label=label)
axes[0].set_aspect('equal')
axes[0].set_title('One commuting cycle')
axes[0].legend()
axes[1].plot(trajectory.times, trajectory.values('mean_sentiment'))
axes[1].set_xlabel('step')
axes[1].set_ylabel('mean sentiment')
axes[1].set_title('Sentiment of the island')
plt.tight_layout()
plt.show()
