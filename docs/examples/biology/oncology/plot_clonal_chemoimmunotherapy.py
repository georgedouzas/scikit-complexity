"""
# Chemoimmunotherapy of two tumour clones

Under chemotherapy alone, the sensitive clone dies out and the resistant clone, freed from competition, grows to a
tumour of more than 20 million cells.
"""

# %%
# The system is built from its class with the non-small cell lung cancer clones of Piretto et al.: a sensitive clone
# of 90,000 cells and a resistant clone of 40,000 cells. The drug kills the sensitive clone ten times faster than the
# resistant one, and this run gives no vaccine. The observer saves every second frame with the default blueprint,
# whose primary view is the time series of the clones and the immune cells.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import ClonalChemoimmunotherapySystem

SYSTEM = 'clonal_chemoimmunotherapy'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(ClonalChemoimmunotherapySystem(u_vaccine=0.0), observers=[observer])

# %%
# Selection of the resistant clone
# --------------------------------
# Chemotherapy alone should remove the sensitive clone and free the resistant clone from competition, so the tumour
# relapses resistant and grows by two orders of magnitude. With the vaccine, both clones should fall below one cell.

alone = trajectory
combined = simulate(ClonalChemoimmunotherapySystem())
fig, (chemo, both) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axes, trajectory, title in (
    (chemo, alone, 'Chemotherapy alone selects the resistant clone'),
    (both, combined, 'The vaccine clears both clones'),
):
    axes.semilogy(trajectory.times, trajectory.values('tumour.A1').clip(1e-2), label='sensitive clone')
    axes.semilogy(trajectory.times, trajectory.values('tumour.A2').clip(1e-2), label='resistant clone')
    axes.set_xlabel('time (days)')
    axes.set_title(title)
    axes.legend()
chemo.set_ylabel('cells')
fig.tight_layout()
plt.show()
