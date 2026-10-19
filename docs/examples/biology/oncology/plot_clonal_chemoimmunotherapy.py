"""
# Chemoimmunotherapy of two tumour clones

Under chemotherapy alone, the sensitive clone dies out and the resistant clone, freed from competition, grows to a
tumour of more than 20 million cells.
"""

# %%
# The model is created by name from the catalog with the non-small cell lung cancer clones of Piretto et al.: a
# sensitive clone of 90,000 cells and a resistant clone of 40,000 cells. The drug kills the sensitive clone ten times
# faster than the resistant one, and this run gives no vaccine. The observer saves every second frame with the default
# blueprint, whose primary view is the time series of the clones and the immune cells.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'clonal_chemoimmunotherapy'

system = create(MODEL, u_vaccine=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=300)

# %%
# Selection of the resistant clone
# --------------------------------
# Chemotherapy alone should remove the sensitive clone and free the resistant clone from competition, so the tumour
# relapses resistant and grows by two orders of magnitude. With the vaccine, both clones should fall below one cell.

alone = simulation.trajectory_
combined = Simulation(create(MODEL), dt=1.0).run(until=300).trajectory_
fig, (chemo, both) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axes, trajectory, title in (
    (chemo, alone, 'Chemotherapy alone selects the resistant clone'),
    (both, combined, 'The vaccine clears both clones'),
):
    axes.semilogy(trajectory.times, trajectory['tumour.A1'].clip(1e-2), label='sensitive clone')
    axes.semilogy(trajectory.times, trajectory['tumour.A2'].clip(1e-2), label='resistant clone')
    axes.set_xlabel('time (days)')
    axes.set_title(title)
    axes.legend()
chemo.set_ylabel('cells')
fig.tight_layout()
plt.show()
