"""
# Chemotherapy of a tumour under immune surveillance

A constant drug infusion drives the tumour cells to zero within two months, while the normal cells dip and then
recover close to their carrying capacity.
"""

# %%
# The model is created by name from the catalog with a constant infusion of one dose unit a day, from the initial
# conditions of de Pillis and Radunskaya: a tumour at a quarter of the capacity, few immune cells, and healthy normal
# cells. The observer saves every second frame with the default blueprint, whose primary view is the time series of
# the cells and the drug.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'de_pillis_radunskaya'

system = create(MODEL, u=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=150)

# %%
# Untreated and treated tumours
# -----------------------------
# Untreated, the same tumour escapes the immune cells and settles at the high equilibrium, where it outnumbers the
# normal cells. With the infusion, the tumour should fall below 1% of its initial size and the normal cells should stay
# above 0.7 of their capacity.

untreated = Simulation(create(MODEL), dt=0.5).run(until=150).trajectory_
treated = simulation.trajectory_
fig, (tumour, normal) = plt.subplots(1, 2, figsize=(10, 4))
tumour.semilogy(untreated.times, untreated['patient.A'], label='untreated')
tumour.semilogy(treated.times, treated['patient.A'], label='infusion u = 1')
tumour.axhline(0.0025, color='grey', linestyle=':', label='1% of the initial tumour')
tumour.set_ylim(1e-8, 1)
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour cells (normalised)')
tumour.set_title('The infusion clears the tumour')
tumour.legend()
normal.plot(untreated.times, untreated['patient.N'], label='untreated')
normal.plot(treated.times, treated['patient.N'], label='infusion u = 1')
normal.axhline(0.7, color='grey', linestyle=':', label='lower bound 0.7')
normal.set_xlabel('time (days)')
normal.set_ylabel('normal cells (normalised)')
normal.set_title('The normal cells recover under treatment')
normal.legend()
fig.tight_layout()
plt.show()
