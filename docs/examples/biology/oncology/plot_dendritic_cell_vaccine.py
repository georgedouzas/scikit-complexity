"""
# Dendritic-cell vaccination

Eight weekly vaccine injections drive the cytotoxic T cells up and the tumour far below the level where it settles
without vaccine, and the tumour regrows once the injections stop.
"""

# %%
# The example builds `DendriticCellVaccineSystem` with its default schedule, a one-hour injection of 0.5 dendritic cells
# per hour every week, 8 times, and runs for 120 days in hours. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the T cells, the tumour, the dendritic cells, and IL-2.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import DendriticCellVaccineSystem

SYSTEM = 'dendritic_cell_vaccine'

solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(DendriticCellVaccineSystem(), dt=12.0, solver=solver, observers=[observer])

# %%
# Vaccinated and unvaccinated tumours
# -----------------------------------
# Without vaccine the cytotoxic T cells stay at their resting level `s2 / d2 = 0.02`, and the tumour should settle at
# `1 - c1 s2 / (d2 r3) = 0.9`. With the vaccine it should fall below 0.05 during the injections.

vaccinated = trajectory
unvaccinated = simulate(DendriticCellVaccineSystem(dose=0.0), dt=12.0, solver=solver)
days = vaccinated.times / 24
fig, (tumour, cells) = plt.subplots(1, 2, figsize=(10, 4))
tumour.plot(days, unvaccinated.get_values('site.A'), label='no vaccine')
tumour.plot(days, vaccinated.get_values('site.A'), label='8 weekly doses')
tumour.axhline(0.9, color='black', linestyle='--', label='equilibrium 0.9')
tumour.set_xlabel('time (days)')
tumour.set_ylabel('tumour (fraction of capacity)')
tumour.set_title('The vaccine shrinks the tumour while it lasts')
tumour.legend()
cells.plot(days, vaccinated.get_values('site.E'), label='cytotoxic T cells')
cells.plot(days, vaccinated.get_values('site.EH'), label='helper T cells')
cells.plot(days, vaccinated.get_values('site.ED'), label='dendritic cells')
cells.set_xlabel('time (days)')
cells.set_ylabel('cells (fraction of capacity)')
cells.set_title('Dendritic cells drive helper and cytotoxic T cells')
cells.legend()
fig.tight_layout()
plt.show()
