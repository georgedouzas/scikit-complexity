"""
# Dendritic-cell vaccination with division delays

Under four weekly vaccine doses the tumour still grows toward its capacity of `5 10^9` cells, because regulatory T
cells hold down the adaptive response, which the vaccine starts only after the divisions that maturation takes.
"""

# %%
# The model is created by name from the catalog with four weekly one-day doses of `10^6` dendritic cells per day and
# runs for 30 days, from the book's initial state. The delays read the history of the run, which before the start is
# the initial state. The observer saves every second frame with the default blueprint, whose primary view is the time
# series of the nine populations.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'delayed_dc_vaccine'

solver = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-10)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, solver=solver, observers=[observer]).run(until=30)

# %%
# The lag of two divisions
# ------------------------
# The gain of type 1 helper cells over an unvaccinated run should stay at 0 until one helper division `tau3`, stay
# small until two divisions `2 tau3`, when the first vaccinated naive helper cells mature, and then rise steeply.

vaccinated = simulation.trajectory_
unvaccinated = Simulation(create(MODEL, dose=0.0), dt=0.1, solver=solver).run(until=30).trajectory_
early = Simulation(create(MODEL), dt=0.02, solver=solver).run(until=2).trajectory_
early_unvaccinated = Simulation(create(MODEL, dose=0.0), dt=0.02, solver=solver).run(until=2).trajectory_
fig, (lag, cells) = plt.subplots(1, 2, figsize=(10, 4))
lag.plot(early.times, early['site.E1H'] - early_unvaccinated['site.E1H'], label='gain of type 1 helper cells')
lag.axvline(0.47, color='grey', linestyle=':', label='one division tau3')
lag.axvline(0.94, color='black', linestyle='--', label='two divisions 2 tau3')
lag.set_xlabel('time (days)')
lag.set_ylabel('cells')
lag.set_title('The helper response lags the first dose')
lag.legend()
for state, name in (('ED', 'dendritic cells'), ('E1H', 'type 1 helper cells'), ('ER', 'regulatory T cells')):
    cells.semilogy(vaccinated.times, vaccinated[f'site.{state}'], label=name)
cells.semilogy(unvaccinated.times, unvaccinated['site.ED'], color='grey', linestyle=':', label='unvaccinated DCs')
cells.set_xlabel('time (days)')
cells.set_ylabel('cells')
cells.set_title('Weekly doses and the populations they drive')
cells.legend()
fig.tight_layout()
plt.show()
