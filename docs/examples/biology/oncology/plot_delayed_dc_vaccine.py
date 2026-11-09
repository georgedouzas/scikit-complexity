"""
# Dendritic-cell vaccination with division delays

Under four weekly vaccine doses the tumour still grows toward its capacity of `5 10^9` cells, because regulatory T
cells hold down the adaptive response, which the vaccine starts only after the divisions that maturation takes.
"""

# %%
# The system is built from its class with four weekly one-day doses of `10^6` dendritic cells per day and runs for 30
# days, from the book's initial state. The delays read the history of the run, which before the start is the initial
# state. The observer saves every second frame with the default blueprint, whose primary view is the time series of
# the nine populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import DelayedDCVaccineSystem

SYSTEM = 'delayed_dc_vaccine'

solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(DelayedDCVaccineSystem(), solver=solver, observers=[observer])

# %%
# The lag of two divisions
# ------------------------
# The gain of type 1 helper cells over an unvaccinated run should stay at 0 until one helper division `tau3`, stay
# small until two divisions `2 tau3`, when the first vaccinated naive helper cells mature, and then rise steeply.

vaccinated = trajectory
unvaccinated = simulate(DelayedDCVaccineSystem(dose=0.0), solver=solver)
early = simulate(DelayedDCVaccineSystem(), dt=0.02, solver=solver, until=2)
early_unvaccinated = simulate(DelayedDCVaccineSystem(dose=0.0), dt=0.02, solver=solver, until=2)
fig, (lag, cells) = plt.subplots(1, 2, figsize=(10, 4))
lag.plot(
    early.times,
    early.get_values('site.E1H') - early_unvaccinated.get_values('site.E1H'),
    label='gain of type 1 helper cells',
)
lag.axvline(0.47, color='grey', linestyle=':', label='one division tau3')
lag.axvline(0.94, color='black', linestyle='--', label='two divisions 2 tau3')
lag.set_xlabel('time (days)')
lag.set_ylabel('cells')
lag.set_title('The helper response lags the first dose')
lag.legend()
for state, name in (('ED', 'dendritic cells'), ('E1H', 'type 1 helper cells'), ('ER', 'regulatory T cells')):
    cells.semilogy(vaccinated.times, vaccinated.get_values(f'site.{state}'), label=name)
cells.semilogy(
    unvaccinated.times,
    unvaccinated.get_values('site.ED'),
    color='grey',
    linestyle=':',
    label='unvaccinated DCs',
)
cells.set_xlabel('time (days)')
cells.set_ylabel('cells')
cells.set_title('Weekly doses and the populations they drive')
cells.legend()
fig.tight_layout()
plt.show()
