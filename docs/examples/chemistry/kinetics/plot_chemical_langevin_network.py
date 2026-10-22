"""
# Chemical Langevin network

The enzyme reaction `E + S <-> C -> E + P` written as a chemical Langevin equation. Every reaction drives all the
species it changes with one Wiener process, so the amounts fluctuate around the rate equations while the total enzyme
`E + C` and the total substrate `S + C + P` stay exactly constant.
"""

# %%
# The model is created by name from the catalog with the rate constants of the Michaelis-Menten model, and starts with
# 100 enzymes and 300 substrates. It is integrated with the Euler-Maruyama scheme for 50 time units. The default
# blueprint shows the four amounts in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'chemical_langevin_network'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
solver = EulerMaruyama(dt=0.001)
simulation = Simulation(system, dt=0.5, solver=solver, observers=[observer], random_state=0).run(until=50)

# %%
# Fluctuations around the rate equations and exact conservation
# -------------------------------------------------------------
# The complex should fluctuate around the solution of the rate equations of `michaelis_menten`, since the noise has
# mean zero. One increment drives every species of a reaction, so the totals `E + C` and `S + C + P` should stay at 100
# and 300 to rounding error at every frame.

trajectory = simulation.trajectory_
deterministic = Simulation(create('michaelis_menten'), dt=0.5).run(until=50).trajectory_
enzyme = trajectory['E.n'] + trajectory['C.n']
substrate = trajectory['S.n'] + trajectory['C.n'] + trajectory['P.n']
fig, (amounts, totals) = plt.subplots(1, 2, figsize=(10, 4))
amounts.plot(trajectory.times, trajectory['C.n'], label='complex, Langevin')
amounts.plot(deterministic.times, deterministic['C.n'], color='black', linestyle='--', label='complex, rate equations')
amounts.set_xlabel('time')
amounts.set_ylabel('amount')
amounts.set_title('Noise around the rate equations')
amounts.legend()
totals.plot(trajectory.times, enzyme - 100, label='E + C - 100')
totals.plot(trajectory.times, substrate - 300, label='S + C + P - 300')
totals.set_xlabel('time')
totals.set_ylabel('deviation of the total')
totals.set_title('Totals conserved under the noise')
totals.legend()
fig.tight_layout()
plt.show()
