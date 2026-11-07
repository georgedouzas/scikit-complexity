"""
# Chemical Langevin network

The enzyme reaction `E + S <-> C -> E + P` written as a chemical Langevin equation. Every reaction drives all the
species it changes with one Wiener process, so the amounts fluctuate around the rate equations while the total enzyme
`E + C` and the total substrate `S + C + P` stay exactly constant.
"""

# %%
# The system is built from its class with the rate constants of the Michaelis-Menten system, and starts with
# 100 enzymes and 300 substrates. It is integrated with the Euler-Maruyama scheme for 50 time units. The default
# blueprint shows the four amounts in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.kinetics import ChemicalLangevinNetworkSystem, MichaelisMentenSystem

SYSTEM = 'chemical_langevin_network'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(ChemicalLangevinNetworkSystem(), observers=[observer], random_state=0)

# %%
# Fluctuations around the rate equations and exact conservation
# -------------------------------------------------------------
# The complex should fluctuate around the solution of the rate equations of `MichaelisMentenSystem`, since the noise has
# mean zero. One increment drives every species of a reaction, so the totals `E + C` and `S + C + P` should stay at 100
# and 300 to rounding error at every frame.

deterministic = simulate(MichaelisMentenSystem())
enzyme = trajectory.values('E.n') + trajectory.values('C.n')
substrate = trajectory.values('S.n') + trajectory.values('C.n') + trajectory.values('P.n')
fig, (amounts, totals) = plt.subplots(1, 2, figsize=(10, 4))
amounts.plot(trajectory.times, trajectory.values('C.n'), label='complex, Langevin')
amounts.plot(
    deterministic.times,
    deterministic.values('C.n'),
    color='black',
    linestyle='--',
    label='complex, rate equations',
)
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
