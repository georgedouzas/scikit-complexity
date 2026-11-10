"""
# Michaelis-Menten

The enzyme binds the substrate into a complex within a few time units, then the complex turns the substrate into
product at a nearly steady rate until the substrate runs out and the enzyme is released.
"""

# %%
# The example builds `MichaelisMentenSystem` with 100 enzymes and 300 substrate molecules as rate equations. The
# default blueprint shows the free enzyme `E`, the substrate `S`, the complex `C`, and the product `P` in one time
# series view.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.kinetics import MichaelisMentenSystem

SYSTEM = 'michaelis_menten'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(MichaelisMentenSystem(), observers=[observer])

# %%
# Rate equations against single molecules
# ---------------------------------------
# With `stochastic=True` the same reactions are random events of Gillespie's algorithm on whole molecules. With
# hundreds of molecules, the stochastic runs should scatter closely around the rate equations, while the totals
# `E + C` of enzyme and `S + C + P` of substrate stay exactly conserved in both.

fig, (curves, totals) = plt.subplots(1, 2, figsize=(10, 4))
for seed in range(5):
    jumps = simulate(MichaelisMentenSystem(stochastic=True), random_state=seed)
    for species, color in [('S', 'C0'), ('C', 'C1'), ('P', 'C2')]:
        curves.step(jumps.times, jumps.get_values(f'{species}.x'), where='post', color=color, alpha=0.3, linewidth=0.8)
for species, color in [('S', 'C0'), ('C', 'C1'), ('P', 'C2')]:
    curves.plot(trajectory.times, trajectory.get_values(f'{species}.x'), color=color, linewidth=2, label=species)
curves.set_xlabel('time')
curves.set_ylabel('molecules')
curves.set_title('Rate equations (thick) and Gillespie runs (thin)')
curves.legend()
enzyme = trajectory.get_values('E.x') + trajectory.get_values('C.x')
substrate = trajectory.get_values('S.x') + trajectory.get_values('C.x') + trajectory.get_values('P.x')
totals.plot(trajectory.times, enzyme, label='E + C')
totals.plot(trajectory.times, substrate, label='S + C + P')
totals.set_ylim(0, 350)
totals.set_xlabel('time')
totals.set_ylabel('molecules')
totals.set_title('The totals of enzyme and substrate are conserved')
totals.legend()
fig.tight_layout()
plt.show()
