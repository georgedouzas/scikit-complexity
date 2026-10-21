"""
# Michaelis-Menten

The enzyme binds the substrate into a complex within a few time units, then the complex turns the substrate into
product at a nearly steady rate until the substrate runs out and the enzyme is released.
"""

# %%
# The model is created by name from the catalog with 100 enzymes and 300 substrate molecules as rate equations. The
# default blueprint shows the free enzyme `E`, the substrate `S`, the complex `C`, and the product `P` in one time
# series view.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'michaelis_menten'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=50)

# %%
# Rate equations against single molecules
# ---------------------------------------
# With `stochastic=True` the same reactions are random events of Gillespie's algorithm on whole molecules. With
# hundreds of molecules, the stochastic runs should scatter closely around the rate equations, while the totals
# `E + C` of enzyme and `S + C + P` of substrate stay exactly conserved in both.

trajectory = simulation.trajectory_
fig, (curves, totals) = plt.subplots(1, 2, figsize=(10, 4))
for seed in range(5):
    jumps = Simulation(create(MODEL, stochastic=True), dt=0.5, random_state=seed).run(until=50).trajectory_
    for species, color in [('S', 'C0'), ('C', 'C1'), ('P', 'C2')]:
        curves.step(jumps.times, jumps[f'{species}.n'], where='post', color=color, alpha=0.3, linewidth=0.8)
for species, color in [('S', 'C0'), ('C', 'C1'), ('P', 'C2')]:
    curves.plot(trajectory.times, trajectory[f'{species}.n'], color=color, linewidth=2, label=species)
curves.set_xlabel('time')
curves.set_ylabel('molecules')
curves.set_title('Rate equations (thick) and Gillespie runs (thin)')
curves.legend()
enzyme = trajectory['E.n'] + trajectory['C.n']
substrate = trajectory['S.n'] + trajectory['C.n'] + trajectory['P.n']
totals.plot(trajectory.times, enzyme, label='E + C')
totals.plot(trajectory.times, substrate, label='S + C + P')
totals.set_ylim(0, 350)
totals.set_xlabel('time')
totals.set_ylabel('molecules')
totals.set_title('The totals of enzyme and substrate are conserved')
totals.legend()
fig.tight_layout()
plt.show()
