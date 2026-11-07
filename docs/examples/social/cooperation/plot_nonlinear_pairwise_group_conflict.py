"""
# Pairwise group conflict

Groups of defectors and altruistic punishers compete at two levels. Within a group, defectors earn more and spread,
while between groups, a group with more punishers earns more and tends to win pairwise conflicts, copying itself over
the loser. The density of groups over their fraction of punishers is advected by the first process and reshaped by the
second, an integral against the probability that one group defeats another, which is no function of the difference of
their compositions. The image shows the density as one row of cells, from groups of defectors on the left to groups
of punishers on the right.
"""

# %%
# The system is created by name from the catalog with the group Fermi rule of sensitivity 2, a punishment of 0.5, and
# a per-interaction cost of 0.1, on 100 cells of the fraction of punishers. Between-group conflict is strong, with
# `lambda = 2`. The observer saves every frame of 0.5 up to `t = 15`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'nonlinear_pairwise_group_conflict'
CELLS = 100

system = create_system(SYSTEM, cells=CELLS, group_strength=2.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='groups/f', interval=1)
simulation = Simulation(system, dt=0.5, solver=RK4Solver(dt=0.01), observers=[observer]).prepare().run(until=15)

# %%
# Strength of between-group conflict
# ----------------------------------
# Weak conflict, `lambda = 0.1`, cannot hold the advantage of defectors within groups, and the density collapses onto
# the groups of defectors. Strong conflict, `lambda = 2`, settles on a density that holds many groups of mostly
# punishers. The three victory rules differ in how a payoff difference becomes a chance to win, and the mean fraction
# of punishers they reach rises with the strength of conflict.

centres = (np.arange(CELLS) + 0.5) / CELLS
density = simulation.trajectory_.values('groups.f').reshape(-1, CELLS)
times = simulation.trajectory_.times
strengths = np.linspace(0.1, 3.0, 8)
fig, (profiles, means) = plt.subplots(1, 2, figsize=(11, 4))
for frame in range(0, len(times), 3):
    profiles.plot(centres, density[frame], color=plt.cm.viridis(frame / len(times)))
profiles.set_xlabel('fraction of punishers z')
profiles.set_ylabel('density of groups f(z)')
profiles.set_title('Strong conflict sustains punishers')
for victory in ('fermi', 'local', 'tullock'):
    finals = []
    for strength in strengths:
        model = create_system(SYSTEM, cells=50, victory=victory, group_strength=strength, exponent=2.0)
        run = Simulation(model, dt=15.0, solver=RK4Solver(dt=0.02)).prepare().run(until=15).trajectory_
        finals.append(run.values('groups.f')[-1].reshape(50) @ ((np.arange(50) + 0.5) / 50) / 50)
    means.plot(strengths, finals, 'o-', label=victory)
means.set_xlabel('strength of between-group conflict lambda')
means.set_ylabel('mean fraction of punishers at t = 15')
means.set_title('Victory rules')
means.legend()
fig.tight_layout()
plt.show()
