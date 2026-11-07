"""
# Two-level replicator equation

Groups of defectors and altruistic punishers compete at two levels. Within groups defectors outgrow punishers, and
between groups those with more punishers win pairwise conflicts and copy themselves over the losers. With an additively
separable victory probability the pairwise conflict reduces to a two-level replicator equation, whose groups keep
punishers only when between-group competition is strong enough.
"""

# %%
# The density of groups over the fraction `z` of punishers starts uniform, with `b = 2`, `c = 1`, `p = 0.5`, and
# `k = 0.5`, and the victory probability depends on the difference of the fractions of punishers. The default
# blueprint shows the density.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'two_level_replicator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), dt=1.0, solver=RK4Solver(dt=0.01), observers=[observer]).prepare()
simulation.run(until=20)

# %%
# The threshold of group selection
# --------------------------------
# Below `lambda* = c + q - p = 0.5` the punishers should vanish, and above it their mean fraction should tend to
# `1 - (c + q - p) / lambda`, Eq. 5.4 of Cooney.

centres = (np.arange(100) + 0.5) / 100
strengths = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
means = []
for strength in strengths:
    run = Simulation(create_system(SYSTEM, group_strength=float(strength)), dt=20.0, solver=RK4Solver(dt=0.01))
    final = run.prepare().run(until=20).trajectory_.values('groups.f')[-1].ravel()
    means.append(np.sum(centres * final) / np.sum(final))
density = simulation.trajectory_.values('groups.f')[-1].ravel()
fig, (steady, threshold) = plt.subplots(1, 2, figsize=(10, 4))
steady.plot(centres, density)
steady.set_xlabel('fraction of punishers z')
steady.set_ylabel('density f(z)')
steady.set_title('Steady density at lambda = 2')
threshold.plot(strengths, means, 'o', label='simulation')
grid = np.linspace(0.25, 4, 100)
threshold.plot(grid, np.where(grid > 0.5, 1 - 0.5 / grid, 0.0), color='black', linestyle=':', label='Eq. 5.4')
threshold.set_xlabel('strength of group competition lambda')
threshold.set_ylabel('mean fraction of punishers')
threshold.set_title('Punishers above lambda* = 0.5')
threshold.legend()
fig.tight_layout()
plt.show()
