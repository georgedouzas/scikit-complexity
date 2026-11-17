"""
# Evolutionary optimisers

Differential evolution, particle swarm optimisation, and simulated annealing search the Rastrigin function, whose many
local minima surround the global minimum 0. Each is a process on a population of candidates, moved one generation per
step, and the best value it has seen falls with time.
"""

# %%
# The default system runs differential evolution with 40 candidates.

import matplotlib.pyplot as plt

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.computer_science.evolutionary_computation import (
    EvolutionaryOptimisersSystem,
)

SYSTEM = 'evolutionary_optimisers'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = EvolutionaryOptimisersSystem().prepare().simulate(observers=[observer], random_state=0)

# %%
# The best value against the generation
# -------------------------------------
# `sweep` runs every optimiser from the same initial candidates, with a random stream of its own spawned from
# `random_state`, so the figure is the same on every build. Annealing, which cools geometrically, reports its best value
# through the tracker of the lowest energy its candidates reach.


def best(trajectory):
    """Return the best value of every generation after the first frame."""
    path = 'search.best' if 'search.best' in trajectory.names else 'tracker/record.best'
    return trajectory.get_values(path)[1:]


algorithms = ['differential-evolution', 'particle-swarm', 'annealing']
optimisers = EvolutionaryOptimisersSystem(random_state=1)
runs = sweep(optimisers, {'algorithm': algorithms}, until=100, outputs=[best], random_state=1)
fig, ax = plt.subplots(figsize=(7, 4))
for row in runs.rows:
    ax.semilogy(row['best'] + 1e-12, label=row['algorithm'])
ax.set_xlabel('generation')
ax.set_ylabel('best value')
ax.set_title('Search on the Rastrigin function')
ax.legend()
fig.tight_layout()
plt.show()
