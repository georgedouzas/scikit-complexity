"""
# Evolutionary optimisers

Differential evolution, particle swarm optimisation, and simulated annealing search the Rastrigin function, whose many
local minima surround the global minimum 0. Each is a process on a population of candidates, moved one generation per
step, and the best value it has seen falls with time.
"""

# %%
# The default model runs differential evolution with 40 candidates.

import matplotlib.pyplot as plt

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create, describe
from skcomplex.models.computer_science.evolutionary_computation import evolutionary_optimisers

MODEL = 'evolutionary_optimisers'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).prepare().run(until=100)

# %%
# The best value against the generation
# -------------------------------------
# An `Experiment` runs each population optimiser over the same seeds, and annealing, which cools geometrically, reports
# its best value through the observable of the catalog.

fig, ax = plt.subplots(figsize=(7, 4))
bounds = {'algorithm': ['differential-evolution', 'particle-swarm'], 'random_state': [1]}
experiment = Experiment(evolutionary_optimisers, GridDesign(), bounds, until=100, outputs=['search.best'])
for name, best in zip(
    bounds['algorithm'],
    experiment.prepare().run().result_.series('search.best'),
    strict=True,
):
    ax.semilogy(best[1:] + 1e-12, label=name)
annealing = evolutionary_optimisers(algorithm='annealing', random_state=1)
outputs = ['tracker/record.best']
observables = describe(MODEL).observables_of(annealing)
result = Experiment(annealing, until=100, outputs=outputs, observables=observables).prepare().run().result_
ax.semilogy(result.series('tracker/record.best')[0, 1:] + 1e-12, label='annealing')
ax.set_xlabel('generation')
ax.set_ylabel('best value')
ax.set_title('Search on the Rastrigin function')
ax.legend()
fig.tight_layout()
plt.show()
