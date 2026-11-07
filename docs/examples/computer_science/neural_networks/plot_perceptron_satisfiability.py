"""
# Binary perceptron satisfiability

A perceptron of `N` weights `+-1` must give the right output on random patterns, revealed one at a time. Each pattern is
a cardinality constraint, and an incremental SAT solver decides whether weights that satisfy them all still exist. The
solutions run out near the critical load `p / N = 0.83`.
"""

# %%
# The system reveals one pattern per step, and logs the weights the solver finds until none exist.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import sweep
from skcomplex.systems.computer_science.neural_networks import PerceptronSatisfiabilitySystem

SYSTEM = 'perceptron_satisfiability'

system = PerceptronSatisfiabilitySystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# The satisfiable share against the load
# --------------------------------------
# The patterns are quenched, so a `sweep` varies the `random_state` of the factory, and the share of satisfiable
# draws falls ever more sharply as the system grows.

fig, ax = plt.subplots(figsize=(7, 4))
for n_weights in (15, 21):
    bounds = {'n_weights': [n_weights], 'random_state': list(range(20))}
    outputs = ['decide.satisfiable']
    result = sweep(PerceptronSatisfiabilitySystem, bounds, until=31, outputs=outputs)
    shares = result.series('decide.satisfiable').mean(axis=0)
    ax.plot(np.arange(1, shares.size) / n_weights, shares[1:], 'o-', markersize=3, label=f'N = {n_weights}')
ax.axvline(0.83, color='black', linestyle='--', label='critical load 0.83')
ax.set_xlabel('load p / N')
ax.set_ylabel('share of satisfiable draws')
ax.set_title('The solutions run out near the critical load')
ax.legend()
fig.tight_layout()
plt.show()
