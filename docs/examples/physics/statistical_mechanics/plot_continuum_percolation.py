"""
# Continuum percolation

Disks of radius 1 at uniform positions overlap more as their filling factor grows, and near a filling of 1.13 a cluster
of touching disks first joins the opposite sides of the square. The transition sharpens with the size of the square, so
the spanning probabilities of two sizes cross near the threshold.
"""

# %%
# The disks are drawn once from the seed, as a static frame of points in the square.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import bind_analysis, clusters
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems.physics.statistical_mechanics import ContinuumPercolationSystem

SYSTEM = 'continuum_percolation'

system = ContinuumPercolationSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=1.0, observers=[observer])

# %%
# Spanning probability against the filling factor
# -----------------------------------------------
# An `Experiment` sweeps the filling factor and the seed for two sizes, and `clusters` reads the contact clusters of
# every run.

fillings = [0.8, 1.0, 1.15, 1.3, 1.5]
seeds = list(range(10))
fig, ax = plt.subplots(figsize=(7, 4))
for side in (20.0, 50.0):
    experiment = Experiment(
        lambda **parameters: ContinuumPercolationSystem(**parameters).prepare().system_,
        GridDesign(),
        {'side': [side], 'filling': fillings, 'random_state': seeds},
        until=1.0,
        analyses=[bind_analysis(clusters, kind='percolation', radius=1.0)],
        outputs=[],
    )
    spans = experiment.prepare().run().result_.column('clusters.spanning').astype(float)
    ax.plot(fillings, spans.reshape(len(fillings), len(seeds)).mean(axis=1), 'o-', label=f'side {side:.0f}')
ax.axvline(1.128, color='black', linestyle='--', label='threshold 1.128')
ax.set_xlabel('filling factor')
ax.set_ylabel('spanning probability')
ax.set_title('The transition sharpens with the size')
ax.legend()
fig.tight_layout()
plt.show()
