"""
# Continuum percolation

Disks of radius 1 at uniform positions overlap more as their filling factor grows, and near a filling of 1.13 a cluster
of touching disks first joins the opposite sides of the square. The transition sharpens with the size of the square, so
the spanning probabilities of two sizes cross near the threshold.
"""

# %%
# The disks are drawn once from the seed, as a static frame of points in the square.

import matplotlib.pyplot as plt

from skcomplex.analysis import ClusterAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create
from skcomplex.models.physics.statistical_mechanics import continuum_percolation

MODEL = 'continuum_percolation'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer]).check().prepare().run(until=1.0)

# %%
# Spanning probability against the filling factor
# -----------------------------------------------
# An `Experiment` sweeps the filling factor and the seed for two sizes, and `ClusterAnalysis` reads the contact clusters
# of every run.

fillings = [0.8, 1.0, 1.15, 1.3, 1.5]
seeds = list(range(10))
fig, ax = plt.subplots(figsize=(7, 4))
for side in (20.0, 50.0):
    experiment = Experiment(
        continuum_percolation,
        GridDesign(),
        {'side': [side], 'filling': fillings, 'random_state': seeds},
        until=1.0,
        analyses=[ClusterAnalysis('percolation', radius=1.0)],
        outputs=[],
    )
    spans = experiment.check().prepare().run().result_.column('clusteranalysis.spanning').astype(float)
    ax.plot(fillings, spans.reshape(len(fillings), len(seeds)).mean(axis=1), 'o-', label=f'side {side:.0f}')
ax.axvline(1.128, color='black', linestyle='--', label='threshold 1.128')
ax.set_xlabel('filling factor')
ax.set_ylabel('spanning probability')
ax.set_title('The transition sharpens with the size')
ax.legend()
fig.tight_layout()
plt.show()
