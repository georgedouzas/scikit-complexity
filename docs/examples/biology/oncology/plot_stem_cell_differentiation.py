"""
# Stem-cell differentiation

Under continuous therapy the normal lineage overshoots, collapses, and recovers, and the eight compartments settle on
a steady state in which the mutation of normal progenitors feeds a small malignant population.
"""

# %%
# The system is built from its class with the book's parameters and initial cells, a mutation rate drawn once with the
# seed 0, and continuous therapy, and runs for 2000 time units. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the eight compartments.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import StemCellDifferentiationSystem

SYSTEM = 'stem_cell_differentiation'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(StemCellDifferentiationSystem(seed=0), observers=[observer])

# %%
# Mutation feeds the malignant lineage
# ------------------------------------
# The malignant progenitors barely renew themselves, so at the steady state they are fed by the mutation `m1 NPr` of
# the normal progenitors. Each seed draws another mutation rate in `[m0, 2 m0)`, and the malignant differentiated cells
# at the end should grow with it.

fig, (lineages, mutation) = plt.subplots(1, 2, figsize=(10, 4))
for state, name in [('NPr', 'normal progenitors'), ('ND', 'normal differentiated'), ('AD', 'malignant differentiated')]:
    lineages.semilogy(trajectory.times, trajectory.values(f'tissue.{state}').clip(1e-3), label=name)
lineages.set_xlabel('time')
lineages.set_ylabel('cells')
lineages.set_title('The lineages settle under therapy')
lineages.legend()
rates, finals = [], []
for seed in range(8):
    run = StemCellDifferentiationSystem(seed=seed).prepare().system_
    rates.append(run.elements[0].m1)
    finals.append(simulate(run, dt=10.0, until=2000).values('tissue.AD')[-1])
mutation.plot(rates, finals, 'o')
mutation.set_xlabel('mutation rate m1')
mutation.set_ylabel('final malignant differentiated cells')
mutation.set_title('More mutation, more malignant cells')
fig.tight_layout()
plt.show()
