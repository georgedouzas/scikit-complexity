"""
# Trimorphic multilevel selection

Groups hold cooperators, defectors, and altruistic punishers, so their compositions are points of a triangle, the
2-simplex of three frequencies. Within groups, defectors outcompete cooperators and punishers, and punishers pay to
punish defectors. Between groups, a group wins a pairwise conflict with a probability that grows with the difference of
the fractions of non-defectors. The image shows the density of groups on the simplex, with the fraction of cooperators
down and the fraction of defectors across, empty beyond the slanted face.
"""

# %%
# The system is created by name from the catalog on a simplex of 30 cells per axis, with a benefit of 3, a punishment
# of 0.5, a per-interaction cost of 0.1, and between-group conflict of strength `lambda = 4`. The density starts
# uniform. The observer saves every frame of 1 up to `t = 30`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'trimorphic_multilevel_selection'
CELLS = 30

system = create_system(SYSTEM, cells=CELLS)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='groups/f', interval=1)
simulation = Simulation(system, dt=1.0, solver=RK4Solver(dt=0.01), observers=[observer]).prepare().run(until=30)

# %%
# Where the groups go
# -------------------
# Between-group conflict over the fraction of non-defectors moves the density away from the defector corner, and the
# punishers, who keep defectors out within groups, collect most of it. The fraction of non-defectors should rise under
# both victory rules, the fraction of non-defectors (`'defectors'`) and the average payoff (`'payoff'`), while the
# integral of the density stays one.

space = system.elements[0].space.prepare()
weights = space.volumes_[0] / CELLS**2
cooperators, defectors = np.meshgrid(space.coordinates_['x'], space.coordinates_['y'], indexing='ij')
final = simulation.trajectory_.values('groups.f')[-1].reshape(CELLS, CELLS)
fig, (simplex, shares) = plt.subplots(1, 2, figsize=(11, 4.5))
image = simplex.imshow(np.where(weights > 0, final, np.nan).T, origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
simplex.set_xlabel('fraction of cooperators x')
simplex.set_ylabel('fraction of defectors y')
simplex.set_title('Density of groups at t = 30')
fig.colorbar(image, ax=simplex)
for victory in ('defectors', 'payoff'):
    model = create_system(SYSTEM, cells=CELLS, victory=victory)
    run = Simulation(model, dt=1.0, solver=RK4Solver(dt=0.01)).prepare().run(until=30).trajectory_
    values = run.values('groups.f').reshape(-1, CELLS, CELLS)
    shares.plot(run.times, 1 - (values * weights * defectors).sum(axis=(1, 2)), label=f'non-defectors, {victory}')
    shares.plot(
        run.times,
        ((1 - cooperators - defectors) * values * weights).sum(axis=(1, 2)),
        '--',
        label=f'punishers, {victory}',
    )
shares.set_xlabel('time t')
shares.set_ylabel('mean fraction in a group')
shares.set_title('Conflict favours non-defectors')
shares.legend()
fig.tight_layout()
plt.show()
