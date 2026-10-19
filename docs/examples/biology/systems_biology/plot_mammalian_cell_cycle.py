"""
# Boolean model of the mammalian cell cycle

With the growth signal CycD on, the ten proteins of the cell cycle leave the quiescent state and run through a cycle
of period 7, in which the cyclins E, A, and B rise and fall in turn.
"""

# %%
# The model is created by name from the catalog and started from the quiescent state, Rb, p27, and Cdh1 on, with
# CycD switched on. The observer saves every step with the default blueprint, whose primary view is the time series of
# the proteins.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'mammalian_cell_cycle'
PROTEINS = ('CycD', 'Rb', 'E2F', 'CycE', 'CycA', 'p27', 'Cdc20', 'Cdh1', 'Ubc', 'CycB')

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=21)

# %%
# The cycle of the cyclins
# ------------------------
# The raster should show the same seven-step pattern three times after the first two steps: E2F turns on CycE, CycA
# follows, CycB and Cdc20 turn on, and Cdh1 returns as CycB is degraded. Without CycD, the second panel, the quiescent
# state stays fixed.

cycling = simulation.trajectory_
resting = Simulation(create(MODEL, active=('Rb', 'p27', 'Cdh1')), dt=1).run(until=21).trajectory_
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, trajectory, title in zip(
    axes,
    (cycling, resting),
    ('CycD on: a cycle of period 7', 'CycD off: quiescence'),
    strict=False,
):
    raster = [trajectory[f'{name}.x'] for name in PROTEINS]
    ax.imshow(raster, aspect='auto', cmap='Greys', interpolation='nearest')
    ax.set_yticks(range(len(PROTEINS)), PROTEINS)
    ax.set_xlabel('step')
    ax.set_title(title)
fig.tight_layout()
plt.show()
