"""
# Boolean system of the mammalian cell cycle

With the growth signal CycD on, the ten proteins of the cell cycle leave the quiescent state and run through a cycle
of period 7, in which the cyclins E, A, and B rise and fall in turn.
"""

# %%
# The system is built from its class and started from the quiescent state, Rb, p27, and Cdh1 on, with CycD switched
# on. The observer saves every step with the default blueprint, whose primary view is the time series of the proteins.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import MammalianCellCycleSystem

SYSTEM = 'mammalian_cell_cycle'
PROTEINS = ('CycD', 'Rb', 'E2F', 'CycE', 'CycA', 'p27', 'Cdc20', 'Cdh1', 'Ubc', 'CycB')

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(MammalianCellCycleSystem(), observers=[observer])

# %%
# The cycle of the cyclins
# ------------------------
# The raster should show the same seven-step pattern three times after the first two steps: E2F turns on CycE, CycA
# follows, CycB and Cdc20 turn on, and Cdh1 returns as CycB is degraded. Without CycD, the second panel, the quiescent
# state stays fixed.

cycling = trajectory
resting = simulate(MammalianCellCycleSystem(active=('Rb', 'p27', 'Cdh1')))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, trajectory, title in zip(
    axes,
    (cycling, resting),
    ('CycD on: a cycle of period 7', 'CycD off: quiescence'),
    strict=False,
):
    raster = [trajectory.get_values(f'{name}.x') for name in PROTEINS]
    ax.imshow(raster, aspect='auto', cmap='Greys', interpolation='nearest')
    ax.set_yticks(range(len(PROTEINS)), PROTEINS)
    ax.set_xlabel('step')
    ax.set_title(title)
fig.tight_layout()
plt.show()
