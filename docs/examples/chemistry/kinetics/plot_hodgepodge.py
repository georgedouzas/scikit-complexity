"""
# Hodgepodge machine

Healthy cells get infected by ill neighbours, infected cells grow more ill, and ill cells heal at once. From random
levels the grid organizes into waves of infection that wind into rotating spirals, like the chemical waves of the
Belousov-Zhabotinsky reaction.
"""

# %%
# The model is created by name from the catalog on a 100 by 100 grid with 100 levels of illness, `k1 = 2`, `k2 = 3`,
# and the default infection rate `g = 34`, from random levels, and runs for 600 steps. The default blueprint shows the
# levels on the grid.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import autocorrelation, fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hodgepodge'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=3)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=600)

# %%
# Every cell repeats one cycle
# ----------------------------
# Every cell goes through the same cycle of infection, growing illness, and healing, so once the waves are established
# the level of each cell should repeat with one period. The autocorrelation of the level of a cell, averaged over all
# cells, should then peak at that period and its multiples.

fields = fields_of(simulation.trajectory_, 'cells.level')[300:]
levels = fields.reshape(len(fields), -1)
correlations = np.mean([autocorrelation(levels[:, cell], max_lag=60) for cell in range(0, levels.shape[1], 7)], axis=0)
period = np.argmax(correlations[5:]) + 5
fig, (cell, correlation) = plt.subplots(1, 2, figsize=(10, 4))
for index in (2020, 5070, 8030):
    cell.plot(simulation.trajectory_.times[300:400], levels[:100, index], linewidth=1, label=f'cell {index}')
cell.set_xlabel('step')
cell.set_ylabel('level of illness')
cell.set_title('Cells cycle through infection and healing')
cell.legend()
correlation.plot(correlations)
correlation.axvline(period, color='black', linestyle='--', label=f'period {period} steps')
correlation.axhline(0.0, color='gray', linewidth=0.8)
correlation.set_xlabel('lag (steps)')
correlation.set_ylabel('autocorrelation of the level')
correlation.set_title('Mean autocorrelation over cells')
correlation.legend()
fig.tight_layout()
plt.show()
