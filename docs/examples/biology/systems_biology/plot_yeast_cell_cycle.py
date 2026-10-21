"""
# Yeast cell cycle

Eleven proteins of budding yeast, each switched on or off by the proteins that activate and inhibit it, run through
the phases of the cell cycle in order once the cell-size signal Cln3 appears: G1, the start of DNA synthesis, mitosis,
and back to the stationary G1 state.
"""

# %%
# The model is created by name from the catalog in the G1 state with Cln3 on. The observer saves every step with the
# default blueprint, whose primary view is the time series of the protein values.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'yeast_cell_cycle'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=15)

# %%
# The sequence of the cycle
# -------------------------
# Li et al. found that the network passes through the states of the biological cycle in order: the SBF and MBF
# transcription factors, the G1 cyclins Cln1,2, the S-phase cyclins Clb5,6, the mitotic cyclins Clb1,2 with Mcm1,
# then Cdc20 and Swi5, ending with Cdh1 and Sic1 on, the stationary G1 state.

proteins = ['Cln3', 'MBF', 'SBF', 'Cln1_2', 'Cdh1', 'Swi5', 'Cdc20', 'Clb5_6', 'Sic1', 'Clb1_2', 'Mcm1']
trajectory = simulation.trajectory_
states = np.column_stack([trajectory[f'{protein}.x'] for protein in proteins])
changes = np.nonzero(np.any(states[1:] != states[:-1], axis=1))[0]
settled = changes[-1] + 1
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.imshow(states.T, aspect='auto', cmap='Greens', interpolation='nearest')
ax.set_yticks(range(len(proteins)), proteins)
ax.set_xlabel('step')
ax.set_ylabel('protein')
ax.axvline(settled - 0.5, color='tab:red', label='stationary G1 state')
ax.legend(loc='upper right')
ax.set_title(f'The proteins switch in the order of the cycle and stay at G1 from step {settled}')
plt.show()
