"""
# Three-gene Boolean network

Three genes updated synchronously by AND, OR, and NOT run from the state 100 into a cycle of period three, while the
state 011 is a point attractor.
"""

# %%
# The model is created by name from the catalog and started from the state `100`, `g1` on. The observer saves every
# step with the default blueprint, whose primary view is the time series of the three genes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'three_gene_boolean_network'

system = create(MODEL, x=(1, 0, 0))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=10)

# %%
# Attractors from every state
# ---------------------------
# Running the network from each of its eight states should reproduce the state transition map of the book: the
# states `000`, `011`, `101` reach the point attractor `011`, and all the others fall on the cycle `001 -> 010 -> 111`.

fig, ax = plt.subplots(figsize=(7, 4))
for start in range(8):
    bits = tuple(int(bit) for bit in format(start, '03b'))
    run = Simulation(create(MODEL, x=bits), dt=1).run(until=8).trajectory_
    code = 4 * run['g1.x'] + 2 * run['g2.x'] + run['g3.x']
    ax.plot(run.times, code + 0.06 * (start - 3.5), 'o-', label=format(start, '03b'))
ax.set_yticks(np.arange(8), [format(code, '03b') for code in range(8)])
ax.set_xlabel('step')
ax.set_ylabel('state g1 g2 g3')
ax.set_title('Every state ends on 011 or on the cycle 001, 010, 111')
ax.legend(title='start', ncol=2, fontsize='small')
fig.tight_layout()
plt.show()
