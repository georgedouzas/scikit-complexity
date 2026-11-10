"""
# Random Boolean network

A Kauffman network of genes switched on and off by random logical functions of two other genes runs through a
transient and then falls onto a cycle of states that it repeats for ever, an attractor, which Kauffman read as a cell
type.
"""

# %%
# The example builds `RandomBooleanNetworkSystem` with twenty nodes of two inputs each, the critical wiring. The
# observer saves every step with the default blueprint, whose primary view is the time series of the node values.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import RandomBooleanNetworkSystem

SYSTEM = 'random_boolean_network'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(RandomBooleanNetworkSystem(n_nodes=20, k=2, random_state=4), observers=[observer])

# %%
# Transient and attractor
# -----------------------
# The state space is finite and the update deterministic, so the run must revisit a state and repeat from it. The
# first repeated state marks the end of the transient, and the gap to its earlier visit is the length of the cycle.

states = np.column_stack([trajectory.get_values(f'node_{i}.x') for i in range(20)]).astype(int)
seen = {}
for step, state in enumerate(map(tuple, states)):
    if state in seen:
        transient, cycle = seen[state], step - seen[state]
        break
    seen[state] = step
fig, ax = plt.subplots(figsize=(9, 4))
ax.imshow(states.T, aspect='auto', cmap='Greys', interpolation='nearest')
ax.axvline(transient - 0.5, color='tab:red', label=f'end of the transient, step {transient}')
ax.set_xlabel('step')
ax.set_ylabel('node')
ax.set_title(f'The network settles on a cycle of length {cycle}')
ax.legend(loc='upper right')
plt.show()
