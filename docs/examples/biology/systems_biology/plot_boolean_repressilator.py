"""
# Boolean repressilator

Three genes that repress each other in a ring switch on and off in turn, so each one oscillates with a period of six
steps.
"""

# %%
# The example builds `BooleanRepressilatorSystem` and starts it with LacI on. The observer saves every step with the
# default blueprint, whose primary view is the time series of LacI, TetR, and cI.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import BooleanRepressilatorSystem

SYSTEM = 'boolean_repressilator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(BooleanRepressilatorSystem(x=(1, 0, 0)), observers=[observer])

# %%
# Delayed repression
# ------------------
# A repression delayed by `d` steps passes through `d` relay nodes, so the ring holds `3 (d + 1)` nodes with three
# inversions. The oscillation should survive the delay, and its period should grow to `6 (d + 1)` steps.

fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
for ax, delay in zip(axes, (0, 1, 2), strict=False):
    trajectory = simulate(BooleanRepressilatorSystem(x=(1, 0, 0), delay=delay), until=48)
    for offset, name in enumerate(('LacI', 'TetR', 'cI')):
        ax.step(trajectory.times, trajectory.get_values(f'{name}.x') + 1.5 * offset, where='post', label=name)
    ax.set_yticks([0.5, 2.0, 3.5], ['LacI', 'TetR', 'cI'])
    ax.set_title(f'delay {delay}: period {6 * (delay + 1)}', fontsize='medium')
axes[-1].set_xlabel('step')
fig.suptitle('The oscillation survives a delayed repression')
fig.tight_layout()
plt.show()
