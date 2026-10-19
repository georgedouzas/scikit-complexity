"""
# Boolean repressilator

Three genes that repress each other in a ring switch on and off in turn, so each one oscillates with a period of six
steps.
"""

# %%
# The model is created by name from the catalog and started with LacI on. The observer saves every step with the
# default blueprint, whose primary view is the time series of LacI, TetR, and cI.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'boolean_repressilator'

system = create(MODEL, x=(1, 0, 0))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=24)

# %%
# Delayed repression
# ------------------
# A repression delayed by `d` steps passes through `d` relay nodes, so the ring holds `3 (d + 1)` nodes with three
# inversions. The oscillation should survive the delay, and its period should grow to `6 (d + 1)` steps.

fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
for ax, delay in zip(axes, (0, 1, 2), strict=False):
    trajectory = Simulation(create(MODEL, x=(1, 0, 0), delay=delay), dt=1).run(until=48).trajectory_
    for offset, name in enumerate(('LacI', 'TetR', 'cI')):
        ax.step(trajectory.times, trajectory[f'{name}.x'] + 1.5 * offset, where='post', label=name)
    ax.set_yticks([0.5, 2.0, 3.5], ['LacI', 'TetR', 'cI'])
    ax.set_title(f'delay {delay}: period {6 * (delay + 1)}', fontsize='medium')
axes[-1].set_xlabel('step')
fig.suptitle('The oscillation survives a delayed repression')
fig.tight_layout()
plt.show()
