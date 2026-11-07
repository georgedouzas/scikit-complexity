"""
# Winner-take-all

Units that grow on their input and inhibit each other compete until a single one remains: with strong inhibition the
unit with the largest input wins and settles at an activity equal to its input, and all the others fall silent.
"""

# %%
# The system is built from its class with five units whose inputs differ by at most 20% and the inhibition `beta = 2`.
# The observer saves every frame with the default blueprint, whose primary view is the time series of the activities.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.neuroscience import WinnersTakeAllSystem

SYSTEM = 'winners_take_all'

inputs = (1.0, 1.1, 0.9, 1.05, 0.95)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(WinnersTakeAllSystem(inputs=inputs, beta=2.0), observers=[observer])

# %%
# One winner or several
# ---------------------
# Above `beta = 1` only the unit with the largest input, 1.1, should stay active, at activity 1.1. Below it, several
# units should share the activity.

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, beta in zip(axes, [2.0, 0.5], strict=True):
    trajectory = simulate(WinnersTakeAllSystem(inputs=inputs, beta=beta))
    for unit, value in enumerate(inputs):
        ax.plot(trajectory.times, trajectory.values(f'unit_{unit}.x'), label=f'input {value}')
    ax.set_xlabel('time')
    ax.set_title(f'beta = {beta}')
axes[0].axhline(max(inputs), color='black', linestyle=':', label='largest input')
axes[0].set_ylabel('activity')
axes[0].legend()
fig.suptitle('Strong inhibition leaves one winner, weak inhibition several')
fig.tight_layout()
plt.show()
