"""
# Mutualism

Two species that feed each other both end above the carrying capacity they would reach alone. The benefit saturates
with the handling time of the partner's resource, so the populations settle instead of growing without bound.
"""

# %%
# The example builds `MutualismSystem` with its default saturating mutualism and runs it from small populations. The
# observer saves every second frame with the default blueprint, whose primary view is the time series of both
# populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import MutualismSystem

SYSTEM = 'mutualism'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(MutualismSystem(first=5.0, second=2.0), observers=[observer], until=30)

# %%
# With and without the partner
# ----------------------------
# Running each species alone, with the other set to zero, gives logistic growth to its carrying capacity of 50. With
# the partner, both should settle clearly above that line.

alone = simulate(MutualismSystem(first=5.0, second=0.0), until=30)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(trajectory.times, trajectory.get_values('first.x'), label='first, with partner')
ax.plot(trajectory.times, trajectory.get_values('second.x'), label='second, with partner')
ax.plot(alone.times, alone.get_values('first.x'), '--', color='black', label='first, alone')
ax.axhline(50.0, color='grey', linestyle=':', label='carrying capacity')
ax.set_xlabel('time')
ax.set_ylabel('population')
ax.set_title(
    f'Mutualists settle at {trajectory.get_values("first.x")[-1]:.0f} and '
    f'{trajectory.get_values("second.x")[-1]:.0f}, above 50',
)
ax.legend()
plt.show()
