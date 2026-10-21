"""
# Meadow capacity

Cows graze a meadow for a summer of 25 weeks. Each cow eats the grass of its patch and moves to the nearest free patch
with enough grass, and a cow that finds none dies. A small herd lives through the summer, while a herd too large for
the meadow grazes it bare and dies out within weeks, which brackets how many cows the meadow can carry.
"""

# %%
# The model is created by name from the catalog with 80 cows on a meadow of 40 by 40 patches. The observer saves every
# week with the default blueprint, whose primary view shows the grass and the cows.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'meadow_capacity'

system = create(MODEL, cows=80, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=25)

# %%
# Herds of different sizes
# ------------------------
# Without regrowth the meadow feeds `width * height / 25 = 64` cows for the summer. With regrowth herds of up to 80
# cows should all survive the summer, above that bound, while larger herds should collapse once the grass is gone.

fig, ax = plt.subplots(figsize=(8, 4))
for herd in [40, 80, 160, 320]:
    trajectory = Simulation(create(MODEL, cows=herd, random_state=0), dt=1, random_state=0).run(until=25).trajectory_
    ax.plot(trajectory.times, trajectory['cows.count'], 'o-', markersize=3, label=f'{herd} cows')
ax.axhline(64, color='black', linestyle='--', label='capacity without regrowth')
ax.set_xlabel('week')
ax.set_ylabel('cows alive')
ax.set_title('Up to 80 cows live through the summer, larger herds collapse')
ax.legend()
plt.show()
