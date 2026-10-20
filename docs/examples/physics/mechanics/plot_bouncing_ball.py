"""
# Bouncing ball

A ball dropped on the floor bounces lower and lower, each bounce shorter than the last by the coefficient of
restitution, so infinitely many bounces fit before a finite time, the Zeno time, when the ball comes to rest.
"""

# %%
# The ball is dropped from rest at height 1 with `e = 0.8`, so the Zeno time is `sqrt(2 h / g) (1 + e) / (1 - e)`,
# about 4.06. The run stops shortly before it. The default blueprint shows the height and the velocity, which jumps
# from negative to positive at every impact.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bouncing_ball'

system = create(MODEL, h=1.0, v=0.0, e=0.8)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=3.8)

# %%
# Impact times
# ------------
# The first impact happens at `t0 = sqrt(2 h / g)`, and each flight after an impact lasts `e` times the previous one,
# so the `n`-th impact happens at `t0 (1 + 2 e (1 - e^n) / (1 - e))`. The impacts located by the event detection
# should lie on this geometric series, which converges to the Zeno time.

impacts = np.array([event.time for event in simulation.trajectory_.events])
first = np.sqrt(2 / 9.81)
counts = np.arange(len(impacts))
predicted = first * (1 + 2 * 0.8 * (1 - 0.8**counts) / (1 - 0.8))
zeno = first * 1.8 / 0.2
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(counts, impacts, 'o', label='simulated impacts')
ax.plot(counts, predicted, color='black', label='geometric series')
ax.axhline(zeno, linestyle='--', color='grey', label=f'Zeno time {zeno:.2f}')
ax.set_xlabel('impact number n')
ax.set_ylabel('time of the impact')
ax.set_title('Impacts accumulate at the Zeno time')
ax.legend()
plt.show()
