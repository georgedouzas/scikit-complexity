"""
# Run and tumble

Bacteria such as E. coli swim in straight runs interrupted by tumbles, short events that send them off in a random
new direction. With tumbles at rate `0.1` per step a bacterium keeps its heading for ten steps on average, so over
short times it moves ballistically and over long times it diffuses. The view shows 100 bacteria in a periodic
square, each heading in its own direction.
"""

# %%
# The model is created by name from the catalog with speed `0.1` per step and tumble rate `0.1` in a square of side
# 10. The observer saves every step with the default blueprint, whose primary view shows the bacteria.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'run_and_tumble'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=299)

# %%
# Persistent random walk
# ----------------------
# A run keeps its heading for one more step with probability `q = exp(-rate)`, so the headings of steps `k` apart
# correlate as `q^k` and the mean squared displacement after `n` steps of length `v` is
# `v^2 (n (1 + q) / (1 - q) - 2 q (1 - q^n) / (1 - q)^2)`: ballistic, `v^2 n^2`, for few steps and diffusive with
# `D = v^2 (1 + q) / (4 (1 - q))` per step after many. Bacteria in a box far larger than their excursions should
# follow it for two tumble rates.

fig, ax = plt.subplots(figsize=(7, 4))
steps = np.arange(1, 301)
for rate, color in ((0.1, 'tab:blue'), (1.0, 'tab:orange')):
    run = Simulation(create(MODEL, n_agents=500, tumble_rate=rate, size=1e5, random_state=1), dt=1, random_state=1)
    run.run(until=300)
    positions = np.stack([run.trajectory_.agents('bacteria', frame).locations for frame in range(301)])
    squared = np.mean(np.sum((positions[1:] - positions[0]) ** 2, axis=-1), axis=1)
    q = np.exp(-rate)
    exact = 0.01 * (steps * (1 + q) / (1 - q) - 2 * q * (1 - q**steps) / (1 - q) ** 2)
    ax.loglog(steps, squared, 'o', markersize=3, color=color, label=f'simulation, rate = {rate}')
    ax.loglog(steps, exact, color=color, label=f'persistent walk, q = {q:.2f}')
ax.set_xlabel('steps n')
ax.set_ylabel('mean squared displacement')
ax.set_title('Runs are ballistic, many tumbles make diffusion')
ax.legend()
fig.tight_layout()
plt.show()
