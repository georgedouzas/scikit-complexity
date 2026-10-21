"""
# Fibonacci rabbits

Every pair of rabbits older than one month breeds a new pair each month, so the number of pairs runs through the
Fibonacci numbers 1, 1, 2, 3, 5, 8, ... and grows by the golden ratio every month.
"""

# %%
# The model is created by name from the catalog from one newborn pair and run for 20 months. The observer saves
# every month with the default blueprint, whose primary view is the time series of the pairs.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fibonacci_rabbits'

system = create(MODEL, pairs=1.0, previous=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=20)

# %%
# The golden ratio
# ----------------
# The ratio of successive Fibonacci numbers tends to the golden ratio `(1 + sqrt 5) / 2 ≈ 1.618`, alternately from
# above and below. The simulated ratio should converge to it within a few months.

pairs = simulation.trajectory_['rabbits.pairs']
months = simulation.trajectory_.times
golden = (1 + np.sqrt(5)) / 2
fig, (counts, ratio) = plt.subplots(1, 2, figsize=(10, 4))
counts.semilogy(months, pairs, 'o-')
counts.set_xlabel('month')
counts.set_ylabel('pairs (log scale)')
counts.set_title('Pairs grow geometrically')
ratio.plot(months[2:], pairs[2:] / pairs[1:-1], 'o-', label='simulated ratio')
ratio.axhline(golden, color='black', linestyle='--', label='golden ratio')
ratio.set_xlabel('month')
ratio.set_ylabel('pairs this month / last month')
ratio.set_title(f'Ratio after {months[-1]:.0f} months: {pairs[-1] / pairs[-2]:.6f}')
ratio.legend()
fig.tight_layout()
plt.show()
