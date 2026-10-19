"""
# Doubling map

One hundred points start in an interval of width 0.01 and the doubling map spreads them over the unit interval within
about eight steps. They keep covering it evenly for the whole run, long after floating-point doubling would have sent
every point to 0.
"""

# %%
# The ensemble iterates the map for 100 steps on the fractions `k / q` with the default prime denominator `q`, where the
# model applies `k' = 2k (mod q)` exactly. The default blueprint shows the values of all points as time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'doubling_map'

system = create(MODEL, n_points=100, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=100)

# %%
# Exact orbits against floating-point doubling
# --------------------------------------------
# Doubling a float and dropping its integer part shifts out one of its 53 binary digits, so the same starting points
# iterated with `2x mod 1` in floating point should all reach exactly 0 after about 55 steps. The model's points should
# stay spread, with a flat histogram at the uniform invariant density `p(x) = 1` of Bertin's Exercise 7.2.

values = simulation.trajectory_['points.x']
naive = [values[0]]
for _ in range(len(values) - 1):
    naive.append(np.mod(2 * naive[-1], 1.0))
fig, (collapse, density) = plt.subplots(1, 2, figsize=(10, 4))
collapse.plot(simulation.trajectory_.times, np.mean(np.array(naive) == 0, axis=1), label='floating-point $2x$ mod 1')
collapse.plot(simulation.trajectory_.times, np.mean(values == 0, axis=1), label='model, exact on $k / q$')
collapse.set_xlabel('step')
collapse.set_ylabel('fraction of points at 0')
collapse.set_title('Floating-point doubling collapses to 0')
collapse.legend()
density.hist(values[20:].ravel(), bins=20, range=(0, 1), density=True, label='model, last 80 steps')
density.axhline(1.0, color='black', label='invariant density $p(x) = 1$')
density.set_ylim(0, 1.5)
density.set_xlabel('x')
density.set_ylabel('density')
density.set_title('The exact orbits stay uniform')
density.legend(loc='lower center')
fig.tight_layout()
plt.show()
