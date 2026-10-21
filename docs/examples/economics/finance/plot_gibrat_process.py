"""
# Gibrat process

Gibrat's law of proportionate effect: every firm grows each period by a random factor independent of its size. The
logarithm of a size is then a sum of independent increments, so the sizes spread into a lognormal distribution whose
variance grows linearly in time, and the largest firm pulls ever further ahead of the mean.
"""

# %%
# A thousand firms of size 100 grow by normal factors of mean 1.02 and standard deviation 0.05 for 150 periods. The
# default blueprint shows the mean and the largest size.

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'gibrat_process'
N_FIRMS, SIZE, MU, SIGMA, STEPS = 1000, 100.0, 1.02, 0.05, 150

system = create(MODEL, n_firms=N_FIRMS, x=SIZE, mu=MU, sigma=SIGMA)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=STEPS)

# %%
# Lognormal sizes
# ---------------
# With `m` and `v` the mean and the variance of the logarithm of one growth factor, computed here by Gauss-Hermite
# quadrature, `log x_t` is normal with mean `log x_0 + m t` and variance `v t`. The histogram of the final log sizes
# should follow that normal density, and the variance of the log sizes should grow along the line `v t`.

nodes, weights = np.polynomial.hermite_e.hermegauss(40)
logs = np.log(MU + SIGMA * nodes)
m = weights @ logs / weights.sum()
v = weights @ (logs - m) ** 2 / weights.sum()
trajectory = simulation.trajectory_
sizes = np.array([trajectory.agents('firms', frame).values['x'] for frame in range(STEPS + 1)])
grid = np.linspace(np.log(sizes[-1]).min(), np.log(sizes[-1]).max(), 200)
fig, (density, spread) = plt.subplots(1, 2, figsize=(10, 4))
density.hist(np.log(sizes[-1]), bins=40, density=True, alpha=0.5, label='simulation')
density.plot(grid, stats.norm.pdf(grid, np.log(SIZE) + m * STEPS, np.sqrt(v * STEPS)), 'k', label='theory')
density.set_xlabel('log size')
density.set_ylabel('density')
density.set_title(f'Log sizes after {STEPS} periods are normal')
density.legend()
spread.plot(trajectory.times, np.log(sizes).var(axis=1), label='simulation')
spread.plot(trajectory.times, v * trajectory.times, 'k--', lw=0.8, label='v t')
spread.set_xlabel('period t')
spread.set_ylabel('variance of log size')
spread.set_title('The spread grows linearly')
spread.legend()
fig.tight_layout()
plt.show()
