"""
# Levy-Solomon multiplicative process

Firms grow by random proportional factors, as in Gibrat's law, but no firm may shrink below a fixed fraction of the
mean size. The reflecting floor turns the ever-widening lognormal into a stationary distribution of relative sizes
with a power-law, Pareto tail, whose exponent is set by the floor alone.
"""

# %%
# Two thousand firms grow by normal factors of mean 1 and standard deviation 0.2, with a floor at 30 percent of the
# mean size of the previous period, for 250 periods. The default blueprint shows the mean and the largest size: the
# mean drifts up as the floor pushes the small firms up, and the largest firm leaps when a lucky run of growth carries
# it into the tail.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import ccdf
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'multiplicative_process'
N_FIRMS, MU, SIGMA, FLOOR, STEPS = 2000, 1.0, 0.2, 0.3, 250

system = create(MODEL, n_firms=N_FIRMS, mu=MU, sigma=SIGMA, c=FLOOR)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=STEPS)

# %%
# Pareto tail of the relative sizes
# ---------------------------------
# Levy and Solomon showed that the relative sizes `w = x / <x>` settle into a distribution whose complementary
# cumulative distribution falls as `w^-alpha` with `alpha = 1 / (1 - c)`, about 1.43 here. Pooling the last 100
# periods, the tail should run parallel to that line on log-log axes, while the log sizes spread no further.

trajectory = simulation.trajectory_
sizes = np.array([trajectory.agents('firms', frame).values['x'] for frame in range(STEPS + 1)])
relative = sizes / sizes.mean(axis=1, keepdims=True)
values, probabilities = ccdf(relative[-100:].ravel())
alpha = 1 / (1 - FLOOR)
fig, (tail, spread) = plt.subplots(1, 2, figsize=(10, 4))
tail.loglog(values, probabilities, label='simulation')
tail.loglog(values, (values / FLOOR) ** -alpha, 'k--', lw=0.8, label=f'w^-{alpha:.2f}')
tail.set_ylim(probabilities.min(), 1.5)
tail.set_xlabel('relative size w = x / <x>')
tail.set_ylabel('P(W > w)')
tail.set_title('A power-law tail of relative sizes')
tail.legend()
spread.plot(trajectory.times, np.log(relative).std(axis=1))
spread.set_xlabel('period t')
spread.set_ylabel('standard deviation of log w')
spread.set_title('The distribution becomes stationary')
fig.tight_layout()
plt.show()
