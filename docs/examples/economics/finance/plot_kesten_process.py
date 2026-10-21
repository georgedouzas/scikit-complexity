"""
# Kesten process

Each firm's size is multiplied by a random factor and then receives a small random addition, `x' = lambda x + b`.
The factors shrink the sizes on average, the additions keep them away from zero, and the balance of the two produces
a stationary distribution with a power-law tail, even though no factor is drawn from a heavy-tailed law.
"""

# %%
# Three thousand firms of size 100 draw `lambda` uniformly from `[0.48, 1.48]` and `b` uniformly from `[0, 1]` every
# period, for 250 periods. The mean of `log lambda` is negative, so typical firms shrink towards the stationary mean
# size `<b> / (1 - <lambda>) = 25`. The default blueprint shows the mean and the largest size, which keeps leaping
# with rare runs of large factors.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.analysis import ccdf
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'kesten_process'
N_FIRMS, LOW, HIGH, STEPS = 3000, 0.48, 1.48, 250

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, n_firms=N_FIRMS), observers=[observer], random_state=0).run(until=STEPS)

# %%
# Tail exponent
# -------------
# Kesten's theorem gives the tail `P(X > x) ~ x^-alpha`, where `alpha` solves `<lambda^alpha> = 1`. For uniform
# factors, `<lambda^alpha> = (HIGH^(alpha + 1) - LOW^(alpha + 1)) / ((alpha + 1) (HIGH - LOW))`, which gives `alpha`
# close to 1.4. Pooling the sizes of the last 100 periods, the tail of the complementary cumulative distribution should
# run parallel to `x^-alpha`, and the mean size should relax to 25.


def moment(alpha):
    """Return `<lambda^alpha> - 1` for uniform factors."""
    return (HIGH ** (alpha + 1) - LOW ** (alpha + 1)) / ((alpha + 1) * (HIGH - LOW)) - 1


alpha = brentq(moment, 0.1, 10.0)
trajectory = simulation.trajectory_
sizes = np.array([trajectory.agents('firms', frame).values['x'] for frame in range(STEPS - 100, STEPS + 1)])
values, probabilities = ccdf(sizes.ravel())
anchor = np.searchsorted(values, 100.0)
fig, (tail, mean) = plt.subplots(1, 2, figsize=(10, 4))
tail.loglog(values, probabilities, label='simulation')
tail.loglog(values, probabilities[anchor] * (values / values[anchor]) ** -alpha, 'k--', lw=0.8, label=f'x^-{alpha:.2f}')
tail.set_ylim(probabilities.min(), 1.5)
tail.set_xlabel('size x')
tail.set_ylabel('P(X > x)')
tail.set_title('A power-law tail from bounded factors')
tail.legend()
mean.semilogy(trajectory.times, trajectory['mean_size'], label='mean size')
mean.axhline(0.5 / (1 - (LOW + HIGH) / 2), color='black', linestyle='--', lw=0.8, label='<b> / (1 - <lambda>)')
mean.set_xlabel('period t')
mean.set_ylabel('mean size')
mean.set_title('The mean relaxes to its stationary value')
mean.legend()
fig.tight_layout()
plt.show()
