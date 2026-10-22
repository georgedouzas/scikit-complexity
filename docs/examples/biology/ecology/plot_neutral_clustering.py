"""
# Neutral clustering

Organisms that divide, die, and hop on a ring of 128 patches gather in clusters, one column per patch and one row per
frame: from a uniform start, dense patches grow next to empty stretches, although births balance deaths everywhere.
"""

# %%
# The model is created by name from the catalog with 10 organisms per patch, division and death at the rate 1, and
# hops to each neighbour at the rate 0.5. The observer saves every second frame with the default blueprint, whose
# primary view is the space-time image of the numbers of organisms, the oldest frame at the top.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'neutral_clustering'
SITES, N, ALPHA, BETA = 128, 10.0, 1.0, 0.5

system = create(MODEL, shape=SITES, n=N, alpha=ALPHA, beta=BETA)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(
    system,
    dt=0.25,
    backend='numba',
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=60)

# %%
# Growth of the fluctuations
# --------------------------
# The covariances `G_k` of the numbers of patches `k` apart, less their Poisson part, obey the exact equations
# `dG_k/dt = 2 beta (G_(k-1) + G_(k+1) - 2 G_k) + 2 alpha n delta_k0`. Their solution gives the expected variance over
# the patches, which grows like `t^(1/2)` on the ring. The variance of the run should follow it, far below the growth
# `2 alpha n t` of patches without hops, which a run with `beta = 0` should follow.


def expected_variance(times):
    """Return the expected variance over the patches of the ring from the moment equations."""

    def derivative(t, covariance):
        rates = 2 * BETA * (np.roll(covariance, 1) + np.roll(covariance, -1) - 2 * covariance)
        rates[0] += 2 * ALPHA * N
        return rates

    initial = -N * (np.arange(SITES) == 0)
    covariance = solve_ivp(derivative, (0, times[-1]), initial, t_eval=times, rtol=1e-8).y
    covariance += N * (np.arange(SITES) == 0)[:, None]
    return covariance[0] - covariance.mean(axis=0)


times = simulation.trajectory_.times
still = create(MODEL, shape=SITES, n=N, alpha=ALPHA, beta=0.0)
isolated = (
    Simulation(still, dt=0.25, backend='numba', random_state=0, observables=describe(MODEL).observables_of(still))
    .run(until=60)
    .trajectory_['variance']
)

fig, (growth, profile) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(times, simulation.trajectory_['variance'], label='simulated, beta = 0.5')
growth.plot(times, expected_variance(times), color='black', label='moment equations, beta = 0.5')
growth.plot(times, isolated, label='simulated, beta = 0')
growth.plot(times, 2 * ALPHA * N * times, color='black', linestyle='--', label='2 alpha n t')
growth.set_xlabel('time')
growth.set_ylabel('variance over the patches')
growth.set_title('Hops slow the growth of the fluctuations')
growth.legend()
profile.bar(np.arange(SITES), simulation.trajectory_['patches.n'][-1], width=1.0)
profile.axhline(N, color='black', linestyle='--', label='mean')
profile.set_xlabel('patch')
profile.set_ylabel('organisms')
profile.set_title(f'Clusters at t = {times[-1]:g}')
profile.legend()
fig.tight_layout()
plt.show()
