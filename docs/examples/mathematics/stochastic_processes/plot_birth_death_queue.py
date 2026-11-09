"""
# Birth-death process and M/M/1 queue

The lengths of three independent queues at half load jump up by one at every arrival and down by one at every
departure, and spend about half of the time empty and rarely grow beyond a few customers.
"""

# %%
# The system is built from its class with three queues of arrival rate 0.5 and service rate 1, so of load
# `rho = 0.5`, and simulated for 60 time units with the Gillespie algorithm. The observer saves every frame with the
# default blueprint, whose primary view holds the three queue lengths.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import BirthDeathQueueSystem

SYSTEM = 'birth_death_queue'

system = BirthDeathQueueSystem(n_queues=3, arrival_rate=0.5, service_rate=1.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, until=59.8, dt=0.2, observers=[observer], random_state=0)

# %%
# Stationary distribution
# -----------------------
# Detailed balance between `n` and `n + 1` customers, `lambda p_n = mu p_{n + 1}`, gives the stationary geometric
# distribution `p_n = (1 - rho) rho^n` for the load `rho = lambda / mu < 1`, with the mean length `rho / (1 - rho)`.
# Frames at regular times sample the time average, so the share of frames with `n` customers in 200 queues over 2000
# time units, after a transient, should match `p_n` for every load, where at least 100 frames are counted, and the mean
# length should grow without bound as the load approaches 1.

loads = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
occupations, means = {}, []
for rho in loads:
    run = simulate(
        BirthDeathQueueSystem(n_queues=200, arrival_rate=rho),
        until=2000,
        dt=0.5,
        random_state=1,
        backend=NumbaBackend(),
    )
    lengths = run.get_values('queues.n')[200:]
    counts = np.bincount(lengths.astype(int).ravel(), minlength=12)[:12]
    occupations[rho] = np.where(counts >= 100, counts / lengths.size, np.nan)
    means.append(lengths.mean())

fig, (distribution, mean) = plt.subplots(1, 2, figsize=(10, 4))
customers = np.arange(12)
for rho in (0.2, 0.5, 0.8):
    (points,) = distribution.semilogy(customers, occupations[rho], 'o', label=f'rho = {rho}')
    distribution.semilogy(customers, (1 - rho) * rho**customers, color=points.get_color(), linestyle='--')
distribution.set_xlabel('number of customers n')
distribution.set_ylabel('share of time')
distribution.set_title('Geometric occupation (dashed: (1 - rho) rho^n)')
distribution.legend()
grid = np.linspace(0.05, 0.85, 100)
mean.plot(grid, grid / (1 - grid), color='black', label='rho / (1 - rho)')
mean.plot(loads, means, 'o', label='simulated')
mean.set_xlabel('load rho')
mean.set_ylabel('mean queue length')
mean.set_title('The queue grows as the load approaches 1')
mean.legend()
fig.tight_layout()
plt.show()
