"""
# Shaken spring-block chain

The space-time image of the block positions, one row per frame with the oldest at the top, shows the cycles of the
protocol: in every shaking phase the positions change quickly, and in every relaxation phase the blocks slow down and
stick one after the other, until the whole chain is frozen in a blocked configuration that lasts until the next shake.
"""

# %%
# The model is created by name from the catalog with 60 blocks of unit weight joined by unit springs, with static and
# dynamic friction coefficients 1 and 0.8, shaken for 2 time units with a random force of amplitude 2 and relaxed for
# 20 time units in every cycle, and simulated for 13 cycles. The observer saves every frame with the default
# blueprint, whose primary view is the space-time image of the positions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'shaken_spring_block'

system = create(MODEL, n_blocks=60, amplitude=2.0, shake_time=2.0, relax_time=20.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, solver=EulerMaruyama(dt=0.01), observers=[observer], random_state=0)
simulation.run(until=286)

# %%
# Blocked configurations and their correlations
# ---------------------------------------------
# At the end of every relaxation the chain is blocked, so every block is at rest and the spring force on it,
# `xi_{i+1} - xi_i` with the elongations `xi_i = x_{i+1} - x_i`, is below the static friction 1. Stronger shaking
# leaves blocked configurations of higher energy per spring `<xi^2> / 2`, and the elongations are correlated over a
# length that grows in proportion to that energy (Bertin 2021, section 3.4.3). The fraction of moving blocks of the run
# above should fall to zero before the end of every cycle, and the correlation lengths of chains of 100 blocks shaken
# with amplitudes `A` from 0.75 to 3 should lie close to a line through the origin against their energies.

moving = np.mean(np.abs(simulation.trajectory_['chain.v']) > 0.02, axis=1)


def blocked_statistics(amplitude, relax_time):
    period = 2.0 + relax_time
    run = Simulation(
        create(MODEL, n_blocks=100, amplitude=amplitude, relax_time=relax_time),
        dt=period,
        solver=EulerMaruyama(dt=0.01),
        random_state=1,
    )
    elongations = np.diff(run.run(until=12 * period).trajectory_['chain.x'][1:], axis=1)
    correlations = np.array([np.mean(elongations[:, : 99 - lag] * elongations[:, lag:]) for lag in range(40)])
    correlations /= correlations[0]
    positive = correlations[: np.argmax(correlations < 0)] if np.any(correlations < 0) else correlations
    return np.mean(elongations**2 / 2), positive.sum()


amplitudes = [(0.75, 8.0), (1.0, 10.0), (1.5, 15.0), (2.0, 20.0), (3.0, 40.0)]
statistics = np.array([blocked_statistics(amplitude, relax_time) for amplitude, relax_time in amplitudes])

fig, (relaxation, correlation) = plt.subplots(1, 2, figsize=(10, 4))
relaxation.plot(simulation.trajectory_.times, moving)
relaxation.set_xlabel('time')
relaxation.set_ylabel('fraction of moving blocks')
relaxation.set_title('Every cycle ends in a blocked state')
slope = np.sum(statistics[:, 0] * statistics[:, 1]) / np.sum(statistics[:, 0] ** 2)
energies = np.linspace(0.0, statistics[:, 0].max() * 1.1, 2)
correlation.plot(energies, slope * energies, color='black', linestyle='--', label='proportional fit')
correlation.plot(statistics[:, 0], statistics[:, 1], 'o', label='simulated')
correlation.legend()
for (amplitude, _), (energy, length) in zip(amplitudes, statistics, strict=True):
    correlation.annotate(f'A = {amplitude}', (energy, length), textcoords='offset points', xytext=(5, -12))
correlation.set_xlabel('energy per spring')
correlation.set_ylabel('correlation length of the elongations')
correlation.set_title('Correlation length grows with the energy')
fig.tight_layout()
plt.show()
