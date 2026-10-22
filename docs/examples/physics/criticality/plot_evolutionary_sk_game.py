"""
# Evolutionary Sherrington-Kirkpatrick game

256 agents learn to play ±1 on a Sherrington-Kirkpatrick landscape, each with a softmax policy at its own temperature,
and the least fit agent is replaced by a newcomer with a temperature drawn uniformly on [0, 2]. The temperatures start
uniform and evolve, by selection alone, to a narrow band of low temperatures at the edge between ergodic and frozen
learning. The view shows the histogram of the temperatures of the agents, which piles up at low temperatures.
"""

# %%
# The model is created by name from the catalog with memory 0.75, the value of Lesmana et al. Every third step
# replaces the least fit agent, and two steps of learning alone follow. The observer saves every 50th step with the
# default blueprint, whose primary view is the histogram of the temperatures. Every third frame of the
# trajectory holds the fitnesses that decide a replacement.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import avalanches, ergodicity_distance, fit_power_law, histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'evolutionary_sk_game'
STEPS = 14999
TRANSIENT = 1500

system = create(MODEL, memory=0.75, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=50)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=STEPS)
trajectory = simulation.trajectory_
temperatures = np.asarray(trajectory['agents.temperature'])
fitness = np.asarray(trajectory['agents.fitness'])[::3][TRANSIENT // 3 :]
rewards = np.asarray(trajectory['agents.reward']).mean(axis=1)

# %%
# Selection, cascades, and the edge of ergodicity breaking
# --------------------------------------------------------
# The temperatures should leave their uniform start and gather at low values, as in Fig. 2 of Lesmana et al.
# A replacement starts a cascade of agents whose fitness falls below a threshold `f_0`, the avalanche of the
# Bak-Sneppen model, here with `f_0` at the 97th percentile of the least fitness. Its sizes should follow
# `P(S) ~ S^-3/2`, the exponent of a critical branching process, which Lesmana et al. measure as 1.58. The ergodicity
# coefficient is the median total-variation distance between the reward distribution of one of four independent runs
# of the game without replacement and that of all four, at the evolved temperatures rescaled by `2^k`. It should be
# high for frozen runs at low `k`, near 0 for ergodic runs at high `k`, and in between at the evolved temperatures.

threshold = float(np.quantile(fitness.min(axis=1), 0.97))
sizes = avalanches((fitness < threshold).sum(axis=1)).sizes
fit = fit_power_law(sizes, discrete=True, xmin=5)
edges, density = histogram(sizes, 15, log=True)
centers, shown = np.sqrt(edges[1:] * edges[:-1]), density > 0
tail = shown & (centers >= 5)
scale = np.exp(np.mean(np.log(density[tail]) + 1.5 * np.log(centers[tail])))

evolved = temperatures[-1]
scales = np.arange(-2, 3)
coefficients = []
for scale in scales:
    runs = [
        Simulation(create(MODEL, temperature=evolved * 2.0**scale, random_state=0), dt=1, random_state=run)
        .run(until=300)
        .trajectory_['agents.reward']
        .mean(axis=1)[150:]
        for run in range(4)
    ]
    coefficients.append(ergodicity_distance(np.asarray(runs)))

fig, (spread, cascades, edge) = plt.subplots(1, 3, figsize=(14, 4))
spread.hist(temperatures[0], bins=20, range=(0, 2), density=True, alpha=0.5, label='initial')
spread.hist(evolved, bins=20, range=(0, 2), density=True, alpha=0.7, label='evolved')
spread.set_xlabel('temperature T')
spread.set_ylabel('density')
spread.set_title('Selection lowers the temperatures')
spread.legend()
cascades.loglog(centers[shown], density[shown], 'o', label=f'{sizes.size} cascades, fitted exponent {fit.alpha:.2f}')
guide = np.logspace(0, np.log10(centers[shown][-1]), 50)
cascades.loglog(guide, scale * guide**-1.5, color='black', label='S^-3/2')
cascades.set_xlabel('cascade size S')
cascades.set_ylabel('probability density')
cascades.set_title('Replacement cascades of all sizes')
cascades.legend()
edge.plot(scales, coefficients, 'o-')
edge.axvline(0, color='gray', linestyle='--', label='evolved temperatures')
edge.set_xlabel('rescaling k of the temperatures, T 2^k')
edge.set_ylabel('ergodicity coefficient')
edge.set_title(f'At the edge of ergodicity breaking, mean reward {rewards[TRANSIENT:].mean():.3f}')
edge.legend()
fig.tight_layout()
plt.show()
