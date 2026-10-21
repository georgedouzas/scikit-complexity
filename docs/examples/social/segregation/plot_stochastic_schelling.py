"""
# Stochastic Schelling

Grauwin, Bertin, Lemoy, and Jensen's stochastic Schelling model: agents move between the blocks of a city, and a move
is accepted with a logit probability of the change of utility, which mixes the mover's own utility of the density of
its block with the change of the collective utility. When the moves are noisy the agents spread evenly over the
blocks, and above the inverse temperature `beta = 2` they pile into a few full blocks and leave the others empty, a
segregation that nobody's utility asks for.
"""

# %%
# Two hundred agents live on a 20 by 20 grid of sixteen 5 by 5 blocks, half of the cells occupied, with the decreasing
# utility `u(rho) = 1 - 4 rho / 3`, the altruism 0.5, and `beta = 4`. The blueprint shows the agents after 150 steps,
# crowded into full blocks.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stochastic_schelling'
BLOCKS, CELLS, STEPS = 16, 25, 150

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, beta=4.0, random_state=0), observers=[observer], random_state=0).run(until=STEPS)

# %%
# A segregation transition at beta = 2
# ------------------------------------
# The spread of the block densities measures segregation: random placement gives the binomial standard deviation
# `sqrt(rho (1 - rho) / 25) = 0.1`, and full and empty blocks give 0.5. The spread should stay near the random value for
# small `beta`, rise steeply around `beta = 2`, and saturate near 0.5, and the histogram of block densities should
# turn from one bump around 0.5 into two peaks at 0 and 1.


def densities(trajectory, frame):
    """Return the density of every block at a frame."""
    blocks = trajectory.agents('dwellers', frame).values['block'].astype(int)
    return np.bincount(blocks, minlength=BLOCKS) / CELLS


def final_densities(beta, seed):
    """Return the block densities after a run."""
    trajectory = Simulation(create(MODEL, beta=beta, random_state=seed), random_state=seed).run(until=STEPS).trajectory_
    return densities(trajectory, STEPS)


betas = np.linspace(0.5, 6.0, 12)
spreads = [np.mean([final_densities(beta, seed).std() for seed in range(2)]) for beta in betas]
fig, (sweep, histogram) = plt.subplots(1, 2, figsize=(10, 4))
sweep.plot(betas, spreads, 'o-')
sweep.axhline(0.1, color='grey', linestyle=':', label='random placement')
sweep.axhline(0.5, color='grey', linestyle='--', label='full and empty blocks')
sweep.axvline(2.0, color='C3', linestyle='--', label='beta = 2')
sweep.set_xlabel('inverse temperature beta')
sweep.set_ylabel('spread of block densities')
sweep.set_title('Segregation sets in above beta = 2')
sweep.legend()
bins = np.linspace(0, 1, 11)
for beta, color in ((1.0, 'C0'), (4.0, 'C1')):
    pooled = np.concatenate([final_densities(beta, seed) for seed in range(4)])
    histogram.hist(pooled, bins=bins, alpha=0.5, color=color, label=f'beta = {beta}')
histogram.set_xlabel('block density')
histogram.set_ylabel('blocks in four runs')
histogram.set_title('Even spread or full and empty blocks')
histogram.legend()
fig.tight_layout()
plt.show()
