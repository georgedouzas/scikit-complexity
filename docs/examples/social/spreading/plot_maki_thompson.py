"""
# Maki-Thompson rumour

One spreader tells the rumour to the people it meets. Spreaders who meet someone who already knows it lose interest
and become stiflers, so the rumour dies out before it reaches everyone: about a fifth of the people never hear it.
"""

# %%
# The model is created by name from the catalog with 1000 people, one of them a spreader, and equal rates of spreading
# and stifling, and runs for 20 time units. The default blueprint shows the ignorants, spreaders, and stiflers in one
# time series view.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'maki_thompson'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=0.5, observers=[observer], random_state=0).run(until=20)

# %%
# The fifth who never hear
# ------------------------
# For a large population the fraction `x` of people who never hear the rumour solves `x = exp(-2 (1 - x))`, so
# `x = 0.203`. Runs with different seeds should end close to this fraction, with a spread that shrinks for larger
# populations.

theory = brentq(lambda x: x - np.exp(-2 * (1 - x)), 0.01, 0.9)
fig, ax = plt.subplots(figsize=(6, 4.5))
for n_people in (200, 1000, 5000):
    finals = []
    for seed in range(20):
        run = Simulation(create(MODEL, n_people=n_people), dt=1.0, random_state=seed).run(until=40).trajectory_
        finals.append(run['ignorants.n'][-1] / n_people)
    finals = np.array(finals)
    spread_out = finals[finals < 0.9]
    ax.plot(np.full(len(spread_out), n_people), spread_out, 'o', alpha=0.5, label=f'{n_people} people')
ax.axhline(theory, color='black', linestyle='--', label=f'theory x = {theory:.3f}')
ax.set_xscale('log')
ax.set_xticks([200, 1000, 5000], labels=['200', '1000', '5000'])
ax.minorticks_off()
ax.set_xlabel('number of people')
ax.set_ylabel('fraction who never hear the rumour')
ax.set_title('Runs where the rumour took off, 20 seeds each')
ax.legend()
plt.show()
