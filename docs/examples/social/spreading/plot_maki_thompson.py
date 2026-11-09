"""
# Maki-Thompson rumour

One spreader tells the rumour to the people it meets. Spreaders who meet someone who already knows it lose interest
and become stiflers, so the rumour dies out before it reaches everyone: about a fifth of the people never hear it.
"""

# %%
# The system is built from its class with 1000 people, one of them a spreader, and equal rates of spreading
# and stifling, and runs for 20 time units. The default blueprint shows the ignorants, spreaders, and stiflers in one
# time series view.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.spreading import MakiThompsonSystem

SYSTEM = 'maki_thompson'

system = MakiThompsonSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# The fifth who never hear
# ------------------------
# For a large population the fraction `x` of people who never hear the rumour solves `x = exp(-2 (1 - x))`, so
# `x = 0.203`. Runs with different seeds should end close to this fraction, with a spread that shrinks for larger
# populations.

theory = brentq(lambda x: x - np.exp(-2 * (1 - x)), 0.01, 0.9)
fig, ax = plt.subplots(figsize=(6, 4.5))
for n_people in (200, 1000, 5000):
    fractions = []
    for seed in range(20):
        run = simulate(MakiThompsonSystem(n_people=n_people), until=40, dt=1.0, random_state=seed)
        fractions.append(run.get_values('ignorants.n')[-1] / n_people)
    finals = np.array(fractions)
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
