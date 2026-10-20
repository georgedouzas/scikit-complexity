"""
# Ajani's choice and fair food

Consumers on a grid choose between fair-trade and regular food from its price, their budget, their ethical concern,
and what their neighbours bought. Fair food spreads in clusters, and its share falls as its price premium rises,
less so when consumers follow their neighbours.
"""

# %%
# Four hundred consumers of uniform ethical concern and budgets between 1 and 3 fill a 20 by 20 torus. Regular food
# costs 1 and fair food `1 + premium`. Every step, all at once, a consumer who can afford fair food buys it with the
# logit probability of the utility `2 ethics - 5 premium / budget + 2 share`, where `share` is the share of its eight
# neighbours who bought it the step before. The agents view shows the buyers of fair food (yellow) and of regular food
# (purple).

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fair_food'

system = create(MODEL, width=20, height=20, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=30)

# %%
# Price premium and social influence
# ----------------------------------
# Without social influence the consumers choose independently, so the expected share of fair food is the mean over
# consumers of `1(budget >= 1 + premium) / (1 + exp(-(2 ethics - 5 premium / budget)))`, computed from the drawn
# ethics and budgets. Simulations without influence should follow this curve, and with influence the share should be
# higher at every premium, since buyers of fair food make it more attractive to their neighbours.

consumers = simulation.trajectory_.agents('consumers', 0)
ethics, budgets = consumers.values['ethics'], consumers.values['budget']
premiums = np.linspace(0.0, 1.0, 5)
fine = np.linspace(0.0, 1.0, 101)
expected = [np.mean((budgets >= 1 + premium) * expit(2 * ethics - 5 * premium / budgets)) for premium in fine]
shares = {}
for social_weight in (0.0, 2.0):
    shares[social_weight] = [
        Simulation(create(MODEL, premium=premium, social_weight=social_weight, random_state=0), random_state=0)
        .run(until=15)
        .trajectory_['fair_share'][-5:]
        .mean()
        for premium in premiums
    ]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(fine, expected, color='black', linestyle='--', label='independent choice, theory')
ax.plot(premiums, shares[0.0], 'o', label='simulation, no influence')
ax.plot(premiums, shares[2.0], 's', label='simulation, social influence 2')
ax.set_xlabel('price premium of fair food')
ax.set_ylabel('share of fair food')
ax.set_title('Fair food loses buyers as its premium rises')
ax.legend()
fig.tight_layout()
plt.show()
