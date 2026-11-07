"""
# Fair Food Program

The Coalition of Immokalee Workers presses tomato buyers to join the Fair Food Program, whose farms pay pickers more.
Buyers first agree to prefer member farms, then buy only from them once there are enough, and farms join when the
buyers outside the programme no longer need their tomatoes. Farms hold out for years and then join all at once.
"""

# %%
# Thirty buyers and twenty farms of random sizes share 1000 acres, and every buyer outside the programme joins it with
# probability 0.001 a period, for 5000 periods. The default blueprint shows the buyers and farms at every level.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'fair_food_program'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=10)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run(until=5000)

# %%
# The tipping point
# -----------------
# As in Fig. 7.4 of Chesney, the buyers committed to the programme should rise slowly, the member farms should stay
# at zero until few buyers remain outside, and then join in a cascade, after which every buyer moves to level 2.
# Across seeds, the period at which all farms have joined spreads widely.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
tipping = []
for seed in range(10):
    farms = (
        create_simulation(SYSTEM, random_state=seed).prepare().run(until=5000).trajectory_.values('market.member_farms')
    )
    tipping.append(np.argmax(farms == farms.max()))
fig, (levels, spread) = plt.subplots(1, 2, figsize=(10, 4))
levels.plot(times, trajectory.values('market.committed_shops'), label='buyers at level 1')
levels.plot(times, trajectory.values('market.member_shops'), label='buyers at level 2')
levels.plot(times, trajectory.values('market.member_farms'), label='member farms')
levels.set_xlabel('period')
levels.set_ylabel('agents')
levels.set_title('Joining the programme')
levels.legend()
spread.hist(tipping, bins=8)
spread.set_xlabel('period when the last farm joins')
spread.set_ylabel('runs')
spread.set_title('Tipping points over ten seeds')
fig.tight_layout()
plt.show()
