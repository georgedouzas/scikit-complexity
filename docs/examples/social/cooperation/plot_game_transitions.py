"""
# Evolutionary dynamics with game transitions

Su, McAvoy, Wang, and Nowak let the game on every edge of a graph depend on the actions played on it. Mutual
cooperation keeps a valuable donation game, and any defection moves the edge to a less valuable one. The transitions
make defection costly in the future and lower the benefit-to-cost ratio that cooperation needs.
"""

# %%
# A hundred players on a random regular graph of degree four play donation games of benefits `b1 = 3.5` and
# `b2 = 2.6` at cost 1, under death-birth updating with weak selection. The default blueprint shows the strategies.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'game_transitions'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), observers=[observer], random_state=0).prepare().run(until=100)

# %%
# Transitions against a single game
# ---------------------------------
# Under pairwise comparison with strong selection cooperation is favoured when `b1 - b2 > 2 c`. With `b1 = 6` and
# `b2 = 2`, cooperators should take over with transitions and die out with game 1 alone, while the share of edges of
# game 1 follows the mutual cooperation.

settings = {'update': 'pairwise', 'selection': 1.0, 'first_benefit': 6.0, 'second_benefit': 2.0}
fig, (strategies, games) = plt.subplots(1, 2, figsize=(10, 4))
for name, transitions in (('with transitions', True), ('game 1 only', False)):
    system = create_system(SYSTEM, transitions=transitions, **settings)
    trajectory = Simulation(system, random_state=0).prepare().run(until=50).trajectory_
    share = [np.mean(trajectory.agents('players', frame).values['strategy']) for frame in range(51)]
    strategies.plot(trajectory.times, share, label=name)
    games.plot(trajectory.times[1:], trajectory.values('evolution.game_one')[1:], label=name)
strategies.set_xlabel('generations')
strategies.set_ylabel('share of cooperators')
strategies.set_title('Transitions favour cooperation')
strategies.legend()
games.set_xlabel('generations')
games.set_ylabel('share of edges with game 1')
games.set_title('The valuable game follows cooperation')
games.legend()
fig.tight_layout()
plt.show()
