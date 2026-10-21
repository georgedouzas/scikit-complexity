"""
# Replicator dynamics

In the hawk-dove game a hawk does well among doves and badly among hawks, so neither strategy can take over: whatever
the start, the share of hawks settles at `V / C`, the mixed equilibrium of the game.
"""

# %%
# The model is created by name from the catalog with the value `V = 2` and the cost `C = 3` of a fight, and started
# with 5% hawks. The observer saves every second frame with the default blueprint, whose primary view is the time
# series of the shares of hawks and doves.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'replicator_dynamics'

system = create(MODEL, x=[0.05, 0.95], V=2.0, C=3.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.1, observers=[observer]).run(until=30)

# %%
# The mixed equilibrium
# ---------------------
# The hawks grow when their payoff `(V - C) / 2 x + V (1 - x)` exceeds that of the doves `V / 2 (1 - x)`, that is
# while their share `x` is below `V / C`. Started from few or many hawks, the share should converge to `2 / 3`.

fig, ax = plt.subplots(figsize=(7, 4))
for start in [0.05, 0.3, 0.6, 0.95]:
    trajectory = Simulation(create(MODEL, x=[start, 1 - start]), dt=0.1).run(until=30).trajectory_
    ax.plot(trajectory.times, trajectory['population.x'][:, 0], label=f'{start:.0%} hawks at the start')
ax.axhline(2 / 3, color='black', linestyle='--', label='V / C')
ax.set_xlabel('time')
ax.set_ylabel('share of hawks')
ax.set_title('Every start converges to V / C hawks')
ax.legend()
plt.show()
