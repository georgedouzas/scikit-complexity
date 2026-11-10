"""
# Replicator dynamics

In the hawk-dove game a hawk does well among doves and badly among hawks, so neither strategy can take over: whatever
the start, the share of hawks settles at `V / C`, the mixed equilibrium of the game.
"""

# %%
# The example builds `ReplicatorDynamicsSystem` with the value `V = 2` and the cost `C = 3` of a fight, and starts it
# with 5% hawks. The observer saves every second frame with the default blueprint, whose primary view is the time series
# of the shares of hawks and doves.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import ReplicatorDynamicsSystem

SYSTEM = 'replicator_dynamics'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(ReplicatorDynamicsSystem(x=[0.05, 0.95], V=2.0, C=3.0), observers=[observer])

# %%
# The mixed equilibrium
# ---------------------
# The hawks grow when their payoff `(V - C) / 2 x + V (1 - x)` exceeds that of the doves `V / 2 (1 - x)`, that is
# while their share `x` is below `V / C`. Started from few or many hawks, the share should converge to `2 / 3`.

fig, ax = plt.subplots(figsize=(7, 4))
for start in [0.05, 0.3, 0.6, 0.95]:
    trajectory = simulate(ReplicatorDynamicsSystem(x=[start, 1 - start]))
    ax.plot(trajectory.times, trajectory.get_values('population.x')[:, 0], label=f'{start:.0%} hawks at the start')
ax.axhline(2 / 3, color='black', linestyle='--', label='V / C')
ax.set_xlabel('time')
ax.set_ylabel('share of hawks')
ax.set_title('Every start converges to V / C hawks')
ax.legend()
plt.show()
