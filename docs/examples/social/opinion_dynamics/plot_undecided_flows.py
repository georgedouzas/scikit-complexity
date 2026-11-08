"""
# Mean-field flows of the undecided

The numbers of agents against, undecided, and for, with undecided agents joining either side and decided agents
becoming undecided at rates that grow with the number of undecided. Below a critical population every agent ends on
one side, and above it the surplus never makes up its mind.
"""

# %%
# A hundred agents start undecided, with `omega = 1`, `gamma = 0.05`, and `eta = 0.02`, so the critical size is 70.
# The default blueprint shows the three numbers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.opinion_dynamics import UndecidedFlowsSystem

SYSTEM = 'undecided_flows'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(UndecidedFlowsSystem(), observers=[observer])

# %%
# The critical population
# -----------------------
# The final number of undecided agents should be zero below `N_c = omega (1 / gamma + 1 / eta) = 70` and grow as
# `N - N_c` above it, while the sides hold `omega / gamma = 20` and `omega / eta = 50` agents.

sizes = np.linspace(20, 150, 14)
finals = []
for size in sizes:
    run = simulate(UndecidedFlowsSystem(n_agents=float(size)), until=200, dt=1.0)
    finals.append(run.values('population.undecided')[-1])
fig, (flows, critical) = plt.subplots(1, 2, figsize=(10, 4))
for name, label in (('against', 'against A'), ('support', 'for B'), ('undecided', 'undecided C')):
    flows.plot(trajectory.times, trajectory.values(f'population.{name}'), label=label)
flows.set_xlabel('time')
flows.set_ylabel('agents')
flows.set_title('One hundred agents')
flows.legend()
critical.plot(sizes, finals, 'o', label='final undecided')
critical.plot(sizes, np.maximum(sizes - 70, 0), color='black', linestyle=':', label='max(N - 70, 0)')
critical.set_xlabel('population N')
critical.set_ylabel('undecided agents')
critical.set_title('A transition at N_c = 70')
critical.legend()
fig.tight_layout()
plt.show()
