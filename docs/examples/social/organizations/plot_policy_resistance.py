"""
# Policy resistance

One stock, the drug supply on the streets of Meadows' example, is pulled by three actors toward inconsistent goals:
addicts want it high, enforcement wants it low, and pushers want it in the middle. Every actor pulls in proportion to
its discrepancy and intensifies its effort while the discrepancy persists. The stock is stuck where no one wants it,
while the efforts escalate.
"""

# %%
# The goals are 80, 20, and 50, all actors pull with the same strength, and the run lasts 100 years. The default
# blueprint shows the stock and the efforts.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'policy_resistance'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), dt=0.5, observers=[observer]).prepare().run(100)

# %%
# A standoff, and what happens when one actor lets up
# ---------------------------------------------------
# The stock should settle at 50, the mean of the goals weighted by the strengths, while the efforts of the first two
# actors grow without bound in opposite directions. When enforcement lets up, its strength set to zero, the others
# should drag the stock to 65, the mean of their goals.

trajectory = simulation.trajectory_
let_up = Simulation(create_system(SYSTEM, strengths=(0.1, 0.0, 0.1)), dt=0.5).prepare().run(100).trajectory_
fig, (stock, efforts) = plt.subplots(1, 2, figsize=(10, 4))
stock.plot(trajectory.times, trajectory.values('standoff.stock'), label='all actors pull')
stock.plot(let_up.times, let_up.values('standoff.stock'), label='enforcement lets up')
stock.set_xlabel('years')
stock.set_ylabel('stock')
stock.set_title('The stock is stuck at the compromise')
stock.legend()
for actor, name in (('a', 'addicts'), ('b', 'enforcement'), ('c', 'pushers')):
    efforts.plot(trajectory.times, trajectory.values(f'standoff.effort_{actor}'), label=name)
efforts.set_xlabel('years')
efforts.set_ylabel('effort')
efforts.set_title('The efforts escalate')
efforts.legend()
fig.tight_layout()
plt.show()
