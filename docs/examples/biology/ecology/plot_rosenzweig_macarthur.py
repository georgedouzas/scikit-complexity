"""
# Rosenzweig-MacArthur with a shrinking capacity

Fast prey and slow predators sit at their coexistence point while the carrying capacity of the prey shrinks. When it
shrinks slowly the populations track the moving point, but when it shrinks fast the prey cannot keep up: they
collapse by thirty orders of magnitude before they return, a rate-induced tipping.
"""

# %%
# The system is created by name from the catalog with the crowding rising at the rate 0.002, above the critical rate of
# about 0.0004, and integrated with its registered solver, which controls only the relative error so that the
# collapsed prey keep their sign. The observer saves every frame with the default blueprint, whose primary view is the
# time series of both populations.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system, describe

SYSTEM = 'rosenzweig_macarthur'

solver = describe(SYSTEM).defaults['solver']
system = create_system(SYSTEM, rate=0.002)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, solver=solver, observers=[observer]).prepare().run(until=149)

# %%
# Slow and fast change
# --------------------
# A slow rise of the crowding, 0.0002, should let the prey decline smoothly along the coexistence point, while the
# fast rise should send them through a collapse of many orders of magnitude before they recover.

fast = simulation.trajectory_
slow = Simulation(create_system(SYSTEM, rate=0.0002), dt=0.5, solver=solver).prepare().run(until=149).trajectory_
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(slow.times, slow.values('prey.n'), label='rate 0.0002: tracks the point')
ax.semilogy(fast.times, fast.values('prey.n'), label='rate 0.002: tips')
ax.set_xlim(0, 40)
ax.set_xlabel('time')
ax.set_ylabel('prey (log scale)')
ax.set_title(f'Lowest prey: {slow.values("prey.n").min():.2g} when slow, {fast.values("prey.n").min():.2g} when fast')
ax.legend()
plt.show()
