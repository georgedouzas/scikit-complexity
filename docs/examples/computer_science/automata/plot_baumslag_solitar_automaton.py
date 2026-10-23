"""
# Majority automaton on the Baumslag-Solitar group

The cells sit on a ball of radius 6 of the Baumslag-Solitar group `BS(1, 2) = <a, t | t a t^-1 = a^2>`, built from a
confluent rewriting system that Knuth-Bendix completion in GAP finds. Each cell takes the majority value of itself and
its neighbours along `a`, `a^-1`, `t`, and `t^-1` inside the ball, so the random start freezes into clusters within a
few steps.
"""

# %%
# The ball's elements sit on rings by word length, colored by their state.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'baumslag_solitar_automaton'

system = create(MODEL, radius=6, density=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).run(until=20)

# %%
# Exponential growth and freezing
# -------------------------------
# `BS(1, 2)` is solvable but not virtually nilpotent, so its balls grow exponentially, slower than those of the free
# group of rank 2. The ratio of consecutive ball sizes should approach a constant below 3, and the density of ones
# should settle after a few steps.

space = system.elements[0].space
sizes = np.asarray(space.ball_sizes(), dtype=float)
alive = simulation.trajectory_['cells.alive']
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
left.plot(np.arange(1, sizes.size), sizes[1:] / sizes[:-1], 'o-')
left.set_xlabel('radius $n$')
left.set_ylabel('$|B(n)| / |B(n - 1)|$')
left.set_title('Exponential growth of BS(1, 2)')
right.plot(simulation.trajectory_.times, alive.mean(axis=1), 'o-')
right.set_xlabel('step')
right.set_ylabel('density of ones')
right.set_title('The majority rule settles')
plt.show()
