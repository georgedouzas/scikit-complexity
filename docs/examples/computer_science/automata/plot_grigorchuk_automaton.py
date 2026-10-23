"""
# Majority automaton on the Grigorchuk group

The cells sit on a ball of radius 8 of the Cayley graph of the Grigorchuk group, on rings by word length. Each cell
takes the majority value of itself and its neighbours along the generators `a`, `b`, `c`, and `d`, counting only the
neighbours inside the ball, so the random start freezes into clusters of agreeing cells.
"""

# %%
# The ball holds 271 elements of the group, each a node colored by its state, and the edges join every element `g` to
# `g s` for the four generators `s`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.space import GroupSpace
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create
from skcomplex.models.computer_science.automata import grigorchuk_group

MODEL = 'grigorchuk_automaton'

system = create(MODEL, radius=8, density=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).run(until=20)

# %%
# Growth of the balls
# -------------------
# The Grigorchuk group grows faster than any polynomial and slower than any exponential. On logarithmic axes the
# local log-log slope of the ball sizes rises with the radius, which no polynomial does, while `log |B(n)| / n` falls,
# which no exponential does. The ball sizes are 1, 5, 11, 23, 40, 68, 108, 176, 271, 427, and 643 for radii 0 to 10,
# far below the free group of rank 2 and the exponential `5^n`.

sizes = np.asarray(GroupSpace(grigorchuk_group(), ball=10).ball_sizes(), dtype=float)
radii = np.arange(sizes.size)
slopes = np.diff(np.log(sizes[1:])) / np.diff(np.log(radii[1:]))
rates = np.log(sizes[1:]) / radii[1:]
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.loglog(radii[1:], sizes[1:], 'o-', label='Grigorchuk group')
left.loglog(radii[1:], 1 + 2 * (3.0 ** radii[1:] - 1), '--', label='free group of rank 2')
left.loglog(radii[1:], 5.0 ** radii[1:], ':', label='$5^n$')
left.set_xlabel('radius $n$')
left.set_ylabel('ball size $|B(n)|$')
left.set_title('Ball sizes of the Grigorchuk group')
left.legend()
right.plot(radii[2:], slopes, 'o-', label='local log-log slope')
right.plot(radii[1:], rates, 's-', label='$\\log |B(n)| / n$')
right.set_xlabel('radius $n$')
right.set_title('Faster than polynomial, slower than exponential')
right.legend()
plt.show()
