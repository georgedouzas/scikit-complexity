"""
# Discrete Laplacian as a linear automaton

The cells sit on a ring, the cyclic group of order 101, and each becomes `x(g - 1) + x(g + 1) - 2 x(g)` modulo 3, the
element `a^-1 + a - 2` of the group ring over the field of three elements. From a single 1 the values spread at one
cell per step on each side, as the coefficients of `(x + 1/x - 2)^t` reduced modulo 3, a self-similar triangle.
"""

# %%
# The space-time image shows the ring at every step, the oldest at the top, colored by value.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import LinearAutomaton
from skcomplex.core.runtime import Simulation
from skcomplex.core.space import Neighborhood
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'linear_ring_automaton'

system = create(MODEL, size=101, modulus=3)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).run(until=40)

# %%
# Nonzero cells and invertibility
# -------------------------------
# The values are the coefficients of `(x + 1/x - 2)^t` modulo 3, so the number of nonzero cells jumps up and down
# as Lucas' theorem predicts, and it stays below the width `2 t + 1` of the light cone. The automaton is the circulant
# matrix of `a^-1 + a - 2`, whose rank over GF(3) decides whether a configuration can be recovered from its image, and
# on the integers its Laurent polynomial is surjective, not being zero, but not invertible, not being a monomial.

space = system.elements[0].space
values = simulation.trajectory_['cells.value']
steps = np.arange(values.shape[0])
laplacian = LinearAutomaton(space, Neighborhood.radius(), (1, -2, 1), 'GF(3)')
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(steps, (values != 0).sum(axis=1), 'o-', label='nonzero cells')
ax.plot(steps, 2 * steps + 1, '--', label='light cone $2t + 1$')
ax.set_xlabel('step $t$')
ax.set_ylabel('cells')
ax.set_title(
    f'Invertible on the ring: {laplacian.is_invertible()}, surjective on Z: {laplacian.is_surjective(infinite=True)}',
)
ax.legend()
plt.show()
