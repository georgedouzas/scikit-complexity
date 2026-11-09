"""
# Muller's automaton on the free group

The cells sit on a ball of radius 5 of the free group F2 and hold pairs `(h, k)` of integers modulo 2. Every cell
becomes `p1(x(ga)) + p2(x(gb)) + p1(x(ga^-1)) + p2(x(gb^-1))`, with `p1(h, k) = (h, 0)` and `p2(h, k) = (k, 0)`. The
image lies in `H x {0}`, so the automaton is not surjective, yet it is pre-injective, which the Garden of Eden theorem
forbids on amenable groups such as the integers.
"""

# %%
# The ball holds 485 elements on rings by word length, and the run logs both coordinates of the cells.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import MullerAutomatonSystem

SYSTEM = 'muller_automaton'

system = MullerAutomatonSystem(radius=5, modulus=2, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer])

# %%
# Image and pre-injectivity
# -------------------------
# After one step the second coordinate should be 0 everywhere, so no configuration with a nonzero second coordinate is
# an image. A nonzero configuration supported inside the ball of radius 3 should never map to 0: its element of
# largest length leaves a nonzero value one step further out, which the proof of the theorem reads.

second = trajectory.get_values('cells.second')
space = system.prepare().system_.elements[0].space.prepare()
interior = np.flatnonzero(space.lengths_ <= 3)
rng = np.random.default_rng(1)
supports, images = [], []
for _ in range(40):
    pairs = np.zeros((space.size_, 2))
    chosen = rng.choice(interior, size=rng.integers(1, 6), replace=False)
    pairs[chosen] = rng.integers(0, 2, (chosen.size, 2))
    pairs[chosen[0], 0] = 1
    image = simulate(MullerAutomatonSystem(radius=5, initial=pairs), until=1)
    supports.append(int(pairs.any(axis=1).sum()))
    images.append(int(image.get_values('cells.first')[-1].astype(bool).sum()))
fig, (coordinates, check) = plt.subplots(1, 2, figsize=(10, 4))
coordinates.plot(trajectory.times, (second != 0).mean(axis=1), 'o-', label='second coordinate')
coordinates.plot(
    trajectory.times,
    (trajectory.get_values('cells.first') != 0).mean(axis=1),
    's-',
    label='first coordinate',
)
coordinates.set_xlabel('step')
coordinates.set_ylabel('fraction of nonzero cells')
coordinates.set_title('The image lies in H x {0}')
coordinates.legend()
check.plot(supports, images, 'o')
check.set_xlabel('cells of a finite nonzero configuration')
check.set_ylabel('nonzero cells of its image')
check.set_ylim(bottom=0)
check.set_title('Finite nonzero configurations never vanish')
fig.tight_layout()
plt.show()
