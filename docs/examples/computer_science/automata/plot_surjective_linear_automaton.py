"""
# Surjective linear automaton on the free group

The cells sit on a ball of radius 5 of the free group F2 and hold vectors of `K^2` over the field of two elements.
Every cell becomes `q1(x(ga)) + q1(x(ga^-1)) + q2(x(gb)) + q2(x(gb^-1))`, with `q1(h, k) = (h, 0)` and
`q2(h, k) = (0, h)`. The automaton is surjective but not pre-injective, the other half of the failure of the Garden of
Eden theorem on the free group.
"""

# %%
# The run starts from random vectors and logs both coordinates of the cells.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'surjective_linear_automaton'

system = create_system(SYSTEM, radius=5, modulus=2, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer]).prepare().run(until=10)

# %%
# A vanishing vector and a preimage
# ---------------------------------
# The vector `(0, 1)` at the identity and 0 elsewhere should map to 0, so two configurations that differ at one cell
# share their image. A preimage of a random target on the ball of radius 3 is built cell by cell from the identity
# outwards, as in the proof, and its image should equal the target there.

space = system.elements[0].space.prepare()
table = space.generator_table_
kernel = np.zeros((space.size_, 2))
kernel[0, 1] = 1
vanishing = Simulation(create_system(SYSTEM, radius=5, initial=kernel)).prepare().run(until=1).trajectory_
rng = np.random.default_rng(1)
target = rng.integers(0, 2, (space.size_, 2))
preimage = np.zeros((space.size_, 2))
assigned = np.zeros(space.size_, dtype=bool)
assigned[0] = True
for site in np.argsort(space.lengths_, kind='stable'):
    if space.lengths_[site] > 3:
        break
    for axis in (0, 1):
        neighbours = table[site, 2 * axis : 2 * axis + 2]
        fresh = neighbours[~assigned[neighbours]]
        known = preimage[neighbours[assigned[neighbours]], 0].sum()
        if fresh.size:
            preimage[fresh[0], 0] = (target[site, axis] - known) % 2
            assigned[fresh] = True
image = Simulation(create_system(SYSTEM, radius=5, initial=preimage)).prepare().run(until=1).trajectory_
inner = space.lengths_ <= 3
hits = (image.values('cells.first')[-1][inner] == target[inner, 0]) & (
    image.values('cells.second')[-1][inner] == target[inner, 1]
)
fig, ax = plt.subplots(figsize=(7, 4))
lengths = space.lengths_[inner]
ax.plot(np.unique(lengths), [hits[lengths == length].mean() for length in np.unique(lengths)], 'o-')
ax.set_xlabel('word length')
ax.set_ylabel('fraction of cells where the image hits the target')
ax.set_ylim(0, 1.05)
zero = not (vanishing.values('cells.first')[-1].any() or vanishing.values('cells.second')[-1].any())
ax.set_title(f'(0, 1) at the identity maps to 0: {zero}')
plt.show()
