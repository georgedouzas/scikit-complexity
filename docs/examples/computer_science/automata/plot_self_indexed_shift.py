"""
# Self-indexed shift over a group alphabet

Every cell holds an element of the group and takes the value of the cell at the product of its site and that value,
`τ(x)(g) = x(g · x(g))`. The cell it reads moves with the configuration, so on an infinite group no finite memory set
describes the map, which is continuous and equivariant but no cellular automaton. The run truncates the free group of
rank 2 to a ball of radius 4, where a product outside the ball reads `-1`, an element that is no element.
"""

# %%
# The ball holds 161 elements on rings by word length, colored by the number of the element each cell holds.

import matplotlib.pyplot as plt
import numpy as np
from sage.all__sagemath_gap import SymmetricGroup

from skcomplex.core.components.spaces import StateTranslationNeighborhood
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import SelfIndexedShiftSystem

SYSTEM = 'self_indexed_shift'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(SelfIndexedShiftSystem(radius=4, random_state=0), observers=[observer])

# %%
# Truncation and equivariance
# ---------------------------
# On the ball, a cell whose product leaves the ball loses its element and keeps losing it, so the fraction of cells
# without an element should only grow, the price of the truncation. On a finite group, the symmetric group of degree
# 4 with every product a site, translating a configuration by an element `h` should translate its image by `h`, the
# equivariance `τ(h x) = h τ(x)` that holds on every group.

values = trajectory.get_values('cells.value')
lost = (values == -1).mean(axis=1)
finite = SelfIndexedShiftSystem(group=SymmetricGroup(4), radius=None, random_state=1)
space = finite.prepare().system_.elements[0].space.prepare()
products = space._get_translation_table(StateTranslationNeighborhood())[:, :, 0]
start = finite.prepare().system_.elements[0].values['value']
image = simulate(finite, until=1).get_values('cells.value')[-1]
gaps = []
for element in range(space.size_):
    inverse = int(np.flatnonzero(products[element] == 0)[0])
    moved = SelfIndexedShiftSystem(group=SymmetricGroup(4), radius=None, initial=start[products[inverse]])
    moved_image = simulate(moved, until=1).get_values('cells.value')[-1]
    gaps.append(int((moved_image != image[products[inverse]]).sum()))
fig, (truncation, equivariance) = plt.subplots(1, 2, figsize=(10, 4))
truncation.plot(trajectory.times, lost, 'o-')
truncation.set_xlabel('step')
truncation.set_ylabel('fraction of cells without an element')
truncation.set_title('Products leave the ball of F2')
equivariance.plot(np.arange(space.size_), gaps, 'o')
equivariance.set_xlabel('translating element h of S4')
equivariance.set_ylabel(r'cells where $\tau(hx) \neq h\tau(x)$')
equivariance.set_ylim(-0.5, 1)
equivariance.set_title('Equivariant on a finite group')
fig.tight_layout()
plt.show()
