"""
# Group-valued difference automaton

The cells sit on a ring of 60 cells and hold elements of the symmetric group of degree 3, and every cell becomes
`x(n)^-1 x(n + 1)`, the difference of a cell and its right neighbour in the group. Over the integers the automaton is
surjective but not injective, since every constant configuration maps to the identity. On the ring the product of the
image around the ring is the identity.
"""

# %%
# The run starts from random elements, numbered by Sage's Cayley table with the identity 0.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import GroupDifferenceAutomatonSystem

SYSTEM = 'group_difference_automaton'

system = GroupDifferenceAutomatonSystem(size=60, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer])

# %%
# Constants and the product around the ring
# -----------------------------------------
# Every constant configuration should map to the identity, so the six constants share one image. The ordered product
# of the image around the ring telescopes to `x(0)^-1 x(0)`, the identity, at every step.

product = system.prepare().system_.interactions[0].rule.keywords['product']
values = trajectory.get_values('cells.value').astype(int)
loops = []
for row in values[1:]:
    total = 0
    for element in row:
        total = product[total, element]
    loops.append(total)
constants = [
    simulate(GroupDifferenceAutomatonSystem(size=10, initial=np.full(10, element)), until=1) for element in range(6)
]
fig, (image, check) = plt.subplots(1, 2, figsize=(10, 4))
image.imshow(values, aspect='auto', cmap='tab10', interpolation='nearest')
image.set_xlabel('cell n')
image.set_ylabel('step')
image.set_title('Elements of S3 along the ring')
check.plot(np.arange(1, values.shape[0]), loops, 'o')
check.set_xlabel('step')
check.set_ylabel('product of the image around the ring')
identity = all(not trajectory.get_values('cells.value')[-1].any() for trajectory in constants)
check.set_title(f'Every constant maps to the identity: {identity}')
fig.tight_layout()
plt.show()
