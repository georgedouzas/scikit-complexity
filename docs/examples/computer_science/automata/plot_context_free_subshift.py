"""
# Context-free subshift

The cells sit on a ring of 60 cells and hold 0, 1, or 2, in a configuration where every block `0 1^h 2^k 0` between
consecutive zeros has `h = k`. The subshift is not sofic, since no finite graph can match the counts of ones and twos.
Every step proposes to change every cell, in a random order, to one of the two other symbols, and rejects the changes
that would break a block, so the chain samples the admissible rings uniformly.
"""

# %%
# The space-time image shows the ring at every step, starting from balanced blocks `0 1^h 2^h`.

import re

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import ContextFreeSubshiftSystem

SYSTEM = 'context_free_subshift'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(ContextFreeSubshiftSystem(size=60, random_state=0), observers=[observer], random_state=0)

# %%
# Blocks and symbols
# ------------------
# No visited ring should hold an unbalanced block `0 1^h 2^k 0`. Blocks between zeros that hold the factor `21` escape
# the constraint, so they spread as the chain mixes, and the frequencies of the three symbols move away from those of
# the balanced start.

symbols = trajectory.get_values('cells.symbol').astype(int)
pattern = re.compile('1*2*')


def unbalanced(ring):
    zeros = np.flatnonzero(ring == 0)
    if zeros.size == 0:
        return 0
    blocks = ''.join(map(str, np.roll(ring, -zeros[0])))[1:].split('0')
    return sum(1 for block in blocks if pattern.fullmatch(block) and block.count('1') != block.count('2'))


fig, (image, check) = plt.subplots(1, 2, figsize=(10, 4))
image.imshow(symbols, aspect='auto', cmap='viridis', interpolation='nearest')
image.set_xlabel('cell n')
image.set_ylabel('step')
image.set_title('Rings of the context-free subshift')
for symbol in range(3):
    check.plot(trajectory.times, (symbols == symbol).mean(axis=1), label=f'symbol {symbol}')
check.set_xlabel('step')
check.set_ylabel('frequency')
check.set_title(f'Unbalanced blocks visited: {sum(unbalanced(ring) for ring in symbols)}')
check.legend()
fig.tight_layout()
plt.show()
