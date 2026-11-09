"""
# Adding machine

The cells hold the six lowest binary digits of a 2-adic integer `r`, the lowest first, the first letters of its
infinite word, a vertex of the binary tree. Every step adds 1: the lowest digit turns, and a digit turns when every
lower digit is 1, the carry. The odometer is an automorphism of the tree of infinite order whose orbits are dense, and
on the first six levels it visits all 64 vertices in turn.
"""

# %%
# The image shows the digits at every step, the lowest on the left, counting from 0.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import AddingMachineSystem

SYSTEM = 'adding_machine'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(AddingMachineSystem(digits=6, base=2), observers=[observer])

# %%
# Counting and the portrait
# -------------------------
# The digits should read `t mod 64` at step `t`, so the orbit of 0 visits every vertex of level 6 once before it comes
# back. In base 3 the digit at level `k` turns once every `3^k` steps, the self-similar portrait of the odometer, whose
# restriction to a subtree below the digit `q - 1` is the odometer again.

digits = trajectory.get_values('cells.digit')
numbers = digits @ 2 ** np.arange(6)
ternary = simulate(AddingMachineSystem(digits=4, base=3), until=81).get_values('cells.digit')
turns = (np.diff(ternary, axis=0) != 0).sum(axis=0)
fig, (count, portrait) = plt.subplots(1, 2, figsize=(10, 4))
count.plot(trajectory.times, numbers, 'o', markersize=3, label='digits read in base 2')
count.plot(trajectory.times, trajectory.times % 64, '--', color='black', label='t mod 64')
count.set_xlabel('step t')
count.legend()
count.set_title('The odometer counts through level 6')
portrait.semilogy(np.arange(4), turns, 'o', label='turns of the digit in 81 steps')
portrait.semilogy(np.arange(4), 81 / 3 ** np.arange(4), '--', color='black', label='81 / 3^k')
portrait.set_xlabel('level k')
portrait.legend()
portrait.set_title('Base 3: level k turns every 3^k steps')
fig.tight_layout()
plt.show()
