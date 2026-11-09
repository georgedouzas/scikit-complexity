"""
# Turing machine

A Turing machine is a head with an internal state that reads the symbol under it, writes a symbol, moves one cell left
or right, and changes its state, following an action table. The busy beaver of three states and two symbols writes the
most ones any halting machine of its size can write on a blank tape: six ones in 13 steps.
"""

# %%
# The tape is a ring of 21 blank cells with the head in state A on the middle cell. Every step is a local rule on the
# tape, which moves the head and its state with it. The default blueprint shows the tape and the head.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import TuringMachineSystem

SYSTEM = 'turing_machine'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(TuringMachineSystem(), observers=[observer], random_state=0)

# %%
# The busy beaver at work
# -----------------------
# Each row is the tape at one step, ones in yellow, and the marker shows the head and its state. The head should halt,
# state H, at step 13 with six ones on the tape, which then stays unchanged.

symbols = np.asarray(trajectory.get_lattice('tape.symbol'))
heads = np.asarray(trajectory.get_lattice('tape.head'))
names = 'ABCH'
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(symbols[:15], cmap='viridis')
for step in range(15):
    cell = int(np.argmax(heads[step] >= 0))
    ax.text(cell, step, names[int(heads[step, cell])], ha='center', va='center', color='red')
ax.set_xlabel('cell of the tape')
ax.set_ylabel('step')
ax.set_title(f'Busy beaver: {int(symbols[13].sum())} ones after 13 steps')
plt.show()
