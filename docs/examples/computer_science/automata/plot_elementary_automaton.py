"""
# Elementary cellular automaton

From a single live cell, rule 30 grows a triangle whose left edge is regular and whose interior is a disordered mix of
triangles of every size: a one-dimensional rule of three neighbours that produces apparent randomness.
"""

# %%
# The line has 201 cells and starts with one live cell in the middle, and rule 30 runs for 100 generations, so the
# pattern never reaches the edges. The space-time view stacks the generations from top to bottom.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'elementary_automaton'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(create(MODEL, rule=30, size=201), dt=1, observers=[observer]).run(until=100)

# %%
# Regular and random rules
# ------------------------
# Rule 90 adds its two neighbours modulo 2, so from one cell it draws Pascal's triangle modulo 2, and its number of
# live cells at generation `t` should be exactly `2^b(t)`, with `b(t)` the number of ones in the binary digits of `t`.
# Rule 30 should instead fill its triangle with a density of live cells near 1/2.

rule30 = simulation.trajectory_.lattice('cells.alive')
rule90 = Simulation(create(MODEL, rule=90, size=201), dt=1).run(until=100).trajectory_.lattice('cells.alive')
generations = np.arange(rule90.shape[0])
fig, (counts, density) = plt.subplots(1, 2, figsize=(11, 4))
counts.semilogy(generations, rule90.sum(axis=1), 'o', markersize=3, label='rule 90, simulated')
counts.semilogy(
    generations,
    [2 ** bin(t).count('1') for t in generations],
    color='black',
    linewidth=0.8,
    label='$2^{b(t)}$',
)
counts.set_xlabel('generation t')
counts.set_ylabel('live cells')
counts.set_title('Rule 90: Pascal triangle modulo 2')
counts.legend()
density.plot(generations[1:], rule30.sum(axis=1)[1:] / (2 * generations[1:] + 1), label='rule 30')
density.axhline(0.5, color='black', linestyle=':', label='1/2')
density.set_ylim(0, 1)
density.set_xlabel('generation t')
density.set_ylabel('live share of the light cone')
density.set_title('Rule 30 fills its triangle at random')
density.legend()
fig.tight_layout()
plt.show()
