"""
# Automaton over an unbounded alphabet

Each cell adds its left neighbour's value to its own, so from a single 1 the line writes out Pascal's triangle: after
`t` steps the cells hold the binomial coefficients `C(t, k)`, whose largest values grow as `2^t`, so no finite alphabet
can hold the states.
"""

# %%
# The ring has 101 cells and the rule runs for 50 steps from a single 1. The model's space-time view shows the raw
# values, in which the last rows outshine all others. After the run the same space-time diagram is logged once more on a
# logarithmic colour scale, and the blueprint puts it first, as the primary view.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.special import comb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'unbounded_alphabet_automaton'

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(origin='/log_values', name='log10 of the values'),
        rrb.Spatial2DView(origin='/cells', name='values'),
        column_shares=[3, 2],
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=50)
values = simulation.trajectory_.lattice('cells.value')
logarithm = np.where(values > 0, np.log10(np.maximum(values, 1)) + 1, 0)
image = plt.get_cmap('magma')(logarithm / logarithm.max())[:, :, :3]
observer.recording_.log('log_values', rr.Image((255 * image).astype(np.uint8)), static=True)
observer.recording_.flush()

# %%
# Binomial rows
# -------------
# Row `t` should equal the binomial coefficients `C(t, k)` exactly, centred on the starting cell, so its sum should be
# `2^t` and its largest value `C(t, ⌊t/2⌋) ≈ 2^t √(2 / (π t))`, which already exceeds `10^14` after 50 steps.

steps = np.arange(values.shape[0])
fig, (rows, growth) = plt.subplots(1, 2, figsize=(11, 4))
cells = np.arange(values.shape[1]) - 50
for t in (10, 30, 50):
    rows.semilogy(cells, np.where(values[t] > 0, values[t], np.nan), 'o', markersize=3, label=f'step {t}')
    rows.semilogy(np.arange(t + 1), comb(t, np.arange(t + 1)), 'k', linewidth=0.6)
rows.set_xlabel('cell, from the starting one')
rows.set_ylabel('value')
rows.set_title('Rows of Pascal triangle (lines: C(t, k))')
rows.legend()
growth.semilogy(steps, values.sum(axis=1), 'o', markersize=3, label='row sum')
growth.semilogy(steps, values.max(axis=1), 's', markersize=3, label='largest value')
growth.semilogy(steps, 2.0**steps, 'k', linewidth=0.8, label='$2^t$')
growth.semilogy(steps[1:], 2.0 ** steps[1:] * np.sqrt(2 / (np.pi * steps[1:])), 'k:', label=r'$2^t \sqrt{2 / \pi t}$')
growth.set_xlabel('step t')
growth.set_ylabel('value')
growth.set_title('The alphabet grows without bound')
growth.legend()
fig.tight_layout()
plt.show()
