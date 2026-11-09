"""
# Automaton over an unbounded alphabet

Each cell adds its left neighbour's value to its own, so from a single 1 the line writes out Pascal's triangle: after
`t` steps the cells hold the binomial coefficients `C(t, k)`, whose largest values grow as `2^t`, so no finite alphabet
can hold the states.
"""

# %%
# The ring has 101 cells and the rule runs for 50 steps from a single 1. The system's space-time view shows the raw
# values, in which the last rows outshine all others. After the run the same space-time diagram is logged once more on a
# logarithmic colour scale, and the blueprint puts it first, as the primary view. Since every value only moves right,
# the logarithmic view draws the cells the rule reaches, `C(t, k)` for `k <= t`, with row `t` shifted left by half a
# cell per step, so the rows stack into the familiar symmetric Pascal triangle.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.special import comb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import UnboundedAlphabetAutomatonSystem

SYSTEM = 'unbounded_alphabet_automaton'

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
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(UnboundedAlphabetAutomatonSystem(), observers=[observer])
values = trajectory.get_lattice('cells.value')
logarithm = np.where(values > 0, np.log10(np.maximum(values, 1)) + 1, 0)
start, steps_run = values.shape[1] // 2, values.shape[0] - 1
triangle = np.zeros((steps_run + 1, 2 * (steps_run + 1)))
for t in range(steps_run + 1):
    triangle[t, steps_run - t : steps_run + t + 2] = np.repeat(logarithm[t, start : start + t + 1], 2)
image = plt.get_cmap('magma')(triangle / triangle.max())[:, :, :3]
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
