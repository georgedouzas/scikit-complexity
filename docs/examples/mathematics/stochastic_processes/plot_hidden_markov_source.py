"""
# Hidden Markov source

A hidden chain stays mostly in state 0, with short visits to state 1, and the source emits symbol 0 more often from
state 0, so the emitted symbols, in the second view, follow the hidden state loosely, with errors in both directions.
"""

# %%
# The hidden chain has the transition matrix `[[0.9, 0.1], [0.4, 0.6]]` and emits through `[[0.7, 0.3], [0.2, 0.8]]`
# for 300 steps. The blueprint shows the hidden states and the symbols over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import HiddenMarkovSourceSystem

SYSTEM = 'hidden_markov_source'
TRANSITION = np.array([[0.9, 0.1], [0.4, 0.6]])
EMISSION = np.array([[0.7, 0.3], [0.2, 0.8]])

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/source/hidden', name='hidden state', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(origin='/source/symbol', name='symbol', axis_x=rrb.TimeAxis(view_range=whole_run)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(
    HiddenMarkovSourceSystem(transition=TRANSITION, emission=EMISSION),
    until=299,
    observers=[observer],
    random_state=0,
)

# %%
# Emission frequencies
# --------------------
# In a longer run of 100000 steps, the share of symbol `j` among the steps in hidden state `i` should match the
# emission probability `E_ij`, and the share of symbol 0 overall should match `0.8 x 0.7 + 0.2 x 0.2 = 0.6` for the
# stationary distribution `(0.8, 0.2)` of the hidden chain.

long = simulate(HiddenMarkovSourceSystem(transition=TRANSITION, emission=EMISSION), until=100000, random_state=1)
hidden, symbol = long.values('source.hidden')[1:], long.values('source.symbol')[1:]
measured = np.array([[np.mean(symbol[hidden == i] == j) for j in (0, 1)] for i in (0, 1)])
fig, ax = plt.subplots(figsize=(6, 4))
positions = np.arange(4)
ax.bar(positions - 0.2, EMISSION.ravel(), width=0.4, label='emission matrix')
ax.bar(positions + 0.2, measured.ravel(), width=0.4, label='simulation')
ax.set_xticks(positions, ['0 → 0', '0 → 1', '1 → 0', '1 → 1'])
ax.set_xlabel('hidden state → symbol')
ax.set_ylabel('probability')
ax.set_title(f'Emission frequencies, symbol 0 overall: {np.mean(symbol == 0):.3f}')
ax.legend()
fig.tight_layout()
plt.show()
