"""
# Markov source

A two-symbol Markov source stays on symbol 0 for long stretches and leaves symbol 1 quickly, since symbol 0 repeats
with probability 0.9 and symbol 1 with probability 0.6.
"""

# %%
# The source with the transition matrix `[[0.9, 0.1], [0.4, 0.6]]` emits 300 symbols. The blueprint shows the
# sequence of symbols over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import block_entropies
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'markov_source'
TRANSITION = np.array([[0.9, 0.1], [0.4, 0.6]])

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/source/symbol', name='symbol', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, transition=TRANSITION), dt=1, observers=[observer], random_state=0).run(299)

# %%
# Entropy rate
# ------------
# The entropy rate of a Markov source is `h = -sum_i pi_i sum_j P_ij log2 P_ij` for the stationary distribution
# `pi = (0.8, 0.2)`, about 0.569 bits per symbol. The differences `S_m - S_{m-1}` of the block entropies of a longer
# run of 100000 symbols should drop from the entropy of single symbols to `h` at `m = 2` and stay there, since one
# symbol of the past holds all the memory of the source.

symbols = Simulation(create(MODEL, transition=TRANSITION), dt=1, random_state=1).run(100000).trajectory_
entropies = block_entropies(symbols['source.symbol'].astype(int), 8)
stationary = np.array([0.8, 0.2])
rate = -np.sum(stationary[:, None] * TRANSITION * np.log2(TRANSITION))
fig, (sequence, rates) = plt.subplots(1, 2, figsize=(10, 4))
sequence.step(simulation.trajectory_.times, simulation.trajectory_['source.symbol'], where='post')
sequence.set_xlabel('step')
sequence.set_ylabel('symbol')
sequence.set_title('Long stretches of symbol 0')
rates.plot(range(1, 9), np.diff(entropies), 'o-', label='$S_m - S_{m-1}$')
rates.axhline(rate, linestyle='--', color='black', label=f'entropy rate {rate:.3f} bits')
rates.set_xlabel('block length m')
rates.set_ylabel('bits per symbol')
rates.set_title('Block entropies reach the entropy rate')
rates.legend()
fig.tight_layout()
plt.show()
