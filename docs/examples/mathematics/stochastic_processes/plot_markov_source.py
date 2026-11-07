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

from skcomplex.core.analysis import EpsilonMachineAnalysis, entropy
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'markov_source'
TRANSITION = np.array([[0.9, 0.1], [0.4, 0.6]])

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/source/symbol', name='symbol', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
simulation = (
    Simulation(create_system(SYSTEM, transition=TRANSITION), dt=1, observers=[observer], random_state=0)
    .prepare()
    .run(299)
)

# %%
# Entropy rate
# ------------
# The entropy rate of a Markov source is `h = -sum_i pi_i sum_j P_ij log2 P_ij` for the stationary distribution
# `pi = (0.8, 0.2)`, about 0.569 bits per symbol. The differences `S_m - S_{m-1}` of the block entropies of a longer
# run of 100000 symbols should drop from the entropy of single symbols to `h` at `m = 2` and stay there, since one
# symbol of the past holds all the memory of the source.

symbols = (
    Simulation(create_system(SYSTEM, transition=TRANSITION), dt=1, random_state=1).prepare().run(100000).trajectory_
)
blocks = entropy(symbols.values('source.symbol').astype(int), kind='block', dimension=8)
entropies = blocks.entropy[0]
stationary = np.array([0.8, 0.2])
rate = -np.sum(stationary[:, None] * TRANSITION * np.log2(TRANSITION))
fig, (sequence, rates) = plt.subplots(1, 2, figsize=(10, 4))
sequence.step(simulation.trajectory_.times, simulation.trajectory_.values('source.symbol'), where='post')
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

# %%
# Causal states
# -------------
# Causal-state splitting groups the pasts of the source by the law of the next symbol. The last symbol fixes that law,
# so the ε-machine should have two causal states, one per symbol, visited 80 and 20 percent of the time. Its
# statistical complexity is then the entropy of `pi`, about 0.722 bits, and its entropy rate is `h`.

source = symbols.values('source.symbol').astype(int)
machine = EpsilonMachineAnalysis(max_length=3).prepare(source).run()
labels = [f'last symbol {state[0][-1]}' for state in machine.histories_]
fig, occupancy = plt.subplots(figsize=(5, 4))
occupancy.bar(labels, machine.stationary_)
occupancy.set_ylabel('fraction of time')
occupancy.set_title(
    f'{len(machine.histories_)} causal states, $C_\\mu$ = {machine.statistical_complexity_:.3f} bits, '
    f'$h_\\mu$ = {machine.entropy_rate_:.3f} bits',
    fontsize=10,
)
fig.tight_layout()
plt.show()
