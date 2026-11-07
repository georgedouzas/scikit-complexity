"""
# Even and S-gap subshifts

The cells are a window of 120 cells of a binary configuration whose runs of zeros between consecutive ones have even
lengths, the even subshift, the S-gap subshift for the even numbers. The configuration is drawn from the measure of
maximal entropy, whose gaps are independent with the probability `φ^-(h + 1)` for the golden ratio `φ`, and every step
shifts the window by one cell, with the cell that enters drawn from the gap since the last one.
"""

# %%
# The space-time image shows the window at every step, the configuration sliding to the left.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import GapSubshiftSystem

SYSTEM = 'gap_subshift'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(GapSubshiftSystem(size=120, random_state=0), until=2000, observers=[observer], random_state=0)

# %%
# Gaps and entropy
# ----------------
# Every gap between consecutive ones should be even, and the gaps should follow `φ^-(h + 1)`, so their histogram falls
# by `φ^2` per allowed length. The entropy of the even subshift is `log φ`, and the gap set `{1, 4}` gives another
# S-gap subshift whose gaps take only those lengths.

alive = trajectory.values('cells.alive')
stream = np.concatenate([alive[0], alive[1:, -1]])
ones = np.flatnonzero(stream)
gaps = np.diff(ones) - 1
golden = (1 + 5**0.5) / 2
lengths = np.arange(0, 12, 2)
other = GapSubshiftSystem(size=120, gaps=(1, 4), random_state=0)
other_alive = simulate(other, until=500, random_state=0).values('cells.alive')
other_ones = np.flatnonzero(np.concatenate([other_alive[0], other_alive[1:, -1]]))
fig, (histogram, other_ax) = plt.subplots(1, 2, figsize=(10, 4))
histogram.bar(*np.unique(gaps, return_counts=True), label='gaps of the stream')
histogram.plot(lengths, gaps.size * golden ** -(lengths + 1.0), 'o--', color='black', label=r'$\varphi^{-(h + 1)}$')
histogram.set_yscale('log')
histogram.set_xlabel('gap h')
histogram.set_title(f'Even subshift, odd gaps: {int((gaps % 2).sum())}, entropy log φ = {np.log(golden):.3f}')
histogram.legend()
other_ax.bar(*np.unique(np.diff(other_ones) - 1, return_counts=True))
other_ax.set_xlabel('gap h')
other_ax.set_title('S-gap subshift for S = {1, 4}')
fig.tight_layout()
plt.show()
