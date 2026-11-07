"""
# Golden mean shift

The cells sit on a ring, the cyclic group of order 200, and start empty. Every step visits them in a random order,
and each proposes to flip, which it does unless a one would sit next to another one. The chain samples the golden mean
shift, the configurations without two adjacent ones, whose number grows as the power of the golden ratio.
"""

# %%
# The space-time image shows the ring at every step, the oldest at the top, with ones never side by side.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import automaton
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.computer_science.automata import SubshiftsSystem

SYSTEM = 'subshifts'

system = SubshiftsSystem(size=200)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Density and entropy
# -------------------
# The chain proposes symmetric moves and rejects the forbidden ones, so it samples every admissible configuration with
# the same probability. The density of ones should approach `(5 - sqrt 5) / 10`, about 0.276, the frequency of ones
# of the measure of maximal entropy, and no configuration should ever hold a forbidden pattern. The entropy of the
# words of length `n` approaches the logarithm of the golden ratio.

alive = trajectory.values('cells.alive')
admissible = all(automaton(system, kind='admissible', configuration=row.astype(int)).admissible for row in alive)
lengths = np.arange(2, 31, 4)
windows = [LatticeSpace(int(length), boundary='fixed') for length in lengths]
entropies = automaton(system, kind='entropy', windows=windows).entropy
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
left.plot(trajectory.times, alive.mean(axis=1), label='density of ones')
left.axhline((5 - np.sqrt(5)) / 10, linestyle='--', color='gray', label='maximal entropy')
left.set_xlabel('step')
left.set_ylabel('density of ones')
left.set_title(f'Every frame admissible: {admissible}')
left.legend()
right.plot(lengths, entropies, 'o-', label='$\\log_2 |X_n| / n$')
right.axhline(np.log2((1 + np.sqrt(5)) / 2), linestyle='--', color='gray', label='$\\log_2 \\varphi$')
right.set_xlabel('word length $n$')
right.set_ylabel('entropy per cell (bits)')
right.set_title('Entropy of the golden mean shift')
right.legend()
plt.show()
