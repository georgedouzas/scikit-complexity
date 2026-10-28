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

from skcomplex.analysis import AutomatonAnalysis
from skcomplex.core.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create
from skcomplex.models.computer_science.automata import Subshift

MODEL = 'subshifts'

system = create(MODEL, size=200)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, observers=[observer], random_state=0).check().run(until=50)

# %%
# Density and entropy
# -------------------
# The chain proposes symmetric moves and rejects the forbidden ones, so it samples every admissible configuration with
# the same probability. The density of ones should approach `(5 - sqrt 5) / 10`, about 0.276, the frequency of ones
# of the measure of maximal entropy, and no configuration should ever hold a forbidden pattern. The entropy of the
# words of length `n` approaches the logarithm of the golden ratio.

space = system.elements[0].space
alive = simulation.trajectory_.values('cells.alive')
golden = Subshift(LatticeNeighborhood(((0,), (1,))), [(1, 1)])
admissible = all(golden.is_admissible(row.astype(int), space) for row in alive)
lengths = np.arange(2, 31, 4)
windows = [LatticeSpace(int(length), boundary='fixed') for length in lengths]
entropies = AutomatonAnalysis('entropy', windows=windows).check().prepare(golden).run().entropy_
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
left.plot(simulation.trajectory_.times, alive.mean(axis=1), label='density of ones')
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
