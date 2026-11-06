"""
# Morse subshift

The Thue-Morse substitution `0 → 01`, `1 → 10` rewrites the word `0`, and generation `n` is the prefix
`T(0) ... T(2^n - 1)` of the Thue-Morse sequence, whose term `T(k)` is the parity of the ones in the binary expansion
of `k`. The orbit closure of the sequence under the shift is the Morse subshift, an infinite minimal subshift that is
not sofic, whose words are the factors of the generations.
"""

# %%
# The word doubles at every generation, as the length shows.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.compile import RewritingCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'morse_subshift'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
simulation = Simulation(create_system(SYSTEM), observers=[observer], compiler=RewritingCompiler()).prepare().run(12)

# %%
# Factor complexity
# -----------------
# The last word should match the parities of the binary expansions. A minimal subshift of zero entropy has few
# factors: the number of distinct words of length `m` in the Thue-Morse sequence stays below `10 m / 3`, far below the
# `2^m` of the full shift, so the entropy `log p(m) / m` falls to 0.

word = ''.join(str(code) for code in simulation.trajectory_.structure('morse', 12).codes)
parities = ''.join(str(bin(k).count('1') % 2) for k in range(len(word)))
lengths = np.arange(1, 40)
factors = np.array([len({word[start : start + m] for start in range(len(word) - m + 1)}) for m in lengths])
fig, (complexity, entropy) = plt.subplots(1, 2, figsize=(10, 4))
complexity.plot(lengths, factors, 'o', label='words of length m')
complexity.plot(lengths, 10 * lengths / 3, '--', color='black', label='10 m / 3')
complexity.set_xlabel('m')
complexity.set_title(f'Generation 12 is the Thue-Morse prefix: {word == parities}')
complexity.legend()
entropy.plot(lengths, np.log2(factors) / lengths, 'o-')
entropy.set_xlabel('m')
entropy.set_ylabel('log2 p(m) / m')
entropy.set_title('Entropy of the Morse subshift tends to 0')
fig.tight_layout()
plt.show()
