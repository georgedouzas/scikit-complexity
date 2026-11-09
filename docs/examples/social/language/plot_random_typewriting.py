"""
# Random typewriting

Miller's monkeys type at random on a typewriter with a few letters and a space bar, and every space ends a word. Long
words are exponentially rare, but there are exponentially many of them, and the two exponentials give Zipf's power
law of word frequencies without any meaning or economy in the text.
"""

# %%
# The typewriter has five letters and a space bar hit with the probability 0.2, and the monkeys type 1000 keys per
# step for 100 steps. The default blueprint shows the number of distinct words.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.language import RandomTypewritingSystem

SYSTEM = 'random_typewriting'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(RandomTypewritingSystem(), until=100, observers=[observer], random_state=0)

# %%
# Zipf's law from two exponentials
# --------------------------------
# The frequency of the word of rank `r` should fall as `r^(-alpha)` with `alpha = 1 - log(1 - q) / log(letters)`, about
# 1.14, in steps, one step per word length, and the mean count of a word should fall by `letters / (1 - q) = 6.25` per
# letter.

words = trajectory.get_agents('words', len(trajectory.times) - 1).values
counts = np.sort(words['count'])[::-1]
ranks = np.arange(1, counts.size + 1)
alpha = 1 - np.log(0.8) / np.log(5)
fig, (zipf, lengths) = plt.subplots(1, 2, figsize=(10, 4))
zipf.loglog(ranks, counts, '.', label='simulation')
zipf.loglog(ranks, counts[29] * (ranks / 30) ** -alpha, color='black', label=f'slope {-alpha:.2f}')
zipf.set_xlabel('rank of the word')
zipf.set_ylabel('count')
zipf.set_title('Rank-frequency')
zipf.legend()
sizes = np.arange(1, 6)
means = [words['count'][words['length'] == size].mean() for size in sizes]
lengths.semilogy(sizes, means, 'o', label='simulation')
lengths.semilogy(sizes, means[0] * 6.25 ** -(sizes - 1.0), color='black', label='factor 1 / 6.25 per letter')
lengths.set_xlabel('letters of the word')
lengths.set_ylabel('mean count')
lengths.set_title('Long words are rare')
lengths.legend()
fig.tight_layout()
plt.show()
