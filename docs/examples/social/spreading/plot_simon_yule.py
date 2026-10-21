"""
# Simon-Yule text growth

A text grows word by word: with a small probability the next word is new, and otherwise it repeats a word already in
the text, chosen in proportion to how often it has been used. The rich get richer, so a few words come to dominate and
the number of distinct words grows in proportion to the length of the text.
"""

# %%
# The model is created by name from the catalog with the innovation probability 0.1 and 100 words written per step, and
# runs for 100 steps, a text of 10,000 words. The default blueprint shows the number of distinct words, which grows
# linearly.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'simon_yule'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=100)

# %%
# Zipf's law and vocabulary growth
# --------------------------------
# The count of the word of rank `r` should fall as `r^(-1 / (1 - innovation))`, a slope of -1.11 on logarithmic axes,
# away from the first word, which has been repeated since the start, and the number of distinct words should be close
# to `innovation` times the length of the text.

trajectory = simulation.trajectory_
counts = np.sort(trajectory.agents('words', len(trajectory.times) - 1).values['count'])[::-1]
ranks = np.arange(1, len(counts) + 1)
lengths = trajectory.times * 100
distinct = [len(trajectory.agents('words', frame).ids) for frame in range(len(trajectory.times))]
fig, (zipf, growth) = plt.subplots(1, 2, figsize=(10, 4))
zipf.loglog(ranks, counts, '.', label='simulation')
zipf.loglog(ranks, counts[9] * (ranks / 10) ** (-1 / 0.9), color='black', label='slope -1 / (1 - 0.1)')
zipf.set_xlabel('rank of the word')
zipf.set_ylabel('count')
zipf.set_title('Rank-frequency: Zipf law')
zipf.legend()
growth.plot(lengths, distinct, label='simulation')
growth.plot(lengths, 0.1 * lengths, color='black', linestyle='--', label='0.1 x length')
growth.set_xlabel('length of the text')
growth.set_ylabel('distinct words')
growth.set_title('Vocabulary grows linearly')
growth.legend()
fig.tight_layout()
plt.show()
