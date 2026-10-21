"""
# HighLife

HighLife, `B36/S23`, differs from the Game of Life only by also giving birth to cells with six live neighbours. From a
random soup it behaves much like Life: the soup burns out into a sparser board of still lifes, blinkers, and gliders.
The extra birth rule shows on special patterns, such as the replicator, which copies itself without end.
"""

# %%
# The torus has 100 by 100 cells, three in ten alive at the start, and the run lasts 200 generations. The image shows
# live cells in yellow.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'highlife'

system = create(MODEL, shape=(100, 100), density=0.3, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer]).run(until=200)

# %%
# Soup decay and the replicator
# -----------------------------
# The density of the soup should fall from 0.3 and level off within 200 generations, near 0.1, a little above
# Life's decay from the same start. The replicator, a pattern of 12 cells, should copy itself along a diagonal, its
# live cells rising in bursts as the copies multiply on the small torus, while the same pattern under Life's rule
# `B3/S23` should settle into a few still lifes within about 25 generations.

life = Simulation(create(MODEL, shape=(100, 100), rule='B3/S23', density=0.3, random_state=0), dt=1).run(until=200)
replicator = np.zeros((64, 64))
replicator[30:35, 30:35] = [[0, 0, 1, 1, 1], [0, 1, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 1, 0], [1, 1, 1, 0, 0]]
grown = {
    rule: Simulation(create(MODEL, shape=(64, 64), rule=rule, initial=replicator), dt=1)
    .run(until=90)
    .trajectory_.lattice('cells.alive')
    .sum(axis=(1, 2))
    for rule in ('B36/S23', 'B3/S23')
}
fig, (soup, copies) = plt.subplots(1, 2, figsize=(11, 4))
times = simulation.trajectory_.times
soup.plot(times, simulation.trajectory_.lattice('cells.alive').mean(axis=(1, 2)), label='HighLife B36/S23')
soup.plot(times, life.trajectory_.lattice('cells.alive').mean(axis=(1, 2)), label='Life B3/S23')
soup.set_xlabel('generation')
soup.set_ylabel('density of live cells')
soup.set_title('Random soups decay alike')
soup.legend()
for rule, counts in grown.items():
    copies.plot(np.arange(len(counts)), counts, label=rule)
copies.set_xlabel('generation')
copies.set_ylabel('live cells')
copies.set_title('The replicator grows only in HighLife')
copies.legend()
fig.tight_layout()
plt.show()
