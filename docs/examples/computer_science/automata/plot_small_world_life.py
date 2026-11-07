"""
# Game of Life on a small-world network

The Game of Life is a rule on the eight Moore neighbours of every cell, so it runs on any network whose nodes have eight
neighbours. Rewiring the Moore lattice of a torus by swapping the ends of random pairs of links keeps every degree at
eight and adds shortcuts, a small-world network, and the density of life the automaton settles at changes with the
fraction of rewired links.
"""

# %%
# The torus has 50 by 50 cells, three in ten alive at the start, and the network is not rewired, so the run is the Game
# of Life itself. The default blueprint shows the density of live cells.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'small_world_life'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# Density against the rewiring probability
# ----------------------------------------
# The settled density of live cells is the order parameter. A few shortcuts should lower it below the lattice value,
# since they break the gliders and oscillators that keep the lattice alive, while a fully random network of degree
# eight should sustain a much denser, disordered life.

probabilities = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
densities = [
    np.mean(
        [
            create_simulation(SYSTEM, p=p, random_state=seed)
            .prepare()
            .run(until=150)
            .trajectory_.values('cells.alive')[-50:]
            .mean()
            for seed in range(3)
        ],
    )
    for p in probabilities
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx([max(p, 1e-3) for p in probabilities], densities, marker='o')
ax.set_xlabel('rewiring probability p (0 drawn at 0.001)')
ax.set_ylabel('settled density of live cells')
ax.set_title('Life on a small world')
plt.show()
