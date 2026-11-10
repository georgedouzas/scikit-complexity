"""
# Triadic percolation

Every link of a random network has a regulating node that switches it on or off. A negative regulator lets its link
be active only while the regulator lies outside the giant component. A large giant component therefore turns its
links off, the network falls apart, and the regulators turn the links on again. Above the percolation point the giant
component alternates between a large and a small size, a cycle of period two.
"""

# %%
# The network is an Erdős-Rényi graph of 1000 nodes with mean degree 4, and every regulator is negative. A link the
# regulation allows is active with probability `p = 0.5`. Every step of the simulation is a time step of Millán et al.
# in two phases: the links follow their regulators, and then the nodes take the giant component of the active links.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.networks import TriadicPercolationSystem

SYSTEM = 'triadic_percolation'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(TriadicPercolationSystem(), observers=[observer], random_state=0)


def giant_fraction(mean_degree):
    """Return the giant component of an Erdős-Rényi graph of a mean degree, the root of `S = 1 - exp(-c S)`."""
    if mean_degree <= 1:
        return 0.0
    return brentq(lambda size: size - 1 + np.exp(-mean_degree * size), 1e-9, 1.0)


def giant(p):
    """Return the fraction of nodes in the giant component after every time step of Millán et al."""
    run = simulate(TriadicPercolationSystem(p=p), until=40, random_state=0)
    return run.get_values('nodes.active').mean(axis=1)


# %%
# Time series
# -----------
# Below the percolation point `p = 1 / c = 0.25` the giant component should vanish. Above it, the giant component
# should alternate between the size of a random network whose links are active with probability `p`, and almost
# nothing.

fig, axis = plt.subplots(figsize=(6, 4))
for p in (0.15, 0.5, 0.9):
    axis.plot(giant(p)[:30], 'o-', markersize=3, label=f'p = {p}')
axis.set_xlabel('time step of Millán et al.')
axis.set_ylabel('fraction of nodes in the giant component')
axis.set_title('Negative regulation turns percolation into a cycle')
axis.legend()
fig.tight_layout()
plt.show()

# %%
# Orbit diagram
# -------------
# The late values of the giant component against `p` should split into two branches at `p = 0.25`. The upper branch
# should follow the giant component `S(c p)` of the links the regulation allows, and the lower branch should stay near
# zero, as the map `R' = S(c p (1 - R))` predicts.

probabilities = np.linspace(0.05, 1.0, 20)
fig, axis = plt.subplots(figsize=(6, 4))
for p in probabilities:
    late = giant(p)[20:]
    axis.plot(np.full(late.size, p), late, '.', color='tab:blue', markersize=3)
grid = np.linspace(0.05, 1.0, 200)
axis.plot(grid, [giant_fraction(4.0 * p) for p in grid], 'k:', label='S(c p)')
axis.axvline(0.25, color='gray', linestyle='--', label='p = 1 / c')
axis.set_xlabel('p')
axis.set_ylabel('late fraction of nodes in the giant component')
axis.set_title('Period two above the percolation point')
axis.legend()
fig.tight_layout()
plt.show()
