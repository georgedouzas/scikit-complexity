"""
# Totally asymmetric simple exclusion process

Particles hop one site to the right with probability `p` at every step, all at once, but only onto empty sites. With
`p < 1` a stopped particle blocks the ones behind it, so on a ring at half filling the traffic breaks into jams that
drift backwards while free particles move forwards. The space-time image shows the occupations of a ring of 200
sites over the last frames, one row per step with the earliest at the top.
"""

# %%
# The model is created by name from the catalog on a ring at density `0.5` with hop probability `p = 0.5`. The
# observer saves every frame with the default blueprint, whose primary view is the space-time image of the
# occupations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'tasep'

system = create(MODEL, n_sites=200, p=0.5, ring=True, density=0.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=250)

# %%
# Fundamental diagram of the ring
# -------------------------------
# A particle at site `i` hops when the site ahead was empty and the site it left is empty at the next step, so the
# current is the number of such hops per site and step. For the parallel update the exact current on a ring at
# density `rho` is `J = (1 - sqrt(1 - 4 p rho (1 - rho))) / 2`, which is symmetric about half filling and peaks
# there. The measured currents for two hop probabilities should lie on these curves.


def current(occupied):
    """Return the mean number of hops per site and step of the occupations of a ring."""
    before, after = occupied[:-1], occupied[1:]
    hops = before * (1 - np.roll(before, -1, axis=1)) * (1 - after)
    return hops.sum(axis=1).mean() / occupied.shape[1]


densities = np.linspace(0.05, 0.95, 19)
grid = np.linspace(0.0, 1.0, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for p, color in ((0.5, 'tab:blue'), (0.9, 'tab:orange')):
    measured = []
    for density in densities:
        system_p = create(MODEL, n_sites=400, p=p, ring=True, density=density, random_state=1)
        run = Simulation(system_p, dt=1, random_state=1).run(until=1000)
        measured.append(current(run.trajectory_['lane.occupied'][200:]))
    ax.plot(densities, measured, 'o', color=color, label=f'simulation, p = {p}')
    ax.plot(grid, (1 - np.sqrt(1 - 4 * p * grid * (1 - grid))) / 2, color=color, label=f'exact, p = {p}')
ax.set_xlabel('density rho')
ax.set_ylabel('current J')
ax.set_title('Fundamental diagram of the parallel-update TASEP on a ring')
ax.legend()
fig.tight_layout()
plt.show()
