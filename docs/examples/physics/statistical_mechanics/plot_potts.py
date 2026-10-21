"""
# Potts

Each site of the q-state Potts model takes one of `q` colors and pays an energy `-J` for each neighbour of the same
color. Quenched from random colors below the critical temperature `T_c = J / ln(1 + sqrt q)`, the lattice coarsens
into domains of the `q` colors that meet at junctions of three. The view shows the colors of a 256 x 256 torus with
`q = 5` some sweeps after the quench.
"""

# %%
# The model is created by name from the catalog with `q = 5` at `T = 0.6 J`, below `T_c = 0.85 J`. One unit of time is
# one sweep of heat-bath updates. The observer saves every frame with the default blueprint, whose primary view is the
# image of the colors.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'potts'

system = create(MODEL, size=256, q=5, temperature=0.6, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=100)

# %%
# Energy through the transition
# -----------------------------
# The energy per site `e = -J (like neighbour pairs) / N` falls from zero at infinite temperature to `-2 J` in the
# ordered phase. Baxter's exact solution puts it at `e_c = -J (1 + 1 / sqrt q)` at `T_c`, and the transition is
# continuous for `q <= 4` and discontinuous, with a latent heat, above. On a 40 x 40 lattice the energy of `q = 3`
# should bend smoothly through its exact critical value, while that of `q = 8` should jump across it at `T_c`.


def energy(colors, side):
    """Return the energy per site of each frame of the colors on a square torus."""
    grid = colors.reshape(-1, side, side)
    like = (grid == np.roll(grid, 1, axis=1)).sum(axis=(1, 2)) + (grid == np.roll(grid, 1, axis=2)).sum(axis=(1, 2))
    return -like / side**2


ratios = np.linspace(0.8, 1.2, 13)
fig, ax = plt.subplots(figsize=(7, 4))
for q, color in ((3, 'tab:blue'), (8, 'tab:orange')):
    critical = 1 / np.log(1 + np.sqrt(q))
    measured = []
    for ratio in ratios:
        system_q = create(MODEL, size=40, q=q, temperature=ratio * critical, random_state=1)
        run = Simulation(system_q, dt=1, random_state=1).run(until=1000)
        measured.append(energy(run.trajectory_['spins.color'][300:], 40).mean())
    ax.plot(ratios, measured, 'o-', color=color, label=f'q = {q}')
    ax.plot(1.0, -(1 + 1 / np.sqrt(q)), '*', color=color, markersize=14, markeredgecolor='black')
ax.axvline(1.0, color='gray', linestyle='--', label='T_c, stars at the exact e_c')
ax.set_xlabel('temperature T / T_c')
ax.set_ylabel('energy per site e / J')
ax.set_title('Smooth for q = 3, a jump for q = 8')
ax.legend()
fig.tight_layout()
plt.show()
