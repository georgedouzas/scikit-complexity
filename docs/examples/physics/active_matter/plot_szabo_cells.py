"""
# Szabó cell system

Two hundred motile cells move at a constant speed in a periodic box. No cell reads the heading of another. Each heading
only turns towards the velocity of its own cell, and adhesion and repulsion between touching cells change those
velocities. At a density above the transition the cells end up moving together, as the keratocytes of Szabó et al.
(2006) do. The points are coloured by heading.
"""

# %%
# The cells have the speed `v0 = 1`, the diameter `R_0 = 1`, the equilibrium distance `R_eq = 5 / 6`, the forces
# `F_rep = 30` and `F_adh = 0.75`, and the noise `eta = 0.6` of the paper, with the number density 1 in a box of side
# 14.1. The Euler-Maruyama step is 0.01, not the 0.05 of the paper, since the pair force of slope `F_rep / R_eq = 36`
# makes a step above `1 / 36` unstable and the pairs then heat the cells until they stop ordering.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import SzaboCellsSystem

SYSTEM = 'szabo_cells'
SIZE = float(np.sqrt(200.0))

box = rrb.Spatial2DView(
    origin='/cells',
    name='cells',
    # Points are drawn with y negated so that y points up, so the vertical range is mirrored.
    visual_bounds=rrb.VisualBounds2D(x_range=(0.0, SIZE), y_range=(-SIZE, 0.0)),
)
blueprint = rrb.Blueprint(
    box,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = SzaboCellsSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint)
trajectory = simulate(system, backend=NumbaBackend(), observers=[observer], random_state=1)


# %%
# Order against density
# ---------------------
# The order parameter is the mean velocity of the cells divided by `v0`, which we read from their displacements between
# frames. It should grow from the disordered start and level off above 0.8 at the density 1, and it should stay lower in
# a dilute box, where the cells meet too seldom to share a direction. The right panel runs 100 cells at three densities
# and shows that the final order grows with the density.


def order_of(trajectory, size):
    """Return the mean normalized velocity of the cells at every frame."""
    moves = np.diff(trajectory.values('cells.x'), axis=0)
    moves -= size * np.round(moves / size)
    return np.linalg.norm(moves.mean(axis=1) / np.diff(trajectory.times)[:, None], axis=-1)


DENSITIES = (0.05, 0.25, 1.0)
CELLS = 100
curves = {}
for density in DENSITIES:
    settings = {'until': 40.0, 'random_state': 1}
    trajectory = simulate(SzaboCellsSystem(n_cells=CELLS, density=density), **settings)
    curves[density] = order_of(trajectory, np.sqrt(CELLS / density))
finals = [curve[-20:].mean() for curve in curves.values()]

fig, (growth, transition) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(trajectory.times[1:], order_of(trajectory, SIZE), label='200 cells at density 1')
growth.plot(0.5 * np.arange(1, 81), curves[0.05], label='100 cells at density 0.05')
growth.axhline(0.8, color='0.6', linestyle='--', label='0.8')
growth.set_xlabel('time')
growth.set_ylabel('order parameter')
growth.set_title('The order grows above 0.8 at density 1')
growth.legend()
transition.plot(DENSITIES, finals, 'o-')
transition.set_xlabel('number density')
transition.set_ylabel('order over the last 10 time units')
transition.set_title('The final order grows with the density')
fig.tight_layout()
plt.show()
