"""
# Cellular Potts model of cell sorting

Every site of a lattice belongs to one cell, and the sites of one cell form a region. Monte Carlo moves copy the label
of a random neighbour into a random site, accepted from the change of the energy
`sum_<i,j> w_ij J(t_i, t_j) [s_i != s_j] + lambda sum_s (a_s - A)^2`: the adhesion between the kinds of the cells that
touch, and an elastic constraint on the area of every cell. When the adhesion between cells of different kinds exceeds
the mean of the adhesions within one kind, the kinds sort into patches, the differential adhesion of Steinberg as
modelled by Graner and Glazier. Nemati and de Graaf run the same model on a disordered lattice, the Voronoi
tessellation of irregular points, whose contacts are weighted by the length of the shared sides.
"""

# %%
# Sorting on a square lattice
# ---------------------------
# Forty cells of two kinds start as 5 by 8 rectangles in random order. The more cohesive kind gathers into patches.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.physiology import CellularPottsSystem

SYSTEM = 'cellular_potts'
SHAPE = (40, 40)

system = CellularPottsSystem(shape=SHAPE)
series = simulate(system, until=3000)
kinds = np.asarray(system.system_.interactions[0].kinds)
cells = np.rint(series.values('tissue.cell')).astype(int).reshape(-1, *SHAPE)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, frame, title in ((axes[0], cells[0], 'start'), (axes[1], cells[-1], 'after 3000 sweeps')):
    ax.imshow(kinds[frame], cmap='coolwarm', interpolation='nearest')
    boundary = (frame != np.roll(frame, 1, axis=0)) | (frame != np.roll(frame, 1, axis=1))
    ax.imshow(np.ma.masked_where(~boundary, boundary), cmap='gray_r', interpolation='nearest', alpha=0.6)
    ax.set_title(title)
    ax.set_axis_off()
plt.show()

# %%
# Homotypic contacts and areas
# ----------------------------
# The share of the contact between different cells that joins cells of one kind grows with sorting, while the areas
# stay near the target of 40 sites. With equal adhesions the kinds stay mixed.

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(series.values('homotypic_contacts'), label='differential adhesion')
mixed = simulate(CellularPottsSystem(shape=SHAPE, adhesion=((1.5, 1.5), (1.5, 1.5))), until=3000)
ax.plot(mixed.values('homotypic_contacts'), label='equal adhesion')
ax.set_xlabel('sweep')
ax.set_ylabel('share of homotypic contacts')
deviation = series.values('area_deviation')[100:].mean()
ax.set_title(f'Mean relative deviation of the areas from 40 sites: {deviation:.3f}')
ax.legend()
plt.show()

# %%
# Sorting on a disordered lattice
# -------------------------------
# On the Voronoi lattice the sites have their own areas and contacts, so the cell boundaries carry no preferred
# direction of the square lattice.

system = CellularPottsSystem(shape=SHAPE, lattice='disordered')
series = simulate(system, until=3000)
space = system.system_.elements[0].space
kinds = np.asarray(system.system_.interactions[0].kinds)
final = np.rint(series.values('tissue.cell')[-1]).astype(int).ravel()
x, y = space.positions_.T

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].scatter(x, y, c=kinds[final], cmap='coolwarm', s=6)
axes[0].set_aspect('equal')
axes[0].set_axis_off()
axes[0].set_title('kinds after 3000 sweeps')
axes[1].plot(series.values('homotypic_contacts'))
axes[1].set_xlabel('sweep')
axes[1].set_ylabel('share of homotypic contacts')
plt.show()
