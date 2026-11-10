"""
# Hyperbolic chemotaxis

Cells and the chemical they secrete move with finite speed and turn at random, the hyperbolic limit of a kinetic model
of velocity jumps. Two bumps of cells spread out, conserving their mass, while the chemical accumulates where the cells
are and draws them together.
"""

# %%
# The example builds `HyperbolicChemotaxisSystem` with the test of Bellomo et al. (2017, section 6.6.2), and integrated
# with the implicit BDF method, since the turning rates are large. The observer saves every frame with the default
# blueprint, whose primary view is the image of the cell density.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import HyperbolicChemotaxisSystem

SYSTEM = 'hyperbolic_chemotaxis'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = ScipyODESolver(method='BDF')
trajectory = simulate(HyperbolicChemotaxisSystem(), solver=solver, observers=[observer])

# %%
# Spreading with a conserved mass
# -------------------------------
# The total cell mass should stay at `2 n0 = 0.5` while the peak density falls, and the chemical should grow where the
# cells are.

cells, chemical = trajectory.get_lattice('tissue.n'), trajectory.get_lattice('tissue.c')
area = (0.8 / 40) ** 2
fig, (start, end, mass) = plt.subplots(1, 3, figsize=(14, 4.2))
extent = (-0.4, 0.4, -0.4, 0.4)
start.imshow(cells[0].T, origin='lower', extent=extent, cmap='viridis')
start.set_title('Cells at t = 0')
end.imshow(cells[-1].T, origin='lower', extent=extent, cmap='viridis')
end.contour(chemical[-1].T, origin='lower', extent=extent, colors='white', linewidths=0.8)
end.set_title('Cells and chemical (contours) at t = 0.015')
mass.plot(trajectory.times, cells.sum(axis=(1, 2)) * area, label='cell mass')
mass.plot(trajectory.times, chemical.sum(axis=(1, 2)) * area, label='chemical mass')
mass.set_xlabel('time')
mass.set_ylabel('mass')
mass.set_title('The cell mass is conserved')
mass.legend()
fig.tight_layout()
plt.show()
