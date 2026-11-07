"""
# Allen-Cahn

An order parameter that is free to change locally, such as the magnetisation of a quenched magnet, orders into domains
of its two values, and the walls between them move to shorten themselves. Small domains shrink and vanish, so the
pattern coarsens and its domains grow as the square root of time. The image shows the order parameter at the end of the
run: large yellow and purple domains with smooth, gently curved walls.
"""

# %%
# The system is built from its class on a periodic square of 128 sites with unit spacing, from noise of
# amplitude 0.05 around 0. The catalog defaults integrate it with exponential time differencing, and the default
# blueprint shows the order parameter as an image, one frame per time unit.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.pattern_formation import AllenCahnSystem

SYSTEM = 'allen_cahn'

system = AllenCahnSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Coarsening by curvature
# -----------------------
# The walls move at a speed set by their curvature, so the length of wall per unit area falls as `t^(-1/2)` (Allen and
# Cahn 1979, Bray 1994). The wall density, the fraction of neighbouring pairs of sites with opposite signs, should
# follow a line of slope -1/2 on log-log axes once the domains have formed. The mean of the order parameter is not
# conserved: it wanders as domains of one sign win over the other.

phi = trajectory.lattice('mixture.phi')
times = trajectory.times
signs = np.sign(phi)
walls = 0.5 * (
    (signs != np.roll(signs, 1, axis=1)).mean(axis=(1, 2)) + (signs != np.roll(signs, 1, axis=2)).mean(axis=(1, 2))
)
late = times >= 10
slope = np.polyfit(np.log(times[late]), np.log(walls[late]), 1)[0]

fig, (density, mean) = plt.subplots(1, 2, figsize=(10, 4))
density.loglog(times[1:], walls[1:], 'o', markersize=3, label='wall density')
density.loglog(times[late], walls[late][0] * (times[late] / times[late][0]) ** -0.5, 'k--', label='t^(-1/2)')
density.set_xlabel('time')
density.set_ylabel('fraction of neighbouring pairs across a wall')
density.set_title(f'Walls shorten with slope {slope:.2f}')
density.legend()
mean.plot(times, phi.mean(axis=(1, 2)))
mean.set_xlabel('time')
mean.set_ylabel('mean order parameter')
mean.set_title('The order parameter is not conserved')
fig.tight_layout()
plt.show()
