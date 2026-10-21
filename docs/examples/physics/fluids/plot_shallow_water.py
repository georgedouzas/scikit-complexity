"""
# Shallow-water dam break

Still water stands twice as deep on the left of a dam as on the right. When the dam is removed, a rarefaction wave
runs upstream into the deep water, lowering it smoothly, while a bore, a moving step in the water surface, runs
downstream into the shallow water, with a plateau of intermediate depth between them. The space-time image shows the
depth along the channel, one row per saved step with the earliest at the top, with the fan of the rarefaction opening
to the left and the bore moving to the right.
"""

# %%
# The model is created by name from the catalog with 400 finite-volume cells in a channel of length 10, depths 2 and 1,
# and `g = 9.81`. The catalog registers the strong-stability-preserving Runge-Kutta stepper for this model, which keeps
# the bore free of oscillations, so it is passed explicitly. The observer saves every frame of 0.005 with the default
# blueprint, whose primary view is the space-time image of the depth.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'shallow_water'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, solver=SSPRK(), dt=0.005, observers=[observer]).run(until=0.6)

# %%
# Stoker's exact solution
# -----------------------
# The plateau depth `h_m` makes the velocity behind the rarefaction, `2 (sqrt(g h_u) - sqrt(g h_m))`, equal to the
# velocity behind the bore, `(h_m - h_d) sqrt(g (h_m + h_d) / (2 h_m h_d))`. Inside the fan the depth is
# `(2 sqrt(g h_u) - x / t)^2 / (9 g)`, and the bore moves at `h_m u_m / (h_m - h_d)`. The simulated depth at the end
# of the run should follow this solution, with the bore smeared over a few cells.

g, upstream, downstream, length = 9.81, 2.0, 1.0, 10.0


def bore_mismatch(middle):
    """Return the difference of the velocities behind the rarefaction and behind the bore for a plateau depth."""
    behind = 2 * (np.sqrt(g * upstream) - np.sqrt(g * middle))
    return behind - (middle - downstream) * np.sqrt(g * (middle + downstream) / (2 * middle * downstream))


middle = brentq(bore_mismatch, downstream, upstream)
velocity = 2 * (np.sqrt(g * upstream) - np.sqrt(g * middle))
bore = middle * velocity / (middle - downstream)
time = simulation.trajectory_.times[-1]
depth = simulation.trajectory_['channel.h'][-1]
x = (np.arange(depth.size) + 0.5) * length / depth.size - length / 2
speed = x / time
exact = np.where(speed < -np.sqrt(g * upstream), upstream, (2 * np.sqrt(g * upstream) - speed) ** 2 / (9 * g))
exact = np.where(speed > velocity - np.sqrt(g * middle), middle, exact)
exact = np.where(speed > bore, downstream, exact)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, depth, label=f'simulation, t = {time:.2f}')
ax.plot(x, exact, '--', color='black', label=f'Stoker, plateau h_m = {middle:.3f}')
ax.set_xlabel('distance from the dam x')
ax.set_ylabel('depth h')
ax.set_title('Rarefaction, plateau, and bore of the dam break')
ax.legend()
fig.tight_layout()
plt.show()
