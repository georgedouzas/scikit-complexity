"""
# Complex Ginzburg-Landau

The complex Ginzburg-Landau equation `A_t = A + (1 + i b) laplacian(A) - (1 + i c) |A|^2 A` is the generic equation
of a medium near the onset of oscillations. With `1 + b c < 0` the uniform oscillation is unstable, and the field
breaks into defect turbulence: spiral cores where the amplitude vanishes are born and annihilated in pairs, and the
phase winds by a full turn around each of them.
"""

# %%
# A 64 x 64 periodic square of side 64 starts from small noise with `b = 1` and `c = -1.5`, and the run uses the
# fourth-order exponential time differencing solver of the model with a step of 0.1. The default blueprint shows the
# real part of the amplitude, whose bands curl around the defects.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.base import wrap_angle
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ETDRK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'complex_ginzburg_landau'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, solver=ETDRK4(dt=0.1), observers=[observer]).run(until=150)

# %%
# Defects and amplitude
# ---------------------
# A defect is a plaquette around which the phase of `A` winds by a full turn. After the noise grows into the limit
# cycle, the number of defects should settle to fluctuate around a steady value, the signature of defect turbulence,
# while the mean amplitude `|A|` stays a little below 1, held down by the defects and the phase gradients.

trajectory = simulation.trajectory_
real = np.asarray(trajectory.lattice('medium.u'))
imaginary = np.asarray(trajectory.lattice('medium.v'))
phase = np.angle(real + 1j * imaginary)
right, down = np.roll(phase, -1, axis=-1), np.roll(phase, -1, axis=-2)
corner = np.roll(right, -1, axis=-2)
winding = wrap_angle(right - phase) + wrap_angle(corner - right) + wrap_angle(down - corner) + wrap_angle(phase - down)
defects = (np.abs(winding) > np.pi).sum(axis=(1, 2))
amplitude = np.abs(real + 1j * imaginary).mean(axis=(1, 2))
fig, (count, size) = plt.subplots(1, 2, figsize=(11, 4))
count.plot(trajectory.times, defects)
count.set_xlabel('time')
count.set_ylabel('defects')
count.set_title('Defects are born and annihilated in pairs')
size.plot(trajectory.times, amplitude)
size.set_xlabel('time')
size.set_ylabel('mean |A|')
size.set_title('The amplitude saturates below 1')
fig.tight_layout()
plt.show()
