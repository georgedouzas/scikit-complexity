"""
# Strongly nonlinear oscillator under periodic impulses

An oscillator with a very stiff spring, `x'' + tan(x) / cos²(x)`, is kicked by impulses of alternating sign. For the
right amplitude every impulse reverses the motion, and the oscillator repeats with exactly the period of the kicks.
"""

# %%
# The defaults start the exact periodic orbit of amplitude `A = 1.2` for the impulse amplitude `H = 2`, whose quarter
# period is `a = q arccos(H q² / sqrt((1 - q²) (1 - q² H²)))` with `q = cos A`. The default blueprint shows the
# displacement and the velocity, which jumps at every impulse.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'impulsive_oscillator'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.005, observers=[observer]).prepare().run(until=10)

# %%
# Temporal dipole shift
# ---------------------
# Shifting the impulses towards each other by `gamma` turns the rectangular wave of the kicks into the derivative of
# an asymmetric saw-tooth. The symmetric orbit should then lose its symmetry, with longer excursions on one side.

fig, ax = plt.subplots(figsize=(6, 4))
for gamma in (0.0, 0.2, 0.4):
    run = Simulation(create_system(SYSTEM, gamma=gamma), dt=0.005).prepare().run(until=5)
    times = np.asarray(run.trajectory_.times)
    ax.plot(times, np.ravel(run.trajectory_.values('oscillator.x')), label=f'gamma = {gamma}')
ax.set_xlabel('time')
ax.set_ylabel('x')
ax.set_title('Impulsive oscillator with a dipole shift')
ax.legend()
plt.show()
