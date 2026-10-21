"""
# Lorenz-84

The westerly jet `x` and the travelling eddies `y` and `z` exchange energy without ever settling: the jet strengthens
while the eddies are weak, the eddies then grow on it and weaken it, and the cycle repeats irregularly.
"""

# %%
# The model is created by name from the catalog with Lorenz's parameters `a = 0.25`, `b = 4`, `F = 8`, and `G = 1`, and
# runs for 100 time units. The default blueprint shows the jet and the two eddy phases in one time series view.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import lyapunov_spectrum
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'lorenz_84'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=8)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=100)

# %%
# A chaotic attractor
# -------------------
# The trajectory should fill a strange attractor rather than close on a cycle, and the Benettin estimate of the
# Lyapunov spectrum should have one positive exponent, one near zero along the flow, and a negative sum, since the flow
# contracts volumes.

trajectory = simulation.trajectory_
settled = trajectory.times > 10
exponents = lyapunov_spectrum(system, until=500.0, dt=0.01)
fig, (attractor, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
attractor.plot(trajectory['atmosphere.x'][settled], trajectory['atmosphere.y'][settled], linewidth=0.5)
attractor.set_xlabel('jet x')
attractor.set_ylabel('eddy phase y')
attractor.set_title('Strange attractor')
spectrum.bar(range(1, len(exponents) + 1), exponents)
spectrum.axhline(0.0, color='black', linewidth=0.8)
for index, value in enumerate(exponents, start=1):
    spectrum.annotate(f'{value:.3f}', (index, value), ha='center', va='bottom' if value > 0 else 'top')
spectrum.set_xticks(range(1, len(exponents) + 1))
spectrum.set_xlabel('exponent')
spectrum.set_ylabel('Lyapunov exponent')
spectrum.set_title('One positive exponent: chaos')
fig.tight_layout()
plt.show()
