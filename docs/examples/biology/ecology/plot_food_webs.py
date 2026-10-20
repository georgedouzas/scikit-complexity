"""
# Food webs

Frogs and mice grow on their own, snakes eat frogs and mice, and hawks eat snakes and mice. After damped oscillations
the four species settle together at a stable equilibrium, every predator held in check by its prey and every prey by
its predators.
"""

# %%
# The model is created by name from the catalog with its default web. The observer saves every frame with the default
# blueprint, whose primary view is the time series of the four populations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import stability
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'food_webs'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=100)

# %%
# Stability of the equilibrium
# ----------------------------
# The equilibrium is stable when every eigenvalue of the Jacobian there has a negative real part. Evaluating the
# Jacobian at the final state, all four eigenvalues should lie in the left half-plane, and their imaginary parts give
# the damped oscillations of the approach.

trajectory = simulation.trajectory_
species = ['frogs', 'mice', 'snakes', 'hawks']
final = {f'{name}.n': trajectory[f'{name}.n'][-1] for name in species}
result = stability(system, final)
eigenvalues = np.asarray(result.eigenvalues, dtype=complex)
fig, (series, spectrum) = plt.subplots(1, 2, figsize=(11, 4))
for name in species:
    series.plot(trajectory.times, trajectory[f'{name}.n'], label=name)
series.set_xlabel('time')
series.set_ylabel('population')
series.set_title('Four species settle together')
series.legend()
spectrum.plot(eigenvalues.real, eigenvalues.imag, 'o', markersize=8)
spectrum.axvline(0, color='black', linestyle='--')
spectrum.set_xlabel('real part')
spectrum.set_ylabel('imaginary part')
spectrum.set_title(f'Largest real part {eigenvalues.real.max():.3f}: {result.classification}')
fig.tight_layout()
plt.show()
