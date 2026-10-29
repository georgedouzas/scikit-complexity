"""
# Fractional Brownian motion

The walkers are driven by noise whose increments anticorrelate over the whole run: a step to one side makes a step back
more likely, so the cloud spreads slower than Brownian motion and every path looks rougher.
"""

# %%
# The model is created by name from the catalog with 500 walkers in the plane and the Hurst index `H = 0.3`, and
# simulated to time 10. The fractional Euler-Maruyama stepper draws the correlated increments of the whole run on its
# grid before the first step. The observer saves every frame with the default blueprint, whose primary view shows the
# walkers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import ScalingAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import FractionalEulerMaruyamaSolver
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create

MODEL = 'fractional_brownian_motion'

system = create(MODEL, n_walkers=500, H=0.3)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = FractionalEulerMaruyamaSolver(dt=0.01)
simulation = (
    Simulation(system, solver=solver, dt=0.05, observers=[observer], random_state=0).check().prepare().run(until=10)
)

# %%
# Scaling and memory
# ------------------
# Fractional Brownian motion has `<x^2> = 2 D t^(2 H)` along every axis, so the ensemble mean squared displacement in
# the plane should follow the dashed line `4 D t^0.6`. Detrended fluctuation analysis of the increments of each walker
# should give Hurst exponents spread around `H = 0.3`, below the 0.5 of independent increments.

trajectory = simulation.trajectory_
times = trajectory.times[1:]
positions = trajectory.values('walkers.x')
squared = np.mean(np.sum(positions[1:] ** 2, axis=-1), axis=1)
exponents = ScalingAnalysis().check().prepare(np.diff(positions[:, :100, 0], axis=0)).run().exponent_
fig, (displacement, memory) = plt.subplots(1, 2, figsize=(10, 4))
displacement.loglog(times, squared, label='simulation')
displacement.loglog(times, 4 * times**0.6, '--', color='black', label='4 D t^(2 H)')
displacement.set_xlabel('time t')
displacement.set_ylabel('ensemble mean squared displacement')
displacement.set_title('Subdiffusion, <x^2> ~ t^(2 H)')
displacement.legend()
memory.hist(exponents, bins=15)
memory.axvline(0.3, color='black', linestyle='--', label='H = 0.3')
memory.axvline(0.5, color='gray', linestyle=':', label='independent increments')
memory.set_xlabel('Hurst exponent by detrended fluctuation analysis')
memory.set_ylabel('number of walkers')
memory.set_title(f'Anticorrelated increments, mean {np.mean(exponents):.2f}')
memory.legend()
fig.tight_layout()
plt.show()
