"""
# Fractional Brownian motion

The walkers are driven by noise whose increments anticorrelate over the whole run: a step to one side makes a step back
more likely, so the cloud spreads slower than Brownian motion and every path looks rougher.
"""

# %%
# The example builds `FractionalBrownianMotionSystem` with 500 walkers in the plane and the Hurst index `H = 0.3`, and
# simulates it to time 10. The fractional Euler-Maruyama stepper draws the correlated increments of the whole run on its
# grid before the first step. The observer saves every frame with the default blueprint, whose primary view shows the
# walkers.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import scaling
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import FractionalEulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import FractionalBrownianMotionSystem

SYSTEM = 'fractional_brownian_motion'

system = FractionalBrownianMotionSystem(n_walkers=500, H=0.3)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = FractionalEulerMaruyamaSolver(dt=0.01)
trajectory = simulate(system, solver=solver, observers=[observer], random_state=0)

# %%
# Scaling and memory
# ------------------
# Fractional Brownian motion has `<x^2> = 2 D t^(2 H)` along every axis, so the ensemble mean squared displacement in
# the plane should follow the dashed line `4 D t^0.6`. Detrended fluctuation analysis of the increments of each walker
# should give Hurst exponents spread around `H = 0.3`, below the 0.5 of independent increments.

times = trajectory.times[1:]
positions = trajectory.get_values('walkers.x')
squared = np.mean(np.sum(positions[1:] ** 2, axis=-1), axis=1)
exponents = scaling(np.diff(positions[:, :100, 0], axis=0)).exponent
fig, (displacement, memory) = plt.subplots(1, 2, figsize=(10, 4))
displacement.loglog(times, squared, label='trajectory')
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
