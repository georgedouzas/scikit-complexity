"""
# Annealed transient time motion

Every walker diffuses with its own coefficient and draws a new one at random times. Slow walkers keep their small
coefficients for long, so the dark slow walkers stay near the origin while the bright fast ones spread out.
"""

# %%
# The system is built from its class with 500 walkers in the plane, whose diffusion coefficients `D` have
# the density `D^(-1/2) / 2` on `(0, 1)` and last for exponential times of mean `1 / D`, and simulated to time 500.
# Noise laws move the walkers, and the solver stops every walker at its own changes of coefficient. The observer saves
# every frame with the default blueprint, whose primary view shows the walkers colored by their coefficients.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.soft_matter import AnnealedTransientTimeMotionSystem

SYSTEM = 'annealed_transient_time_motion'

system = AnnealedTransientTimeMotionSystem(n_walkers=500, sigma=0.5, gamma=1.0, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = EulerMaruyamaSolver(dt=0.5)
trajectory = simulate(system, solver=solver, observers=[observer], random_state=0)

# %%
# Subdiffusion from heterogeneous diffusion
# -----------------------------------------
# Massignan et al. (2014) show that the ensemble mean squared displacement grows as `t^(sigma / gamma)` for
# `sigma < gamma`, here `t^0.5`, although every walker diffuses normally between changes. The curve should approach the
# dashed line of slope 0.5 at long times. The coefficients the walkers hold at a time are biased towards small values,
# since small coefficients last longer.

times = trajectory.times[1:]
squared = np.mean(np.sum(trajectory.get_values('walkers.x')[1:] ** 2, axis=-1), axis=1)
late = times > 50
slope = np.polyfit(np.log(times[late]), np.log(squared[late]), 1)[0]
fig, (displacement, coefficients) = plt.subplots(1, 2, figsize=(10, 4))
displacement.loglog(times, squared, label=f'trajectory, slope {slope:.2f} after t = 50')
displacement.loglog(times, squared[-1] * (times / times[-1]) ** 0.5, '--', color='black', label='slope 0.5')
displacement.set_xlabel('time t')
displacement.set_ylabel('ensemble mean squared displacement')
displacement.set_title('Subdiffusion, <x^2> ~ t^(sigma / gamma)')
displacement.legend()
coefficients.hist(trajectory.get_values('walkers.D')[0], bins=20, alpha=0.6, label='at t = 0, drawn from B(1/2, 1)')
coefficients.hist(trajectory.get_values('walkers.D')[-1], bins=20, alpha=0.6, label='at t = 500')
coefficients.set_xlabel('diffusion coefficient D')
coefficients.set_ylabel('number of walkers')
coefficients.set_title('Small coefficients last longer')
coefficients.legend()
fig.tight_layout()
plt.show()
