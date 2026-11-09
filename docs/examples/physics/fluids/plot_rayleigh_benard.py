"""
# Rayleigh-Bénard convection

A layer of fluid heated from below is unstable when buoyancy beats viscosity and heat diffusion, which happens above
the critical Rayleigh number `Ra_c = 27 pi^4 / 4 = 657.5` for free-slip walls. A tiny perturbation of the critical
mode then grows into steady convection rolls: warm fluid rises in one column, cold fluid sinks in the next, and the
temperature field takes the shape of the rolls. The view shows the temperature departure of a layer at `Ra = 2000`,
warm and cold plumes side by side.
"""

# %%
# The system is built from its class at `Ra = 2000` and `Pr = 1` on a 32 x 32 lattice, which holds the
# layer and its mirror image for the free-slip walls. The catalog registers the fourth-order exponential time
# differencing stepper for this spectral system, so it is passed explicitly with steps of 0.005. The observer saves
# every frame of 0.05 with the default blueprint, here with the temperature as the primary view.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.fluids import RayleighBenardSystem

SYSTEM = 'rayleigh_benard'

system = RayleighBenardSystem(Ra=2000.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='flow/theta', interval=1)
solver = ETDRK4Solver(dt=0.005)
trajectory = simulate(system, solver=solver, dt=0.05, observers=[observer])

# %%
# Onset of convection
# -------------------
# Below `Ra_c` every perturbation decays and the layer conducts heat at rest. Above it the critical mode grows until
# the nonlinear terms saturate it, at an amplitude that rises from zero as `sqrt(Ra - Ra_c)` near onset. The largest
# temperature departure at the end of runs on a 16 x 16 lattice should be negligible below `Ra_c` and finite above
# it, growing with the Rayleigh number.

rayleighs = np.array([500.0, 700.0, 900.0, 1300.0, 2000.0])
amplitudes = []
for rayleigh in rayleighs:
    run = simulate(RayleighBenardSystem(Ra=rayleigh, shape=(16, 16)), solver=solver)
    amplitudes.append(np.abs(run.get_values('flow.theta')[-1]).max())
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rayleighs, amplitudes, 'o-', label='largest temperature departure at t = 10')
ax.axvline(27 * np.pi**4 / 4, color='gray', linestyle='--', label='Ra_c = 657.5')
ax.set_xlabel('Rayleigh number Ra')
ax.set_ylabel('max |theta|')
ax.set_title('Convection sets in above the critical Rayleigh number')
ax.legend()
fig.tight_layout()
plt.show()
