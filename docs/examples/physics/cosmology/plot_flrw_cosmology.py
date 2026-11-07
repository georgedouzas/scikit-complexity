"""
# FLRW cosmology

The Einstein equations of a homogeneous and isotropic metric give the Friedmann equations of the scale factor, so the
expansion of the universe follows from the metric and what fills it.
"""

# %%
# The space derives the Einstein tensor of the Robertson-Walker metric with the unknown scale factor `a(t)`, and the
# interaction keeps its radial component, the acceleration equation of a perfect fluid of matter, radiation, and a
# cosmological constant. The run starts at `a = 0.3` with today's mixture `Ω_m = 0.3` and `Ω_Λ = 0.7`. The default
# blueprint shows `a` and its rate.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.cosmology import FLRWCosmologySystem

SYSTEM = 'flrw_cosmology'

system = FLRWCosmologySystem(a=0.3, omega_matter=0.3, omega_lambda=0.7)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
solver = ScipyODESolver(method='DOP853', rtol=1e-10, atol=1e-12)
trajectory = simulate(system, solver=solver, observers=[observer])

# %%
# Matter, radiation, and vacuum
# -----------------------------
# A universe of one component has an exact scale factor: `a ∝ t^(2/3)` for matter, `a ∝ t^(1/2)` for radiation, and
# `a ∝ e^(H0 t)` for a cosmological constant. Starting each at `a = 0.5`, the simulated curves should lie on them.

cases = {
    'matter': ({'omega_matter': 1.0, 'omega_lambda': 0.0}, lambda t: (1.5 * t + 0.5**1.5) ** (2 / 3)),
    'radiation': ({'omega_matter': 0.0, 'omega_radiation': 1.0, 'omega_lambda': 0.0}, lambda t: np.sqrt(2 * t + 0.25)),
    'vacuum': ({'omega_matter': 0.0, 'omega_lambda': 1.0}, lambda t: 0.5 * np.exp(t)),
}
fig, (mixture, single) = plt.subplots(1, 2, figsize=(10, 4))
mixture.plot(trajectory.times, np.ravel(trajectory.values('universe.a')))
mixture.set_xlabel('time')
mixture.set_ylabel('scale factor a')
mixture.set_title('Matter, then dark energy')
for name, (densities, exact) in cases.items():
    run = simulate(FLRWCosmologySystem(a=0.5, **densities), until=1.5, dt=0.05, solver=solver)
    times = np.asarray(run.times)
    single.plot(times, np.ravel(run.values('universe.a')), label=name)
    single.plot(times, exact(times), '--', color='black')
single.set_xlabel('time')
single.set_title('Exact solutions (dashed)')
single.legend()
fig.tight_layout()
plt.show()
