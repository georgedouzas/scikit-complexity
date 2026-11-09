"""
# Inertial active Brownian particles

Self-propelled particles with mass and moment of inertia, such as small robots, flying insects, or vibrated grains,
cannot follow their orientation at once. Their velocity relaxes towards the swimming direction and lags behind it as
the orientation turns, so the particles move slower along their heading than massless ones.
"""

# %%
# Two hundred particles swim at `v0 = 3` with the mass `m = 0.5` and the moment of inertia `J = 0.1`. The default
# blueprint shows their positions and velocities.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.active_matter import InertialActiveBrownianParticlesSystem

SYSTEM = 'inertial_active_brownian_particles'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = simulate(InertialActiveBrownianParticlesSystem(), until=20, observers=[observer], random_state=0)

# %%
# Velocity along the orientation
# ------------------------------
# The steady projection of the velocity on the orientation should fall with the mass as
# `v0 / m ∫ exp(-t / m) exp(-D_r (t - J (1 - exp(-t / J)))) dt`, dashed, below the swimming speed of a massless
# particle.

masses = [0.05, 0.2, 0.5, 1.0, 2.0]
measured, expected = [], []
for mass in masses:
    trajectory = simulate(InertialActiveBrownianParticlesSystem(mass=mass), until=20, random_state=0)
    velocity = np.asarray(trajectory.get_values('swimmers.x_d1'))
    heading = np.asarray(trajectory.get_values('swimmers.theta'))
    measured.append((velocity[..., 0] * np.cos(heading) + velocity[..., 1] * np.sin(heading))[10:].mean())
    expected.append(
        3.0 * quad(lambda t, m=mass: np.exp(-t / m) / m * np.exp(-(t - 0.1 * (1 - np.exp(-t / 0.1)))), 0, 60)[0],
    )
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(masses, measured, 'o', label='simulation')
ax.plot(masses, expected, '--', color='black', label='theory')
ax.axhline(3.0, color='grey', linestyle=':', label='v0')
ax.set_xscale('log')
ax.set_xlabel('mass m')
ax.set_ylabel('velocity along the orientation')
ax.legend()
plt.show()
