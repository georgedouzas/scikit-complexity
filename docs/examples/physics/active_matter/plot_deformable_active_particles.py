"""
# Deformable active particles of Ohta and Ohkuma

Self-propelled particles that deform as they move: their motion stretches them, and the stretched shape turns their
velocity. A shape that relaxes fast lets them move straight, while a slowly relaxing shape bends their paths into
circles, a bifurcation of their motion.
"""

# %%
# Twenty particles with the slow shape relaxation `kappa = 0.5` circle at constant speed. The default blueprint shows
# their positions, velocities, and deformations.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'deformable_active_particles'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run(until=100)

# %%
# Straight and circular motion
# ----------------------------
# The circular solution, dashed, exists below `kappa ≈ 0.78` with the angular velocity
# `sqrt(a b (gamma - kappa / 2) - kappa²) / 2`. Particles started slowly should reach it for small `kappa`, while close
# to the threshold straight motion stays stable too and they keep moving straight.

kappas = np.linspace(0.1, 1.5, 15)
turning = []
for kappa in kappas:
    trajectory = create_simulation(SYSTEM, n_particles=2, kappa=kappa).prepare().run(until=100).trajectory_
    velocity = np.asarray(trajectory.values('cells.v'))
    angle = np.unwrap(np.arctan2(velocity[..., 1], velocity[..., 0]), axis=0)
    turning.append(np.abs(angle[-1] - angle[-41]).mean() / 10.0)
theory = np.sqrt(np.clip(1.0 - kappas / 2 - kappas**2, 0, None)) / 2
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(kappas, turning, 'o', label='simulation')
ax.plot(kappas, theory, '--', color='black', label='circular solution')
ax.set_xlabel('relaxation rate kappa')
ax.set_ylabel('angular velocity')
ax.legend()
plt.show()
