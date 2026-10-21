"""
# Schnakenberg patterns

Noise of one percent around the uniform steady state grows into a regular pattern of spots of the activator `u`, a
Turing pattern that forms because the inhibitor `v` diffuses forty times faster than `u`.
"""

# %%
# The model is created by name from the catalog on a 64 by 64 grid with `a = 0.1`, `b = 0.9`, `Du = 1`, and `Dv = 40`,
# and integrated with RK4. Naming the `u` field as primary puts its view first, so it becomes the thumbnail.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of, structure_factor
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'schnakenberg'

system = create(MODEL, random_state=0)
blueprint = default_blueprint(system, primary='cells/u')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=2.0, solver=RK4(dt=0.005), observers=[observer]).run(until=200)

# %%
# Pattern wavelength against the linear instability
# -------------------------------------------------
# At the steady state `u = a + b`, `v = b / (a + b)^2` a mode of wavenumber `q` grows at the largest eigenvalue of
# `[[2 u v - 1 - Du q^2, u^2], [-2 u v, -u^2 - Dv q^2]]`, positive only in a band of wavenumbers. The structure factor
# of the final `u` should peak inside that band, near the fastest growing mode.

a, b, Du, Dv = 0.1, 0.9, 1.0, 40.0
u0, v0 = a + b, b / (a + b) ** 2
u = fields_of(simulation.trajectory_, 'cells.u')[-1]
factor = structure_factor(u)
q = np.linspace(0.01, 1.5, 300)
growth = np.array(
    [
        np.linalg.eigvals([[2 * u0 * v0 - 1 - Du * k**2, u0**2], [-2 * u0 * v0, -(u0**2) - Dv * k**2]]).real.max()
        for k in q
    ],
)
fig, (field, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
field.imshow(u, cmap='viridis')
field.set_title('u at the last frame')
field.set_axis_off()
spectrum.plot(factor.wavenumbers, factor.power / factor.power.max(), label='structure factor of u')
spectrum.plot(q, growth / growth.max(), color='black', label='linear growth rate')
spectrum.axhline(0.0, color='gray', linewidth=0.8)
spectrum.axvline(q[np.argmax(growth)], color='black', linestyle='--', linewidth=0.8)
spectrum.set_xlim(0, 1.5)
spectrum.set_ylim(-0.5, 1.1)
spectrum.set_xlabel('wavenumber q')
spectrum.set_ylabel('normalized value')
spectrum.set_title('The pattern grows from the unstable band')
spectrum.legend()
fig.tight_layout()
plt.show()
