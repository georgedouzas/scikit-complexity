"""
# Brusselator patterns

Small noise around the uniform steady state grows into a pattern of spots of the activator `u`, because the inhibitor
`v` diffuses eight times faster: a Turing instability with a wavelength set by the reaction and the diffusion.
"""

# %%
# The model is created by name from the catalog on a 64 by 64 grid with `a = 1`, `b = 3`, `Du = 1`, and `Dv = 8`, and
# integrated with RK4. Naming the `u` field as primary puts its view first, so it becomes the thumbnail.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of, structure_factor
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'brusselator'

system = create(MODEL, random_state=0)
blueprint = default_blueprint(system, primary='cells/u')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, dt=1.0, solver=RK4(dt=0.01), observers=[observer]).run(until=100)

# %%
# Pattern wavelength against the linear instability
# -------------------------------------------------
# Linearized around `(a, b / a)`, a mode of wavenumber `q` grows at the largest eigenvalue of
# `[[b - 1 - Du q^2, a^2], [-b, -a^2 - Dv q^2]]`. With `b = 3` the uniform mode oscillates too, but the Turing band
# grows faster, so the structure factor of the final `u` should peak in that band, near the fastest growing mode, with
# the nonlinear spots settling at a slightly shorter wavelength.

a, b, Du, Dv = 1.0, 3.0, 1.0, 8.0
u = fields_of(simulation.trajectory_, 'cells.u')[-1]
factor = structure_factor(u)
q = np.linspace(0.01, 2.0, 300)
growth = [np.linalg.eigvals([[b - 1 - Du * k**2, a**2], [-b, -(a**2) - Dv * k**2]]).real.max() for k in q]
fig, (field, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
field.imshow(u, cmap='viridis')
field.set_title('u at the last frame')
field.set_axis_off()
spectrum.plot(factor.wavenumbers, factor.power / factor.power.max(), label='structure factor of u')
spectrum.plot(q, np.array(growth) / max(growth), color='black', label='linear growth rate')
spectrum.axvline(q[np.argmax(growth)], color='black', linestyle='--', linewidth=0.8)
spectrum.set_xlim(0, 2)
spectrum.set_ylim(-0.5, 1.1)
spectrum.set_xlabel('wavenumber q')
spectrum.set_ylabel('normalized value')
spectrum.set_title('The pattern selects the fastest growing mode')
spectrum.legend()
fig.tight_layout()
plt.show()
