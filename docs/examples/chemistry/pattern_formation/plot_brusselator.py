"""
# Brusselator patterns

Small noise around the uniform steady state grows into a pattern of spots of the activator `u`, because the inhibitor
`v` diffuses eight times faster: a Turing instability with a wavelength set by the reaction and the diffusion.
"""

# %%
# The example builds `BrusselatorSystem` on a 64 by 64 grid with `a = 1`, `b = 3`, `Du = 1`, and `Dv = 8`, and
# integrates it with RK4. Naming the `u` field as primary puts its view first, so it becomes the thumbnail.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import spectrum
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.chemistry.pattern_formation import BrusselatorSystem

SYSTEM = 'brusselator'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='cells/u')
trajectory = simulate(BrusselatorSystem(random_state=0), observers=[observer])

# %%
# Pattern wavelength against the linear instability
# -------------------------------------------------
# Linearized around `(a, b / a)`, a mode of wavenumber `q` grows at the largest eigenvalue of
# `[[b - 1 - Du q^2, a^2], [-b, -a^2 - Dv q^2]]`. With `b = 3` the uniform mode oscillates too, but the Turing band
# grows faster, so the structure factor of the final `u` should peak in that band, near the fastest growing mode, with
# the nonlinear spots settling at a slightly shorter wavelength.

a, b, Du, Dv = 1.0, 3.0, 1.0, 8.0
u = trajectory.get_lattice('cells.u')[-1]
factor = spectrum(u, kind='structure-factor')
q = np.linspace(0.01, 2.0, 300)
growth = [np.linalg.eigvals([[b - 1 - Du * k**2, a**2], [-b, -(a**2) - Dv * k**2]]).real.max() for k in q]
fig, (field, spectrum_axes) = plt.subplots(1, 2, figsize=(10, 4))
field.imshow(u, cmap='viridis')
field.set_title('u at the last frame')
field.set_axis_off()
spectrum_axes.plot(factor.wavenumbers, factor.power / factor.power.max(), label='structure factor of u')
spectrum_axes.plot(q, np.array(growth) / max(growth), color='black', label='linear growth rate')
spectrum_axes.axvline(q[np.argmax(growth)], color='black', linestyle='--', linewidth=0.8)
spectrum_axes.set_xlim(0, 2)
spectrum_axes.set_ylim(-0.5, 1.1)
spectrum_axes.set_xlabel('wavenumber q')
spectrum_axes.set_ylabel('normalized value')
spectrum_axes.set_title('The pattern selects the fastest growing mode')
spectrum_axes.legend()
fig.tight_layout()
plt.show()
