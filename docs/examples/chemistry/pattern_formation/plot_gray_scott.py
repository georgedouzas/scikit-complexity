"""
# Gray-Scott

A square seed at the centre breaks into spots of the activator `v`, which divide and push outwards until they fill the
domain with a pattern of separate spots, while the substrate `u` is used up where the spots sit.
"""

# %%
# The default blueprint has one 2D view per field state. Naming the `v` field as primary puts its view first, so it
# becomes the thumbnail, with the `u` field beside it.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'gray_scott'

system = create(MODEL, random_state=0)
blueprint = default_blueprint(system, primary='cells/v')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=20.0, solver=RK4(dt=1.0), observers=[observer]).run(until=10000)

# %%
# Final pattern and its radial power spectrum
# -------------------------------------------
# A pattern of spots at a typical spacing gives a ring in the two-dimensional power spectrum of `v`, so its radial
# average should show a peak at the wavenumber of the spot spacing, beyond the low wavenumbers of the round outline
# of the patterned region.

v = simulation.trajectory_.lattice('cells.v')[-1]
power = np.abs(np.fft.fftshift(np.fft.fft2(v - v.mean()))) ** 2
size = v.shape[0]
radius = np.hypot(*np.indices(v.shape) - size // 2).astype(int)
radial = np.bincount(radius.ravel(), power.ravel()) / np.bincount(radius.ravel())
wavenumbers = np.arange(len(radial))[1 : size // 2]
patterned = wavenumbers >= 4
peak = wavenumbers[patterned][radial[1 : size // 2][patterned].argmax()]
fig, (field, spectrum) = plt.subplots(1, 2, figsize=(10, 4))
field.imshow(v, cmap='viridis')
field.set_title('v at the last frame')
field.set_axis_off()
spectrum.semilogy(wavenumbers, radial[1 : size // 2])
spectrum.axvline(peak, color='black', linestyle='--', label=f'peak: spacing {size / peak:.1f} cells')
spectrum.set_xlabel('wavenumber (cycles per domain)')
spectrum.set_ylabel('radial power')
spectrum.set_title('Radial power spectrum of v')
spectrum.legend()
fig.tight_layout()
plt.show()
