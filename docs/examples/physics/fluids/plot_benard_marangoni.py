"""
# Bénard-Marangoni convection

A shallow liquid layer heated from below with an open surface breaks into a mosaic of hexagonal cells. The surface
tension falls with temperature, so the hot fluid that rises in the middle of a cell is pulled outwards along the
surface, which favours hexagons over the rolls of a layer with a fixed lid. The image shows the vertical velocity of
the layer at the end of the run: bright cells of rising fluid packed in a hexagonal mosaic, ringed by sinking fluid.
"""

# %%
# The system is built from its class on a periodic square of side `16π`, eight cell wavelengths, at the
# distance `r = 0.1` above onset and with the quadratic term `g = 1` of the thermocapillary surface. The field starts
# from noise of amplitude 0.01. The catalog defaults integrate it with exponential time differencing, and the default
# blueprint shows the field as an image.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

from skcomplex.core.analysis import domains, spectrum
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.fluids import BenardMarangoniSystem

SYSTEM = 'benard_marangoni'
LENGTH = 16 * np.pi

system = BenardMarangoniSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Hexagons against rolls
# ----------------------
# Both patterns select the wavenumber `k = 1` of the fastest growing mode, so the structure factors should peak at the
# same place. They differ in symmetry: hexagons hold three wave vectors 60 degrees apart, so their orientation order
# `|<exp(2 i theta)>|` should be near 0, while rolls hold one direction per patch and an order well above it. The
# quadratic term also makes the rising cells stronger than the sinking rims, so the values of the hexagons should be
# skewed, while the rolls of the same layer with `g = 0` are symmetric.

hexagons = trajectory.get_lattice('layer.u')[-1]
rolls_run = simulate(BenardMarangoniSystem(g=0.0), random_state=0)
rolls = rolls_run.get_lattice('layer.u')[-1]
spacing = LENGTH / hexagons.shape[0]

fig, (spectrum_axes, histogram) = plt.subplots(1, 2, figsize=(10, 4))
for field, label in [(hexagons, 'hexagons, g = 1'), (rolls, 'rolls, g = 0')]:
    factor = spectrum(field, kind='structure-factor', spacing=spacing)
    order = domains(field, kind='wavelength', spacing=spacing).orientation[0]
    spectrum_axes.plot(factor.wavenumbers, factor.power, 'o-', markersize=3, label=f'{label}, orientation {order:.2f}')
    histogram.hist(
        field.ravel(),
        bins=60,
        histtype='step',
        density=True,
        label=f'{label}, skewness {skew(field.ravel()):.2f}',
    )
spectrum_axes.axvline(1.0, linestyle='--', color='black', label='k = 1')
spectrum_axes.set_xlim(0, 3)
spectrum_axes.set_xlabel('wavenumber k')
spectrum_axes.set_ylabel('structure factor S(k)')
spectrum_axes.set_title('Both patterns select k = 1')
spectrum_axes.legend(fontsize=8)
histogram.set_xlabel('vertical velocity u')
histogram.set_ylabel('density')
histogram.set_title('Hexagons rise in strong, skewed cells')
histogram.legend(fontsize=8)
fig.tight_layout()
plt.show()
