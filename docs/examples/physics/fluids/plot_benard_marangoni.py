"""
# Bénard-Marangoni convection

A shallow liquid layer heated from below with an open surface breaks into a mosaic of hexagonal cells. The surface
tension falls with temperature, so the hot fluid that rises in the middle of a cell is pulled outwards along the
surface, which favours hexagons over the rolls of a layer with a fixed lid. The image shows the vertical velocity of
the layer at the end of the run: bright cells of rising fluid packed in a hexagonal mosaic, ringed by sinking fluid.
"""

# %%
# The model is created by name from the catalog on a periodic square of side `16π`, eight cell wavelengths, at the
# distance `r = 0.1` above onset and with the quadratic term `g = 1` of the thermocapillary surface. The field starts
# from noise of amplitude 0.01. The catalog defaults integrate it with exponential time differencing, and the default
# blueprint shows the field as an image.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

from skcomplex.core.analysis import fields_of, pattern_wavelength, structure_factor
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ETDRK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'benard_marangoni'
LENGTH = 16 * np.pi

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=5.0, solver=ETDRK4(dt=0.5), observers=[observer], random_state=0).run(until=300)

# %%
# Hexagons against rolls
# ----------------------
# Both patterns select the wavenumber `k = 1` of the fastest growing mode, so the structure factors should peak at the
# same place. They differ in symmetry: hexagons hold three wave vectors 60 degrees apart, so their orientation order
# `|<exp(2 i theta)>|` should be near 0, while rolls hold one direction per patch and an order well above it. The
# quadratic term also makes the rising cells stronger than the sinking rims, so the values of the hexagons should be
# skewed, while the rolls of the same layer with `g = 0` are symmetric.

hexagons = fields_of(simulation.trajectory_, 'layer.u')[-1]
rolls_run = Simulation(create(MODEL, g=0.0), dt=5.0, solver=ETDRK4(dt=0.5), random_state=0).run(until=300)
rolls = fields_of(rolls_run.trajectory_, 'layer.u')[-1]
spacing = LENGTH / hexagons.shape[0]

fig, (spectrum, histogram) = plt.subplots(1, 2, figsize=(10, 4))
for field, label in [(hexagons, 'hexagons, g = 1'), (rolls, 'rolls, g = 0')]:
    factor = structure_factor(field, spacing=spacing)
    order = pattern_wavelength(field, spacing=spacing).orientation
    spectrum.plot(factor.wavenumbers, factor.power, 'o-', markersize=3, label=f'{label}, orientation {order:.2f}')
    histogram.hist(
        field.ravel(),
        bins=60,
        histtype='step',
        density=True,
        label=f'{label}, skewness {skew(field.ravel()):.2f}',
    )
spectrum.axvline(1.0, linestyle='--', color='black', label='k = 1')
spectrum.set_xlim(0, 3)
spectrum.set_xlabel('wavenumber k')
spectrum.set_ylabel('structure factor S(k)')
spectrum.set_title('Both patterns select k = 1')
spectrum.legend(fontsize=8)
histogram.set_xlabel('vertical velocity u')
histogram.set_ylabel('density')
histogram.set_title('Hexagons rise in strong, skewed cells')
histogram.legend(fontsize=8)
fig.tight_layout()
plt.show()
