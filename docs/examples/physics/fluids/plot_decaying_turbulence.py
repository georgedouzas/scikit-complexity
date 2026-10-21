"""
# Decaying two-dimensional turbulence

A random vorticity field on a periodic square evolves by the two-dimensional Navier-Stokes equations with a small
viscosity. In two dimensions energy flows to large scales and enstrophy, the mean squared vorticity, to small scales,
so vortices of the same sign merge into ever larger ones while thin filaments of vorticity are stretched and wiped out
by viscosity. The view shows the vorticity of a 128 x 128 lattice at the end of the run, a large vortex wrapped in
thin spiral filaments.
"""

# %%
# The model is created by name from the catalog with viscosity `nu = 0.001` and a random start built from the Fourier
# modes below 6. The catalog registers the fourth-order exponential time differencing stepper for this spectral model,
# so it is passed explicitly with steps of 0.05. The observer saves every frame of 0.2 with the default blueprint,
# whose primary view is the image of the vorticity.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ETDRK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'decaying_turbulence'

system = create(MODEL, shape=(128, 128), random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, solver=ETDRK4(dt=0.05), dt=0.2, observers=[observer]).run(until=40)

# %%
# Energy survives, enstrophy decays
# ---------------------------------
# The energy `E = sum |omega_k|^2 / (2 k^2)` and the enstrophy `Z = sum |omega_k|^2 / 2` of the Fourier modes are both
# conserved without viscosity. Viscosity removes them at the rates `2 nu Z` and `2 nu P`, with the palinstrophy `P`
# of the small scales, and the enstrophy cascade feeds `P`. The enstrophy should therefore fall much faster than the
# energy: by about an order of magnitude over the run, while the energy loses only a fraction.

vorticity = simulation.trajectory_['flow.omega'].reshape(-1, 128, 128)
spectrum = np.abs(np.fft.fft2(vorticity)) ** 2
wavenumbers = np.fft.fftfreq(128, d=1 / 128)
squared = wavenumbers[:, None] ** 2 + wavenumbers[None, :] ** 2
squared[0, 0] = np.inf
energy = (spectrum / squared).sum(axis=(1, 2))
enstrophy = spectrum.sum(axis=(1, 2))
times = simulation.trajectory_.times
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(times, energy / energy[0], label='energy E / E(0)')
ax.plot(times, enstrophy / enstrophy[0], label='enstrophy Z / Z(0)')
ax.set_xlabel('time t')
ax.set_ylabel('relative to the start')
ax.set_title('Enstrophy decays much faster than energy')
ax.legend()
fig.tight_layout()
plt.show()
