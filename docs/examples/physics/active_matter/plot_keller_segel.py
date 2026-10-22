"""
# Keller-Segel chemotaxis

Cells that secrete a chemical and climb its gradient attract each other. When the chemotactic pull beats the diffusion
of the cells, a uniform colony is unstable: small excesses of cells make more chemical, attract more cells, and grow
into dense aggregates that sharpen towards a collapse. The image shows the cell density near the end of the run: a
dark uniform background with a few bright, compact aggregates.
"""

# %%
# The model is created by name from the catalog on a periodic square of side 32 with 32 sites per side, with the
# chemotactic sensitivity `chi = 5`, five times the threshold, and the cells at density 1 with 1% noise. In two
# dimensions the aggregates collapse in finite time, so the run stops at `t = 10.5`, shortly before the density blows
# up. The default blueprint shows the cell density as an image.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'keller_segel'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.25, observers=[observer], random_state=0).run(until=10.5)

# %%
# Linear instability and conservation
# ------------------------------------
# Around the uniform state, a mode grows at the largest eigenvalue of `[[D L, chi rho_0 G], [1, Dc L - k]]`, the
# linearised laws, where `L` is the eigenvalue of the Laplacian, `-q^2` in the continuum, and `G` that of the
# chemotactic term, `q^2` in the continuum. On the lattice of unit spacing, the finite differences give
# `L = -sum(2 - 2 cos q_i)` and, for the central gradient inside the central divergence, `G = sum(sin^2 q_i)`. The
# fluctuations of the density should rise towards the largest of these rates over the modes of the lattice, and then
# grow faster than exponentially as the aggregates collapse. The total number of cells should stay constant.

rho = fields_of(simulation.trajectory_, 'colony.rho')
times = simulation.trajectory_.times
axis = 2 * np.pi * np.fft.fftfreq(32)
qx, qy = np.meshgrid(axis, axis)
laplacian = -(2 - 2 * np.cos(qx)) - (2 - 2 * np.cos(qy))
chemotaxis = np.sin(qx) ** 2 + np.sin(qy) ** 2
fastest = max(
    np.linalg.eigvals([[lap, 5.0 * drive], [1.0, lap - 1.0]]).real.max()
    for lap, drive in zip(laplacian.ravel(), chemotaxis.ravel(), strict=True)
)
spread = rho.std(axis=(1, 2))

fig, (growth, mass) = plt.subplots(1, 2, figsize=(10, 4))
growth.semilogy(times, spread, label='standard deviation of the density')
growth.semilogy(times, spread[4] * np.exp(fastest * (times - times[4])), 'k--', label=f'linear rate {fastest:.2f}')
growth.set_xlabel('time')
growth.set_ylabel('density fluctuations')
growth.set_title('Linear growth, then collapse')
growth.legend()
mass.plot(times, rho.sum(axis=(1, 2)) / rho[0].sum() - 1)
mass.set_xlabel('time')
mass.set_ylabel('relative change of the number of cells')
mass.set_title('The number of cells is conserved')
fig.tight_layout()
plt.show()
