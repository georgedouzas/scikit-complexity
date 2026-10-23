"""
# Active Model B+

Active particles that slow down where they crowd separate into a dense liquid and a dilute vapour, like a passive
mixture. Active Model B+ adds the two lowest-order currents that no free energy gives. With them, the Ostwald ripening
of vapour bubbles in the liquid reverses: small bubbles take vapour from large ones, so the bubbles stop coarsening
at a finite size. The image shows the density at the end of the run: a yellow liquid holding many dark purple bubbles
of about the same size, roughened by the conserved noise.
"""

# %%
# The model is created by name from the catalog on a periodic square of 64 sites with unit spacing, at the mean
# density 0.4, with the active terms `lam = 1` and `zeta = 4` and the noise temperature `D = 0.05`. The catalog
# defaults integrate it with Euler-Maruyama on the Numba backend, and the default blueprint shows the density as an
# image, one frame every 30 time units.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of, structure_factor
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'active_model_b'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    dt=30.0,
    solver=EulerMaruyama(dt=0.02),
    backend='numba',
    observers=[observer],
    random_state=0,
).run(until=1500)

# %%
# Arrested coarsening
# -------------------
# The domain size is `2 pi / <k>`, from the mean wavenumber of the structure factor. In passive Model B it grows
# without bound as `t^(1/3)`, the Lifshitz-Slyozov law. With reverse Ostwald ripening it should rise while the bubbles
# form and then level off well below the `t^(1/3)` curve through its early values. The noise is the divergence of a
# random flux, so the mean density should stay at its initial value to rounding.

phi = fields_of(simulation.trajectory_, 'fluid.phi')
times = simulation.trajectory_.times
sizes = []
for field in phi[1:]:
    factor = structure_factor(field - field.mean())
    sizes.append(2 * np.pi * factor.power.sum() / (factor.wavenumbers * factor.power).sum())
sizes = np.array(sizes)
early = np.searchsorted(times[1:], 300.0)

fig, (growth, mean) = plt.subplots(1, 2, figsize=(10, 4))
growth.loglog(times[1:], sizes, 'o', markersize=3, label='domain size')
growth.loglog(times[1:], sizes[early] * (times[1:] / times[1:][early]) ** (1 / 3), 'k--', label='t^(1/3)')
growth.set_xlabel('time')
growth.set_ylabel('domain size 2 pi / <k>')
growth.set_title(f'Bubbles stop growing at size {sizes[-1]:.0f}')
growth.legend()
mean.plot(times, phi.mean(axis=(1, 2)) - phi[0].mean())
mean.set_xlabel('time')
mean.set_ylabel('change of the mean density')
mean.set_title('The conserved noise keeps the mean')
fig.tight_layout()
plt.show()
