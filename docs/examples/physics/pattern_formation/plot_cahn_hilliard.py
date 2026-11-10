"""
# Cahn-Hilliard

A binary mixture quenched inside its spinodal separates into domains rich in one component or the other. The
composition can only diffuse, so the large domains grow by taking material from the small ones through the bulk, which
is slower than the motion of walls in the Allen-Cahn equation. The image shows the order parameter at the end of the
run: an interpenetrating labyrinth of yellow and purple domains of equal area.
"""

# %%
# The example builds `CahnHilliardSystem` on a periodic square of 64 sites with unit spacing, from noise of
# amplitude 0.05 around the critical composition 0. The catalog defaults integrate it with exponential time
# differencing, and the default blueprint shows the order parameter as an image, one frame every 10 time units.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import spectrum
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.pattern_formation import CahnHilliardSystem

SYSTEM = 'cahn_hilliard'

system = CahnHilliardSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Coarsening by diffusion
# -----------------------
# The domain size is `2 pi / <k>`, from the mean wavenumber of the structure factor. Coarsening by diffusion through
# the bulk, the Lifshitz-Slyozov law, makes it grow as `t^(1/3)`, so it should follow a line of slope 1/3 on log-log
# axes. The mean of the order parameter is conserved by the dynamics and should stay at its initial value.

phi = trajectory.get_lattice('mixture.phi')
times = trajectory.times
sizes = []
for field in phi[1:]:
    factor = spectrum(field - field.mean(), kind='structure-factor')
    sizes.append(2 * np.pi * factor.power.sum() / (factor.wavenumbers * factor.power).sum())
late = times[1:] >= 100
slope = np.polyfit(np.log(times[1:][late]), np.log(np.array(sizes)[late]), 1)[0]

fig, (growth, mean) = plt.subplots(1, 2, figsize=(10, 4))
growth.loglog(times[1:], sizes, 'o', markersize=3, label='domain size')
growth.loglog(times[1:][late], sizes[-1] * (times[1:][late] / times[-1]) ** (1 / 3), 'k--', label='t^(1/3)')
growth.set_xlabel('time')
growth.set_ylabel('domain size 2 pi / <k>')
growth.set_title(f'Domains grow with slope {slope:.2f}')
growth.legend()
mean.plot(times, phi.mean(axis=(1, 2)) - phi[0].mean())
mean.set_xlabel('time')
mean.set_ylabel('change of the mean order parameter')
mean.set_title('The order parameter is conserved')
fig.tight_layout()
plt.show()
