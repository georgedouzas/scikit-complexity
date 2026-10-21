"""
# Kerner-Konhäuser clusters

At an intermediate density, uniform traffic is unstable: a small bump of density grows into a dense cluster, a jam
that keeps its shape and moves around the ring road while the traffic outside it flows freely.
"""

# %%
# The model is created by name from the catalog on a ring of length 10 in 40 cells at density 0.25 with a bump of 0.1,
# and is integrated with SSPRK for 100 time units. The default blueprint puts the history of the density first, where
# the cluster is the bright band.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import SSPRK
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'kerner_konhauser'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.2, solver=SSPRK(cfl=0.15), observers=[observer]).run(until=100)

# %%
# Growth of the cluster
# ---------------------
# The bump should grow until the density in the cluster saturates far above the mean 0.25, while the density outside
# falls below it, and the cluster should then keep its shape and travel around the ring.

trajectory = simulation.trajectory_
rho = fields_of(trajectory, 'road.rho')
x = (np.arange(rho.shape[1]) + 0.5) * 10.0 / rho.shape[1]
fig, (growth, profiles) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(trajectory.times, rho.max(axis=1), label='largest density')
growth.plot(trajectory.times, rho.min(axis=1), label='smallest density')
growth.axhline(0.25, color='black', linestyle='--', linewidth=0.8, label='mean density')
growth.set_xlabel('time')
growth.set_ylabel('density')
growth.set_title('The bump grows into a cluster')
growth.legend()
for time in (0, 50, 75, 100):
    index = np.argmin(np.abs(trajectory.times - time))
    profiles.plot(x, rho[index], label=f't = {time}')
profiles.set_xlabel('position on the ring')
profiles.set_ylabel('density')
profiles.set_title('A cluster of fixed shape travels')
profiles.legend()
fig.tight_layout()
plt.show()
