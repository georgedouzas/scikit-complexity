"""
# Swarmalators

Swarmalators are oscillators that move: like phases attract each other, and nearby units couple their phases. Without
phase coupling they settle into a static phase wave, a still ring in which the phase winds once around the centre.
The view shows the 100 units at the end of the run, coloured by phase: the colour changes steadily around the ring,
with one jump where the phase wraps from pi to -pi.
"""

# %%
# The model is created by name from the catalog with 100 units, the attraction of like phases `J = 1`, and no phase
# coupling, `K = 0`. The units start at random in a square with random phases. The RK4 solver of the catalog
# integrates the run, and the default blueprint shows the positions coloured by phase.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'swarmalators'


def order_parameters(positions, phases):
    """Return the order parameters `S+` and `S-` of the spatial angle and the phase, and the phase coherence."""
    centred = positions - positions.mean(axis=0)
    angles = np.arctan2(centred[:, 1], centred[:, 0])
    plus, minus = (abs(np.exp(1j * (angles + sign * phases)).mean()) for sign in (1, -1))
    return plus, minus, abs(np.exp(1j * phases).mean())


system = create(MODEL, J=1.0, K=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=1.0, solver=RK4(dt=0.1), observers=[observer], random_state=0).run(until=100)

# %%
# Order parameters of the static states
# -------------------------------------
# The order parameters `S± = |mean(exp(i (phi ± theta)))|` of the spatial angle `phi` and the phase `theta` measure
# how the phase follows the position (O'Keeffe, Ceron, and Petersen 2017). In the run above, one of them should grow to
# 1, the static phase wave. Across the phase coupling `K`, the phase coherence should reach 1 for `K > 0`, the
# synchronized disk, while `S±` and the coherence should all stay small for negative `K`, the asynchronous disk.

positions = np.asarray(simulation.trajectory_['units.x'])
phases = np.asarray(simulation.trajectory_['units.theta'])
history = np.array([order_parameters(position, phase) for position, phase in zip(positions, phases, strict=True)])
couplings = [-1.0, -0.5, -0.1, 0.0, 0.5, 1.0]
final = []
for K in couplings:
    run = Simulation(create(MODEL, n_units=60, J=1.0, K=K), dt=1.0, solver=RK4(dt=0.1)).run(until=60)
    final.append(
        order_parameters(np.asarray(run.trajectory_['units.x'])[-1], np.asarray(run.trajectory_['units.theta'])[-1]),
    )
final = np.array(final)

fig, (growth, states) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(simulation.trajectory_.times, history[:, 0], label='S+')
growth.plot(simulation.trajectory_.times, history[:, 1], label='S-')
growth.set_xlabel('time')
growth.set_ylabel('order parameter')
growth.set_title('A static phase wave forms at K = 0')
growth.legend()
states.plot(couplings, final.max(axis=1, where=[True, True, False], initial=0.0), 'o-', label='max(S+, S-)')
states.plot(couplings, final[:, 2], 's--', label='phase coherence |mean(exp(i theta))|')
states.set_xlabel('phase coupling K')
states.set_ylabel('order parameter at the end')
states.set_title('Asynchronous, phase wave, and synchronized disks')
states.legend()
fig.tight_layout()
plt.show()
