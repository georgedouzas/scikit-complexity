"""
# Lévy walk

The walkers fly in straight lines at constant speed and turn at random times. The flight times have no variance, so a
few walkers make very long flights and the cloud spreads faster than diffusion, with long straight arms.
"""

# %%
# The model is created by name from the catalog with 500 walkers in the plane, whose flight times have the tail
# `tau^(-5/2)`, and simulated to time 500. Rate laws move the walkers between turns, and the solver stops every walker
# at its own turns. The observer saves every frame with the default blueprint, whose primary view shows the walkers.

import itertools

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'levy_walk'

system = create(MODEL, n_walkers=500, sigma=1.5)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, solver=RK4(dt=0.5), dt=2.5, observers=[observer], random_state=0).run(until=500)

# %%
# Superdiffusion and ballistic flights
# ------------------------------------
# For flight times of tail exponent `1 < sigma < 2`, the ensemble mean squared displacement grows as `t^(3 - sigma)`,
# here `t^1.5`, between diffusion and ballistic motion. The curve should follow the dashed line of slope 1.5 at long
# times. Between two turns a walker moves in a straight line, so its position at a turn is the position at the previous
# turn plus the velocity times the flight time, which the renewal records confirm to rounding error.

trajectory = simulation.trajectory_
times = trajectory.times[1:]
squared = np.mean(np.sum(trajectory['walkers.x'][1:] ** 2, axis=-1), axis=1)
late = times > 50
slope = np.polyfit(np.log(times[late]), np.log(squared[late]), 1)[0]
turns = [
    (time, before, after)
    for record in trajectory.events
    for time, member, before, after in zip(record.times, record.members, record.before, record.after, strict=True)
    if member == 0
]
errors = [
    np.max(np.abs(later[:2] - (earlier[:2] + earlier[2:] * (end - start))))
    for (start, _, earlier), (end, later, _) in itertools.pairwise(turns)
]
fig, (displacement, path) = plt.subplots(1, 2, figsize=(10, 4))
displacement.loglog(times, squared, label=f'simulation, slope {slope:.2f} after t = 50')
displacement.loglog(times, squared[-1] * (times / times[-1]) ** 1.5, '--', color='black', label='slope 1.5')
displacement.loglog(times, squared[-1] * (times / times[-1]), ':', color='gray', label='diffusion, slope 1')
displacement.set_xlabel('time t')
displacement.set_ylabel('ensemble mean squared displacement')
displacement.set_title('Superdiffusion, <x^2> ~ t^(3 - sigma)')
displacement.legend()
path.plot(trajectory['walkers.x'][:, 0, 0], trajectory['walkers.x'][:, 0, 1])
path.plot([0.0], [0.0], 'o', color='black')
path.set_xlabel('x')
path.set_ylabel('y')
path.set_title(f'Straight flights of one walker, largest error {max(errors, default=0.0):.1e}')
path.axis('equal')
fig.tight_layout()
plt.show()
