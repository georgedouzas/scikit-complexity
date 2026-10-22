"""
# Charged Schrödinger equation

A charged wave packet at rest feels an electric field through the vector potential in its covariant derivative,
`d_x - i q A_x`. While the field is on, the packet accelerates uniformly, and once it is off, the packet keeps the
momentum it gained and moves at constant speed. The space-time image shows the real part of the wave function, one row
per saved step with the earliest at the top: the packet bends to the right along a parabola for the two time units of
the field and then runs along a straight line, while its real part alternates in sign as its phase turns.
"""

# %%
# The model is created by name from the catalog: a Gaussian packet of width 2 at rest at the centre of a periodic line
# of length 80, with `D = 1/2` and charge `q = 1`, in the electric field `E = 1` switched on from `t = 0` to `t = 2`.
# The observer saves every frame of 0.1 time units with the default blueprint.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'charged_schrodinger'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.1, observers=[observer], random_state=0).run(until=8.0)

# %%
# Acceleration by the field, not by a gauge
# -----------------------------------------
# A mode of wave number `k` moves at the group velocity `2 D (k - q A_x)`, so the field, which makes
# `A_x = -E t`, accelerates the centre of `|psi|^2` at `2 D q E`: it should follow `D q E t^2` up to `t = 2` and then a
# line of slope `2 D q E t_E = 2`. A constant `A_x` alone is a pure gauge: a packet of momentum `k = q A_x` should stay
# at rest, since only the kinetic momentum `k - q A_x` moves it. The norm `sum |psi|^2` is conserved in both runs.

positions = 80 / 256 * np.arange(256) - 40
times = simulation.trajectory_.times
density = simulation.trajectory_['packet.u'] ** 2 + simulation.trajectory_['packet.w'] ** 2
centres = density @ positions / density.sum(axis=1)
expected = np.where(times < 2.0, 0.5 * times**2, 2.0 + 2.0 * (times - 2.0))
gauge = Simulation(create(MODEL, E=0.0, A_x=1.0, momentum=1.0), dt=0.1).run(until=8.0).trajectory_
gauge_density = gauge['packet.u'] ** 2 + gauge['packet.w'] ** 2

fig, (motion, norm) = plt.subplots(1, 2, figsize=(10, 4))
motion.plot(times, centres, label='field E = 1 for t < 2')
motion.plot(times, expected, 'k--', label='D q E t^2, then constant speed')
motion.plot(gauge.times, gauge_density @ positions / gauge_density.sum(axis=1), label='pure gauge A_x = k / q')
motion.set_xlabel('time')
motion.set_ylabel('centre of |psi|^2')
motion.set_title('The field accelerates the packet, the gauge does not')
motion.legend()
norm.plot(times, density.sum(axis=1) / density[0].sum() - 1, label='field')
norm.plot(gauge.times, gauge_density.sum(axis=1) / gauge_density[0].sum() - 1, label='pure gauge')
norm.set_xlabel('time')
norm.set_ylabel('relative change of the norm')
norm.set_title('The norm is conserved')
norm.legend()
fig.tight_layout()
plt.show()
