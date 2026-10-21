"""
# Standing waves

A string with fixed ends, plucked into the shape of one of its modes, vibrates in place: the shape keeps its nodes,
where the string never moves, and only swells and flips with the mode frequency `ω = c m π / L`.
"""

# %%
# The string of length 2 starts in its third mode at rest, with the wave speed `c = 1`, so its period is
# `2 L / (m c) = 4 / 3` and the run covers three periods. The default blueprint shows the space-time image of the
# displacement first: the bands of the antinodes alternate in sign, and the two nodes stay still.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import fields_of
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'standing_waves'

system = create(MODEL, n_sites=99, length=2.0, c=1.0, mode=3, amplitude=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=4.0)

# %%
# Nodes and frequency
# -------------------
# The exact solution is `u(x, t) = sin(m π x / L) cos(ω t)` with `ω = c m π / L`. Snapshots of the string should all
# share the nodes at `x = L / 3` and `2 L / 3`, and the displacement at an antinode should follow `cos(ω t)`.

displacement = fields_of(simulation.trajectory_, 'string.u')
times = np.asarray(simulation.trajectory_.times)
positions = 2.0 / 100 * np.arange(1, 100)
omega = 3 * np.pi / 2.0
fig, (shapes, antinode) = plt.subplots(1, 2, figsize=(10, 4))
for frame in range(0, 14, 2):
    shapes.plot(positions, displacement[frame], linewidth=1, label=f't = {times[frame]:.1f}')
for node in (2 / 3, 4 / 3):
    shapes.axvline(node, linestyle='--', color='grey')
shapes.set_xlabel('position along the string')
shapes.set_ylabel('displacement')
shapes.set_title('Snapshots over half a period share the nodes')
shapes.legend(fontsize=7)
middle = np.argmin(np.abs(positions - 1 / 3))
antinode.plot(times, displacement[:, middle], label='simulated antinode')
antinode.plot(
    times,
    np.sin(3 * np.pi * positions[middle] / 2.0) * np.cos(omega * times),
    '--',
    color='black',
    label='exact',
)
antinode.set_xlabel('time')
antinode.set_ylabel('displacement')
antinode.set_title('The antinode oscillates at ω = 3π / 2')
antinode.legend()
fig.tight_layout()
plt.show()
