"""
# Burridge-Knopoff fault

The plate drags the blocks of the fault slowly, while friction holds them in place. Stress builds up until a block
slips and pulls its neighbours along: the fault advances in sudden jumps of one block or of many, earthquakes of many
sizes separated by long quiet loading.
"""

# %%
# The system is built from its class with 20 blocks, the coupling `k_c = 2.5`, and the plate speed 0.01,
# and runs for 2000 time units, about twenty loading cycles, with the system's registered fixed-step
# `RK4Solver(dt=0.05)`, since the adaptive integrators stall where blocks stick and slip. The blueprint puts the history
# of the block speeds first, where blocks caught slipping light up against the stuck ones.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.seismology import BurridgeKnopoffSystem

SYSTEM = 'burridge_knopoff'

system = BurridgeKnopoffSystem()
observer = RerunObserver(
    SYSTEM,
    save_path=f'{SYSTEM}.rrd',
    blueprint='default',
    primary='fault/space_time/v',
)
trajectory = simulate(system, until=2000, dt=10.0, observers=[observer])

# %%
# Stick and slip
# --------------
# Between earthquakes a block creeps at about the plate speed, `0.01 * 10 = 0.1` over one frame, while a slip moves it
# much farther, so a displacement above 0.3 marks a slip. The total slip of all blocks over a frame then measures
# the size of the earthquakes in it: most frames should be quiet, and the others should spread over a wide range of
# sizes.

x = trajectory.values('fault.x')
steps = np.diff(x, axis=0)
slips = np.where(steps > 0.3, steps, 0.0)
sizes = slips.sum(axis=1)
fig, (blocks, events) = plt.subplots(1, 2, figsize=(10, 4))
blocks.plot(trajectory.times, x[:, [0, 5, 10, 15]], linewidth=1)
blocks.plot(trajectory.times, 0.01 * trajectory.times, color='black', linestyle='--')
blocks.set_xlabel('time')
blocks.set_ylabel('block position')
blocks.set_title('Blocks advance in jumps behind the plate (dashed)')
events.stem(trajectory.times[1:], sizes)
events.set_xlabel('time')
events.set_ylabel('total slip in the frame')
events.set_title(f'{np.count_nonzero(sizes)} of {len(sizes)} frames hold earthquakes')
fig.tight_layout()
plt.show()
