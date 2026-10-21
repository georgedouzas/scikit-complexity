"""
# Burridge-Knopoff fault

The plate drags the blocks of the fault slowly, while friction holds them in place. Stress builds up until a block
slips and pulls its neighbours along: the fault advances in sudden jumps of one block or of many, earthquakes of many
sizes separated by long quiet loading.
"""

# %%
# The model is created by name from the catalog with 20 blocks, the coupling `k_c = 2.5`, and the plate speed 0.01,
# and runs for 2000 time units, about twenty loading cycles. The blueprint puts the history of the block speeds
# first, where blocks caught slipping light up against the stuck ones.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver, default_blueprint
from skcomplex.models import create

MODEL = 'burridge_knopoff'

system = create(MODEL)
observer = RerunObserver(
    MODEL,
    save_path=f'{MODEL}.rrd',
    blueprint=default_blueprint(system, primary='fault/space_time/v'),
)
simulation = Simulation(system, dt=10.0, observers=[observer]).run(until=2000)

# %%
# Stick and slip
# --------------
# Between earthquakes a block creeps at about the plate speed, `0.01 * 10 = 0.1` over one frame, while a slip moves it
# much farther, so a displacement above 0.3 marks a slip. The total slip of all blocks over a frame then measures
# the size of the earthquakes in it: most frames should be quiet, and the others should spread over a wide range of
# sizes.

x = simulation.trajectory_['fault.x']
steps = np.diff(x, axis=0)
slips = np.where(steps > 0.3, steps, 0.0)
sizes = slips.sum(axis=1)
fig, (blocks, events) = plt.subplots(1, 2, figsize=(10, 4))
blocks.plot(simulation.trajectory_.times, x[:, [0, 5, 10, 15]], linewidth=1)
blocks.plot(simulation.trajectory_.times, 0.01 * simulation.trajectory_.times, color='black', linestyle='--')
blocks.set_xlabel('time')
blocks.set_ylabel('block position')
blocks.set_title('Blocks advance in jumps behind the plate (dashed)')
events.stem(simulation.trajectory_.times[1:], sizes)
events.set_xlabel('time')
events.set_ylabel('total slip in the frame')
events.set_title(f'{np.count_nonzero(sizes)} of {len(sizes)} frames hold earthquakes')
fig.tight_layout()
plt.show()
