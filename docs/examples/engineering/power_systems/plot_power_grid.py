"""
# Braess's paradox in a power grid

A ring of generators and consumers settles into synchrony, every node turning at the grid frequency. Adding a line
should only help, yet it reroutes the flows so that one line overloads, and the grid loses synchrony.
"""

# %%
# The model is created by name from the catalog with eight nodes on a ring, four generators and four consumers, lines of
# capacity `K = 2.75`, and the extra line `(0, 2)`, and runs for 300 time units. The blueprint shows the frequency
# deviations of all eight nodes, which split apart when synchrony is lost.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'power_grid'

frequencies = rrb.TimeSeriesView(
    origin='/',
    contents=[f'/node_{node}/phi_d1' for node in range(8)],
    name='frequency deviations',
    axis_x=rrb.TimeAxis(
        view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
    ),
)
blueprint = rrb.Blueprint(
    frequencies,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, added_line=(0, 2))
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=1.0, observers=[observer]).run(until=300)

# %%
# Synchrony with and without the extra capacity
# ---------------------------------------------
# In synchrony every frequency deviation `phi_i'` decays to zero. The default ring should synchronise, while the ring
# with the added line `(0, 2)` or with the doubled line `(0, 1)` should keep frequency deviations that never decay.

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
grids = [('default ring', {}), ('added line (0, 2)', None), ('doubled line (0, 1)', {'doubled_line': (0, 1)})]
for ax, (label, arguments) in zip(axes, grids, strict=True):
    run = (
        simulation.trajectory_
        if arguments is None
        else Simulation(create(MODEL, **arguments), dt=1.0).run(300).trajectory_
    )
    for node in range(8):
        ax.plot(run.times, run[f'node_{node}.phi_d1'], linewidth=0.8)
    spread = np.abs([run[f'node_{node}.phi_d1'][-50:] for node in range(8)]).max()
    ax.set_title(f'{label}: max |phi\'| {spread:.2f}', fontsize=10)
    ax.set_xlabel('time')
axes[0].set_ylabel("frequency deviation phi'")
fig.suptitle("Braess's paradox: more capacity, no synchrony")
fig.tight_layout()
plt.show()
