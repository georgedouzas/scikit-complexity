"""
# Declining energy return

A society extracts a finite energy reserve and reinvests part of the net energy in more extraction. As the reserve
depletes, the energy return on investment falls, so the net energy peaks before the gross extraction, and the net
energy per person, with a population that keeps growing, peaks earlier still.
"""

# %%
# The reserve holds 1000 energy units with an initial return of 20 to 1, a tenth of the net energy is reinvested, and
# the population grows by 1.5 percent a year. The run lasts 250 years. The blueprint shows the extraction capacity
# and the reserve.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'energy_return'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/energy/capacity', name='extraction capacity', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/energy/reserve', name='reserve', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1.0, observers=[observer]).run(250)

# %%
# Three peaks
# -----------
# Every unit leaving the reserve is either extraction cost or net energy, so the cost and the net energy add up to
# the gross energy at every year. The cost share is the inverse of the energy return, which grows as the reserve
# empties, so the net energy should peak a few years before the gross energy and fall faster. Dividing by a growing
# population moves the peak of the net energy per person about fifteen years earlier.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
series = {
    'gross energy': trajectory['energy.gross_energy'],
    'net energy': trajectory['energy.net_energy'],
    'extraction cost': trajectory['energy.energy_cost'],
}
fig, (flows, per_capita) = plt.subplots(1, 2, figsize=(10, 4))
for name, values in series.items():
    peak = times[np.argmax(values)]
    flows.plot(times, values, label=f'{name}, peak at year {peak:.0f}')
flows.set_xlabel('years')
flows.set_ylabel('energy per year')
flows.set_title('Gross energy, its cost, and the net energy')
flows.legend(fontsize=8)
net_per_capita = trajectory['energy.net_per_capita']
per_capita.plot(times, net_per_capita, label=f'peak at year {times[np.argmax(net_per_capita)]:.0f}')
per_capita.set_xlabel('years')
per_capita.set_ylabel('net energy per person per year')
per_capita.set_title('Net energy per person')
per_capita.legend(fontsize=8)
fig.tight_layout()
plt.show()
