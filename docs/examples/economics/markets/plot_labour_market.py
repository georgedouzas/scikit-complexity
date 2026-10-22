"""
# Labour market

Firms post vacancies with wage offers, and unemployed workers, whose reservation wages fall the longer they search,
take the best offer they accept. Jobs are destroyed at a fixed rate, and unemployment falls from half the workforce to
the level where the flows into and out of unemployment balance.
"""

# %%
# Two hundred workers and forty firms trade in two phases per step. In the phase of offers each job is destroyed with
# probability `s = 0.05`, every firm posts a vacancy with probability 0.3 at a wage drawn between 0.5 and 1, and the
# reservation wage of a searching worker falls by 10 percent per step of search from 1. In the phase of matching a
# deferred-acceptance market pairs vacancies and workers. The blueprint shows the unemployment rate and the hires.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'labour_market'
SEPARATION = 0.05

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/unemployment', name='unemployment rate', axis_x=rrb.TimeAxis(view_range=whole_run)),
        rrb.TimeSeriesView(
            origin='/reporters',
            contents=['/reporters/labour.volume'],
            name='hires',
            axis_x=rrb.TimeAxis(view_range=whole_run),
        ),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL, n_workers=200, n_firms=40, separation=SEPARATION, random_state=0)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=200)

# %%
# Flow balance
# ------------
# In a steady state the jobs destroyed, `s (1 - u)` per worker, equal the hires, `f u`, where `f` is the share of
# searching workers hired in a step. So the unemployment rate should settle at `s / (s + f)`. The job-finding rate is
# measured from the hires and the searching workers after the transient, and the wages accepted fall below the
# initial reservation wage of 1 as long searches wear it down.

trajectory = simulation.trajectory_
times, unemployment = trajectory.times, trajectory['unemployment']
hires, searching = trajectory['labour.volume'], trajectory['office.searching']
window = slice(50, None)
finding = np.nansum(hires[window]) / np.nansum(searching[window])
balance = SEPARATION / (SEPARATION + finding)
workers = trajectory.agents('workers', len(times) - 1)
wages = workers.values['wage'][workers.values['employer'] >= 0]
fig, (series, histogram) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(times, unemployment, label='simulation')
series.axhline(balance, color='black', linestyle='--', label=f's / (s + f) = {balance:.3f}')
series.set_xlabel('step')
series.set_ylabel('unemployment rate')
series.set_title(f'Unemployment settles at the flow balance (f = {finding:.2f})')
series.legend()
histogram.hist(wages, bins=20)
histogram.set_xlabel('wage')
histogram.set_ylabel('employed workers')
histogram.set_title('Wages of the employed at the end')
fig.tight_layout()
plt.show()
