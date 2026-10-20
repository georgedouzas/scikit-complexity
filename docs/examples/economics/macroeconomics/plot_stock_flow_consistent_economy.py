"""
# Stock-flow consistent monetary economy

Money circulates between households, firms, and a bank as deposits. Every payment is a flow, one sector's outflow and
another's inflow, and only lending and repayment create and destroy money. The economy settles where loans, deposits,
and sales stop changing.
"""

# %%
# The economy starts with 60 in household deposits, 20 in firm deposits, 10 in bank funds, and 50 in loans, and runs
# for 100 periods. A conservation monitor checks at every period that the total money changes only by lending minus
# repayment. The blueprint shows the deposits of the three sectors above the loans of the firms.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import flow_matrix
from skcomplex.core.runtime import ConservationMonitor, Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stock_flow_consistent_economy'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.Horizontal(
            rrb.TimeSeriesView(
                origin='/households/deposits',
                name='household deposits',
                axis_x=rrb.TimeAxis(view_range=whole_run),
            ),
            rrb.TimeSeriesView(
                origin='/firms/deposits',
                name='firm deposits',
                axis_x=rrb.TimeAxis(view_range=whole_run),
            ),
            rrb.TimeSeriesView(origin='/bank/deposits', name='bank funds', axis_x=rrb.TimeAxis(view_range=whole_run)),
        ),
        rrb.TimeSeriesView(origin='/firms/loans', name='loans', axis_x=rrb.TimeAxis(view_range=whole_run)),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=1, observers=[observer, ConservationMonitor()]).run(100)

# %%
# Money changes only at the boundary
# ----------------------------------
# The total of the deposits should change each period by exactly the new loans minus the repayments, since the
# households, the firms, and the bank only pay each other. The transaction-flow matrix of a period lists every flow
# with its outflow and inflow, so its rows sum to zero.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
money = trajectory['households.deposits'] + trajectory['firms.deposits'] + trajectory['bank.deposits']
created = trajectory['circuit.lending'] - trajectory['circuit.repayments']
matrix = flow_matrix(trajectory, 10)
fig, (totals, table) = plt.subplots(1, 2, figsize=(11, 4))
totals.plot(times, money - money[0], label='change of total money')
totals.plot(times[1:], np.cumsum(created[:-1]), '--', color='black', label='cumulative lending - repayment')
totals.set_xlabel('period')
totals.set_ylabel('money')
totals.set_title('Money is created and destroyed only by the bank')
totals.legend()
image = table.imshow(
    matrix.values,
    cmap='coolwarm',
    vmin=-np.abs(matrix.values).max(),
    vmax=np.abs(matrix.values).max(),
)
table.set_xticks(range(len(matrix.holders)), [holder.split('.')[0] for holder in matrix.holders], rotation=30)
table.set_yticks(range(len(matrix.flows)), [flow.split('.')[1] for flow in matrix.flows])
table.set_title('Transaction-flow matrix at period 10')
fig.colorbar(image, ax=table)
fig.tight_layout()
plt.show()
