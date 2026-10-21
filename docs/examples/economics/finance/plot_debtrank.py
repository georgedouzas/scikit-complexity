"""
# DebtRank

A bank in distress loses part of its equity, and the banks that lent to it lose part of theirs in proportion to their
exposure. DebtRank follows this distress through the network of interbank loans, letting each bank pass it on only
once, and measures the share of the economic value of the system that ends up at risk. A few central banks can put
much more of the system at risk than their own size suggests.
"""

# %%
# Fifty banks lend to each other with probability 0.1, and the bank with the largest DebtRank defaults at the start.
# The blueprint shows the economic value at risk, which climbs round by round and stops once every reached bank has
# passed its distress on.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'debtrank'
N_BANKS, DENSITY, ROUNDS = 50, 0.1, 10


def cascade(bank):
    """Return the trajectory of the cascade from the default of one bank."""
    system = create(MODEL, n_banks=N_BANKS, density=DENSITY, distressed=(bank,), random_state=0)
    return Simulation(system, random_state=0).run(until=ROUNDS).trajectory_


def debtrank(trajectory):
    """Return the DebtRank of a cascade, the final value at risk minus the initial one."""
    return trajectory['value_at_risk'][-1] - trajectory['value_at_risk'][0]


ranks = np.array([debtrank(cascade(bank)) for bank in range(N_BANKS)])
system = create(MODEL, n_banks=N_BANKS, density=DENSITY, distressed=(int(ranks.argmax()),), random_state=0)
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/value_at_risk', name='value at risk', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=ROUNDS)

# %%
# DebtRank of every bank
# ----------------------
# Battiston and co-authors iterate `h_j <- min(1, h_j + sum_i W_ij h_i)` over the banks `i` distressed in the previous
# round, after which those banks become inactive, and sum `h_j v_j` over the banks. The same iteration written
# directly with the impact matrix `W` should give the DebtRank the model computes for every bank, and comparing the
# DebtRank with the economic value `v_i` of the defaulting bank shows which banks matter more than their size.


def direct(impact, values, bank):
    """Return the DebtRank of one bank by the iteration of Battiston and co-authors."""
    h, status = np.zeros(len(values)), np.zeros(len(values))
    h[bank], status[bank] = 1.0, 1
    while (status == 1).any():
        active = status == 1
        h = np.minimum(1.0, h + impact.T @ (h * active))
        status = np.where(active, 2, np.where((status == 0) & (h > 0), 1, status))
    return h @ values - values[bank]


impact = np.asarray(system.interactions[0].impact)
values = simulation.trajectory_.agents('banks', 0).values['value']
fig, (check, importance) = plt.subplots(1, 2, figsize=(10, 4))
check.plot(ranks, [direct(impact, values, bank) for bank in range(N_BANKS)], 'o')
check.plot([0, ranks.max()], [0, ranks.max()], 'k:', lw=0.8)
check.set_xlabel('DebtRank of the model')
check.set_ylabel('DebtRank of the direct iteration')
check.set_title('The cascade matches the DebtRank iteration')
importance.plot(values, ranks, 'o')
importance.set_xlabel('economic value v of the defaulting bank')
importance.set_ylabel('DebtRank')
importance.set_title('Systemic impact beyond size')
fig.tight_layout()
plt.show()
