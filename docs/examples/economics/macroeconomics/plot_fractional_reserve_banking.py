"""
# Fractional-reserve banking

Banks keep a fraction of their deposits as reserves and lend the rest, and the loans come back as new deposits. The
broad money grows from the base money to the base money over the reserve ratio, and when the central bank doubles the
ratio, the broad money falls to half.
"""

# %%
# A hundred customers hold deposits of 1000 units of base money in five banks. Every step, in the phase `pay`,
# customers pay random customers and repay loans, and in ten rounds of the phase `lend` banks lend their excess
# reserves, which are spent and redeposited at random banks. A regulator institution sets the reserve ratio of the
# banks, 0.1 until step 150 and 0.2 after, as a live parameter recorded in the trajectory. The blueprint shows the
# broad money and the reserves.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'fractional_reserve_banking'
BASE = 1000.0

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/',
        contents=['/broad_money', '/reserves'],
        name='broad money and reserves',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
system = create(MODEL, base_money=BASE, ratio=0.1, new_ratio=0.2, change_at=150.0, random_state=0)
simulation = Simulation(
    system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(until=300)

# %%
# The money multiplier
# --------------------
# Reserves stay equal to the base money `B`, and banks lend until their reserves are the share `r` of their deposits,
# so the broad money should approach `B / r`, Hamill and Gilbert (2016 ch. 9): 10000 with `r = 0.1` and 5000 after
# the regulator raises it to 0.2.

trajectory = simulation.trajectory_
ratio = trajectory['banks.reserve_ratio']
fig, (money, policy) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
money.plot(trajectory.times, trajectory['broad_money'], label='broad money')
money.plot(trajectory.times, BASE / ratio, '--', color='black', label='base money / reserve ratio')
money.plot(trajectory.times, trajectory['loans'], label='loans')
money.set_ylabel('money')
money.set_title('Lending multiplies the base money')
money.legend()
policy.plot(trajectory.times, ratio, color='tab:red')
policy.set_xlabel('step')
policy.set_ylabel('reserve ratio')
policy.set_title('Set by the regulator during the run')
fig.tight_layout()
plt.show()
