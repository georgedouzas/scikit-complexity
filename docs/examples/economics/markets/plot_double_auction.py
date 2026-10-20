"""
# Double auction

Zero-intelligence traders quote random prices, yet a double auction with a budget constraint gives them nearly all of
the competitive surplus: the efficiency of every trading period stays above 90%. Without the constraint, traders make
losing trades and the efficiency drops.
"""

# %%
# Fifty buyers value one unit each and fifty sellers can each produce one unit at a cost. Every step is a trading
# period of 20 rounds: each trader that has not traded yet quotes a random price, below its value or above its cost,
# and a limit order book with price-time priority matches the quotes. A ledger then reports the realised surplus as a
# share of the maximum one, and every trader gets a new unit. The blueprint shows the efficiency of every period.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'double_auction'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters',
        contents=['/reporters/ledger.efficiency'],
        name='allocative efficiency',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, random_state=0), observers=[observer], random_state=0).run(until=20)

# %%
# The budget constraint, not intelligence
# ---------------------------------------
# Gode and Sunder (1993) found that the efficiency of markets of zero-intelligence traders with a budget constraint
# (ZI-C) is close to that of human traders, around 97-99%, while unconstrained traders (ZI-U) lose a large share of
# the surplus. Both protocols of the library, the continuous order book and the call auction at one uniform price,
# should reproduce the gap.

efficiencies = {}
for protocol in ('order_book', 'call_auction'):
    for constrained in (True, False):
        system = create(MODEL, protocol=protocol, constrained=constrained, random_state=0)
        trajectory = Simulation(system, random_state=0).run(until=20).trajectory_
        efficiencies[protocol, constrained] = trajectory['ledger.efficiency'][1:]
labels = [
    f'{protocol.replace("_", " ")}\n{"ZI-C" if constrained else "ZI-U"}' for protocol, constrained in efficiencies
]
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(list(efficiencies.values()))
ax.set_xticks(np.arange(1, len(labels) + 1), labels)
ax.axhline(0.9, ls='--', color='grey', label='90% of the maximum surplus')
ax.set_ylabel('allocative efficiency per period')
ax.set_title('Budget constraints make random traders efficient')
ax.legend()
fig.tight_layout()
plt.show()
