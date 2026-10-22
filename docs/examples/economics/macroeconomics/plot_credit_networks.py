"""
# Firm and bank credit networks

Firms borrow from banks, sample a few banks every step, and switch to a cheaper lender with a probability that grows
with the saving. The stronger the switching, the more the loans concentrate in the cheapest banks and the narrower the
spread of the rates that firms pay.
"""

# %%
# A hundred firms and ten banks trade in three phases per step: firms plan their loans, a sample-and-switch credit
# market matches them with banks, and loans are settled with interest or written off when a firm defaults. Banks quote
# rates that fall with their equity. The blueprint shows the Herfindahl index of the banks' loans and the spread of
# the rates over the cheapest one.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'credit_networks'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/',
        contents=['/herfindahl', '/reporters/credit.spread'],
        name='Herfindahl index and rate spread',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
model_system = create(MODEL, random_state=0)
simulation = Simulation(
    model_system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(model_system),
).run(until=100)

# %%
# Switching intensity and concentration
# -------------------------------------
# Caiani et al. (2016 ch. 2) let firms switch with probability `1 - exp(intensity (r_new - r_old) / r_new)`. With no
# switching every bank keeps its initial clients, so the Herfindahl index stays near `1 / banks = 0.1`, and with
# strong switching the loans flow to the banks with the lowest rates. The index should rise and the spread fall as
# the intensity grows.

intensities = np.array([0.0, 0.5, 2.0, 5.0, 20.0])
herfindahl, spread = [], []
for intensity in intensities:
    model_system = create(MODEL, intensity=intensity, random_state=0)
    trajectory = (
        Simulation(model_system, random_state=0, observables=describe(MODEL).observables_of(model_system))
        .run(100)
        .trajectory_
    )
    herfindahl.append(trajectory['herfindahl'][50:].mean())
    spread.append(np.nanmean(trajectory['credit.spread'][50:]))
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
left.plot(intensities, herfindahl, 'o-')
left.axhline(0.1, ls='--', color='grey', label='equal shares')
left.set_xlabel('switching intensity')
left.set_ylabel('Herfindahl index of loans')
left.set_title('Switching concentrates credit')
left.legend()
right.plot(intensities, spread, 'o-')
right.set_xlabel('switching intensity')
right.set_ylabel('mean rate spread')
right.set_title('Switching narrows the spread of rates')
fig.tight_layout()
plt.show()
