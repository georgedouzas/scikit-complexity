"""
# Toy firm investment

The toy model of Bargigli and co-authors: every firm invests in proportion to its last profit, produces in
proportion to its capital, and sells at a random price, and a firm whose net worth turns negative is replaced by a
small entrant. Investment out of random profits makes capital grow multiplicatively, so the firms become extremely
unequal in size, with a heavy, power-law tail, and aggregate production grows as the largest firms pull away.
"""

# %%
# A thousand firms start with capital and net worth 1, an investment accelerator `gamma = 1.1`, productivity
# `phi = 0.1`, and the interest rate `r = 0.1`, and trade for 250 periods. An observable added to the model measures
# the share of all capital held by the largest 1 percent of firms, which the blueprint shows climbing from 1 percent
# as a few lucky firms pull away.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.analysis import ccdf, fit_power_law
from skcomplex.core.model import Observable
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'toy_firm_investment'
N_FIRMS, STEPS = 1000, 250


def top_share(capital):
    """Return the share of the capital held by the largest 1 percent of firms."""
    return float(np.sort(capital)[-len(capital) // 100 :].sum() / capital.sum())


system = create(MODEL, n_firms=N_FIRMS, random_state=0)
system.set_params(
    observables=[*system.observables, Observable('top_share', group='firms', variable='K', reduce=top_share)],
)
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/top_share',
        name='capital share of the top 1%',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=STEPS)

# %%
# Firm sizes and concentration
# ----------------------------
# Each period multiplies a firm's capital by `1 + gamma (phi P - r)` with a uniform random price `P`, a Gibrat process
# whose resets of bankrupt firms keep the small firms coming. The complementary cumulative distribution of the final
# capital should have a power-law tail, which a maximum-likelihood fit locates, and the share of the capital held by
# the largest 1 percent of firms should keep growing while aggregate production rises.

trajectory = simulation.trajectory_
capital = trajectory.agents('firms', STEPS).values['K']
fit = fit_power_law(capital)
values, probabilities = ccdf(capital)
tail = values >= fit.xmin
anchor = probabilities[tail][0]
fig, (sizes, concentration) = plt.subplots(1, 2, figsize=(10, 4))
sizes.loglog(values, probabilities, label='simulation')
sizes.loglog(
    values[tail],
    anchor * (values[tail] / fit.xmin) ** (1 - fit.alpha),
    'k--',
    lw=0.8,
    label=f'power law, alpha = {fit.alpha:.2f}',
)
sizes.set_xlabel('capital K')
sizes.set_ylabel('P(K > k)')
sizes.set_title(f'Firm sizes after {STEPS} periods')
sizes.legend()
concentration.plot(trajectory.times, trajectory['top_share'], label='capital share of the top 1%')
concentration.set_xlabel('period')
concentration.set_ylabel('capital share of the largest 1%')
concentration.set_title('Capital concentrates as production grows')
production = concentration.twinx()
production.plot(trajectory.times[1:], trajectory['production'][1:], color='C1')
production.set_ylabel('aggregate production', color='C1')
fig.tight_layout()
plt.show()
