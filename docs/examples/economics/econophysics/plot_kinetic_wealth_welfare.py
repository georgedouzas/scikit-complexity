"""
# Kinetic wealth and welfare dynamics

A population spread over wealth classes meets in pairs, and every encounter moves a particle one class up or down.
When competition beats cooperation, wealth polarises toward the poorest and the richest classes and the Gini
coefficient rises to a high stationary value. A welfare policy, an encounter with a group of fixed density that moves
the richest classes down and the poorest up, keeps the middle class and lowers the stationary Gini coefficient.
"""

# %%
# The population starts concentrated around the middle class, and encounters between classes run without a policy.
# The blueprint shows the Gini coefficient, which rises from about 0.17 to a stationary value above 0.4.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe
from skcomplex.models.economics.econophysics import wealth_classes

MODEL = 'kinetic_wealth_welfare'

# The distribution has no spatial view that shows inequality, so the primary view is the Gini coefficient over the run.
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/gini', name='Gini coefficient', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL)
simulation = Simulation(system, observers=[observer], dt=1.0, observables=describe(MODEL).observables_of(system)).run(
    until=100,
)

# %%
# Stationary distributions under welfare policies
# -----------------------------------------------
# A stronger policy should keep more of the population in the middle classes, and the stationary Gini coefficient
# should fall steadily as the tax rate grows, while the total density stays one.


def stationary(tax):
    """Return the stationary distribution and Gini coefficient of a run with a policy of a tax rate."""
    model_system = create(MODEL, tax=tax)
    run = Simulation(model_system, dt=100.0, observables=describe(MODEL).observables_of(model_system)).run(until=200)
    return run.trajectory_['wealth.f'][-1], run.trajectory_['gini'][-1]


wealth = wealth_classes(9)
taxes = np.linspace(0.0, 0.4, 9)
results = [stationary(tax) for tax in taxes]
fig, (distribution_axis, gini_axis) = plt.subplots(1, 2, figsize=(10, 4))
for tax, (distribution, _) in zip(taxes[[0, 4, 8]], [results[0], results[4], results[8]], strict=True):
    distribution_axis.plot(wealth, distribution, marker='o', label=f'tax = {tax:.2f}')
distribution_axis.set_xlabel('wealth class')
distribution_axis.set_ylabel('density')
distribution_axis.set_title('Stationary distributions over wealth classes')
distribution_axis.legend()
gini_axis.plot(taxes, [gini for _, gini in results], marker='o')
gini_axis.set_xlabel('tax rate of the welfare policy')
gini_axis.set_ylabel('stationary Gini coefficient')
gini_axis.set_title('Welfare policies lower inequality')
fig.tight_layout()
plt.show()
