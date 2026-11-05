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

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system, describe
from skcomplex.systems.economics.econophysics import wealth_classes

SYSTEM = 'kinetic_wealth_welfare'

# The distribution has no spatial view that shows inequality, so the primary view is the Gini coefficient over the run.
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/gini', name='Gini coefficient', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
system = create_system(SYSTEM)
simulation = (
    Simulation(
        system,
        observers=[observer],
        dt=1.0,
        observables=describe(SYSTEM).observables_of(system),
    )
    .prepare()
    .run(
        until=100,
    )
)

# %%
# Stationary distributions under welfare policies
# -----------------------------------------------
# A stronger policy should keep more of the population in the middle classes, and the stationary Gini coefficient
# should fall steadily as the tax rate grows, while the total density stays one.


def stationary(tax):
    """Return the stationary distribution and Gini coefficient of a run with a policy of a tax rate."""
    system_system = create_system(SYSTEM, tax=tax)
    run = (
        Simulation(system_system, dt=100.0, observables=describe(SYSTEM).observables_of(system_system))
        .prepare()
        .run(until=200)
    )
    return run.trajectory_.values('wealth.f')[-1], run.trajectory_.values('gini')[-1]


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
