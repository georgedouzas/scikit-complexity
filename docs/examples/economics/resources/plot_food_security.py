"""
# Food security

Rice self-sufficiency of a country whose population grows by two percent a year while its rice land is developed and
then lost to housing and industry. Input subsidies alone cannot hold self-sufficiency, which falls to about a fifth in
fifty years. Research, extension, bio-fertiliser, and a second crop each year raise the productivity enough to keep
the country near self-sufficiency for decades.
"""

# %%
# The run starts from the rice area, population, and potential yield of Malaysia in 2011, with two thirds of the
# fertiliser subsidised, and lasts fifty years. The blueprint shows the rice area and the population.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'food_security'

# Every view shows the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/rice/rice_area', name='rice area (Mha)', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/rice/population', name='population (millions)', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=0.25, observers=[observer]).run(50)

# %%
# Policies against a growing requirement
# --------------------------------------
# The requirement grows as `exp(0.021 t)`, almost threefold in fifty years, while the rice area shrinks by 0.85
# percent a year after the suitable land is developed. With unchanged productivity the self-sufficiency should fall
# roughly as `exp(-0.03 t)`. Withdrawing the subsidy lowers it at once, and the combined policy should raise it above
# one before the growth of the population catches up.

fig, ax = plt.subplots(figsize=(7, 4))
policies = {
    'subsidy 67%': {},
    'no subsidy': {'subsidy': 0.0},
    'half bio-fertiliser': {'subsidy': 0.335, 'bio_target': 0.5},
    'bio-fertiliser, research, extension, two crops': {
        'subsidy': 0.335,
        'bio_target': 0.5,
        'research': 1.0,
        'training_target': 1.0,
        'intensity_target': 2.0,
    },
}
for name, changes in policies.items():
    run = simulation.trajectory_ if not changes else Simulation(create(MODEL, **changes), dt=0.25).run(50).trajectory_
    ax.plot(np.asarray(run.times), run['rice.self_sufficiency'], label=name)
times = np.asarray(simulation.trajectory_.times)
start = simulation.trajectory_['rice.self_sufficiency'][0]
ax.plot(times, start * np.exp(-0.03 * times), 'k:', label='exp(-0.03 t)')
ax.axhline(1.0, color='grey', lw=0.8)
ax.set_xlabel('years')
ax.set_ylabel('production / requirement')
ax.set_title('Rice self-sufficiency under four policies')
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
