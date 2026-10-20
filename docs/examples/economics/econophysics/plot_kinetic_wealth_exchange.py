"""
# Kinetic wealth exchange

Agents who trade random shares of their pooled wealth end up with an exponential, Boltzmann-Gibbs distribution of
wealth. When every agent saves a fraction of its wealth out of each trade, the poor become rare and the distribution
turns into a Gamma distribution with a most probable wealth above zero, and when agents may go into debt the
exponential shifts to the debt limit.
"""

# %%
# Five hundred agents start with one unit of wealth each and make, on average, two trades per step. The first run
# saves nothing. The blueprint shows the variance of wealth, which grows from zero to the variance of the exponential
# distribution with mean one, which is one.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb
from scipy import stats

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'kinetic_wealth_exchange'
STEPS = 150

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/variance', name='variance of wealth', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_agents=500), observers=[observer], random_state=0).run(until=STEPS)

# %%
# Stationary distributions
# ------------------------
# Chakraborti and Chakrabarti's trades with saving propensity `lambda` lead to a distribution close to the Gamma
# distribution of shape `n = 1 + 3 lambda / (1 - lambda)` and mean one, the exponential for `lambda = 0`. Dragulescu
# and Yakovenko's fixed-amount trades with debt limit `d` lead to the exponential of temperature `<w> + d`, shifted to
# `-d`. The histograms of the final wealth should follow these curves.


def final_wealth(**options):
    """Return the final wealth of every agent of a run."""
    trajectory = Simulation(create(MODEL, n_agents=500, **options), random_state=0).run(until=STEPS).trajectory_
    return trajectory.agents('traders', STEPS).values['wealth']


grid = np.linspace(0, 5, 200)
fig, (saving_axis, debt_axis) = plt.subplots(1, 2, figsize=(10, 4))
samples = {0.0: simulation.trajectory_.agents('traders', STEPS).values['wealth']}
samples |= {saving: final_wealth(saving=saving) for saving in (0.5, 0.8)}
for (saving, wealth), color in zip(samples.items(), ['C0', 'C1', 'C2'], strict=True):
    shape = 1 + 3 * saving / (1 - saving)
    saving_axis.hist(wealth, bins=40, range=(0, 5), density=True, alpha=0.4, color=color)
    saving_axis.plot(grid, stats.gamma.pdf(grid, shape, scale=1 / shape), color=color, label=f'lambda = {saving}')
saving_axis.set_xlabel('wealth w')
saving_axis.set_ylabel('density')
saving_axis.set_title('Saving: Gamma laws of shape 1 + 3 lambda / (1 - lambda)')
saving_axis.legend()
debt = 1.0
wealth = final_wealth(debt=debt, amount=0.25)
temperature = wealth.mean() + debt
shifted = np.linspace(-debt, 10, 200)
debt_axis.hist(wealth, bins=40, density=True, alpha=0.4)
debt_axis.plot(shifted, np.exp(-(shifted + debt) / temperature) / temperature, color='black', label='exp(-(w + d) / T)')
debt_axis.axvline(-debt, color='grey', linestyle=':', label='debt limit')
debt_axis.set_xlabel('wealth w')
debt_axis.set_title(f'Debt limit d = {debt}: shifted exponential, T = {temperature:.1f}')
debt_axis.legend()
fig.tight_layout()
plt.show()
