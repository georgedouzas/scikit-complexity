"""
# Bouchaud-Mezard wealth

Every agent's wealth earns random multiplicative returns, and the agents exchange wealth with each other at a fixed
rate. The returns spread the wealth, the exchange pulls it back towards the mean, and the balance between them is a
stationary distribution of relative wealth with a Pareto tail, whose exponent `1 + 2 J / sigma^2` falls as the
returns become more volatile.
"""

# %%
# Five hundred agents start with one unit each and exchange with the mean at the rate `J = 0.5`, while their returns
# have the volatility `sigma = 1`. The total wealth grows without bound, so the blueprint shows the Gini coefficient of
# the wealth, an observable added to the model, which rises from zero and settles once the relative wealth is
# stationary.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb
from scipy import stats

from skcomplex.core.analysis import ccdf, gini
from skcomplex.core.runtime import Simulation
from skcomplex.core.system import Observable
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'bouchaud_mezard_wealth'
N_AGENTS, J, SIGMA, DT, UNTIL = 500, 0.5, 1.0, 0.01, 30.0

system = create(MODEL, n_agents=N_AGENTS, J=J, sigma=SIGMA)
observables = [*describe(MODEL).observables_of(system), Observable('gini', group='agents', variable='W', reduce=gini)]
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/gini', name='Gini coefficient', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=20)
simulation = Simulation(system, dt=DT, observers=[observer], random_state=0, observables=observables).run(until=UNTIL)

# %%
# Stationary distribution
# -----------------------
# On the complete graph, the relative wealth `w = W / <W>` has the stationary density
# `P(w) ~ exp(-(mu - 1) / w) / w^(1 + mu)`, an inverse gamma distribution of shape `mu` and scale `mu - 1`. Bouchaud and
# Mezard write the noise with variance `2 sigma^2`, which gives `mu = 1 + J / sigma^2`, so with the standard Brownian
# motion of this model `mu = 1 + 2 J / sigma^2 = 2`. Pooling the relative wealth after a transient of ten time units,
# the histogram should follow it, and the complementary cumulative distribution should fall as `w^-mu`.

trajectory = simulation.trajectory_
wealth = trajectory['agents.W']
relative = (wealth / wealth.mean(axis=1, keepdims=True))[trajectory.times > 10][::10].ravel()
mu = 1 + 2 * J / SIGMA**2
law = stats.invgamma(mu, scale=mu - 1)
grid = np.logspace(-1.5, 1.5, 200)
values, probabilities = ccdf(relative)
fig, (density, tail) = plt.subplots(1, 2, figsize=(10, 4))
density.hist(relative, bins=np.logspace(-1.5, 1.5, 50), density=True, alpha=0.5, label='simulation')
density.plot(grid, law.pdf(grid), 'k', label=f'inverse gamma, mu = {mu:.0f}')
density.set_xscale('log')
density.set_xlabel('relative wealth w')
density.set_ylabel('density')
density.set_title('Stationary distribution of relative wealth')
density.legend()
tail.loglog(values, probabilities, label='simulation')
tail.loglog(values, law.sf(values), 'k--', lw=0.8, label='inverse gamma')
tail.set_ylim(probabilities.min(), 1.5)
tail.set_xlabel('relative wealth w')
tail.set_ylabel('P(W / <W> > w)')
tail.set_title(f'Pareto tail w^-{mu:.0f}')
tail.legend()
fig.tight_layout()
plt.show()
