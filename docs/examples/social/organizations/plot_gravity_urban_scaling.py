"""
# Gravity system of urban scaling

If the chance that two residents interact decays as `r^-gamma`, and a city of `N` residents covers the area `N^alpha`,
the total interaction of a city grows as `N^(2 - alpha gamma / 2)`: faster than its population for a small enough decay,
the
increasing returns of urban scaling.
"""

# %%
# The system sums the interaction of each of 400 residents with all the others once, at positions drawn in a disc.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import scaling
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems.social.organizations import GravityUrbanScalingSystem

SYSTEM = 'gravity_urban_scaling'

system = GravityUrbanScalingSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
observables = system.get_run_settings()['observables']
trajectory = simulate(system, until=1.0, observables=observables, observers=[observer])

# %%
# Total interaction against the population
# ----------------------------------------
# An `Experiment` sweeps the population, and `scaling(kind='power')` fits the exponent.

populations = [100, 200, 400, 800]
bounds = {'population': populations, 'gamma': [1.0]}
experiment = Experiment(
    lambda **parameters: GravityUrbanScalingSystem(**parameters).prepare().system_,
    GridDesign(),
    bounds,
    until=1.0,
    outputs=['total_contacts'],
    observables=observables,
)
totals = experiment.prepare().run().result_.series('total_contacts')[:, 0]
fit = scaling(totals, kind='power', parameters=populations, critical=0.0, above=True)
exponent = float(np.ravel(fit.exponent)[0])
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(populations, totals, 'o', label=f'fitted exponent {exponent:.2f}')
ax.loglog(populations, totals[0] * (np.array(populations) / populations[0]) ** 1.5, color='black', label='exponent 1.5')
ax.set_xlabel('population N')
ax.set_ylabel('total interaction')
ax.set_title('Interaction grows faster than the population')
ax.legend()
fig.tight_layout()
plt.show()
