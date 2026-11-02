"""
# Gravity model of urban scaling

If the chance that two residents interact decays as `r^-gamma`, and a city of `N` residents covers the area `N^alpha`,
the total interaction of a city grows as `N^(2 - alpha gamma / 2)`: faster than its population for a small enough decay,
the
increasing returns of urban scaling.
"""

# %%
# The model sums the interaction of each of 400 residents with all the others once, at positions drawn in a disc.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import ScalingAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create, describe
from skcomplex.models.social.organizations import gravity_urban_scaling

MODEL = 'gravity_urban_scaling'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
observables = describe(MODEL).observables_of(system)
simulation = Simulation(system, observers=[observer], observables=observables).check().prepare().run(until=1.0)

# %%
# Total interaction against the population
# ----------------------------------------
# An `Experiment` sweeps the population, and `ScalingAnalysis(kind='power')` fits the exponent.

populations = [100, 200, 400, 800]
bounds = {'population': populations, 'gamma': [1.0]}
experiment = Experiment(
    gravity_urban_scaling,
    GridDesign(),
    bounds,
    until=1.0,
    outputs=['total_contacts'],
    observables=observables,
)
totals = experiment.check().prepare().run().result_.series('total_contacts')[:, 0]
scaling = ScalingAnalysis('power', parameters=populations, critical=0.0, above=True)
exponent = float(np.ravel(scaling.check().prepare(totals).run().exponent_)[0])
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(populations, totals, 'o', label=f'fitted exponent {exponent:.2f}')
ax.loglog(populations, totals[0] * (np.array(populations) / populations[0]) ** 1.5, color='black', label='exponent 1.5')
ax.set_xlabel('population N')
ax.set_ylabel('total interaction')
ax.set_title('Interaction grows faster than the population')
ax.legend()
fig.tight_layout()
plt.show()
