"""
# Urban scaling of interactions

A city grows by residents who link to residents close to them in hyperbolic space, and its residents also add links
among themselves at a fixed rate per resident. The links per resident therefore grow with the size of the city, and the
meetings of a city grow faster than its population, the superlinear scaling of socioeconomic quantities that Ribeiro and
Netto review near the exponent 1.15.
"""

# %%
# The city grows to 300 residents, one per step, and every resident meets each neighbour with probability one half per
# step. The view draws the residents by their meetings of the step.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.analysis import ScalingAnalysis
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create, describe
from skcomplex.models.social.organizations import urban_scaling_interactions

MODEL = 'urban_scaling_interactions'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
observables = describe(MODEL).observables_of(system)
simulation = (
    Simulation(system, observers=[observer], observables=observables, random_state=0).check().prepare().run(until=297)
)

# %%
# Meetings against the population
# -------------------------------
# An `Experiment` sweeps the size of the city at three rates of internal links, and `ScalingAnalysis(kind='allometric')`
# fits the exponent of the meetings with a confidence interval. Without internal links the exponent stays near 1, and
# it rises with the rate of the internal links.

sizes = [100, 200, 400, 800]
seeds = [0, 1, 2]
rates = [0.0, 0.002, 0.005]
fig, ax = plt.subplots(figsize=(7, 4))
for rate in rates:
    bounds = {'population': sizes, 'internal': [rate], 'random_state': seeds}
    experiment = Experiment(
        urban_scaling_interactions,
        GridDesign(),
        bounds,
        until=float(max(sizes)),
        outputs=['total_meetings'],
        observables=observables,
        random_state=0,
    )
    meetings = experiment.check().prepare().run().result_.series('total_meetings')[:, -1]
    samples = dict(zip(sizes, meetings.reshape(len(sizes), len(seeds)), strict=True))
    fit = ScalingAnalysis('allometric', resamples=500).check().prepare(samples).run()
    label = f'internal {rate}: exponent {fit.exponent_:.2f} [{fit.interval_[0]:.2f}, {fit.interval_[1]:.2f}]'
    ax.loglog(sizes, [np.mean(samples[size]) for size in sizes], 'o-', label=label)
ax.set_xlabel('population N')
ax.set_ylabel('meetings per step')
ax.set_title('Internal links make the meetings grow faster than the population')
ax.legend()
fig.tight_layout()
plt.show()
