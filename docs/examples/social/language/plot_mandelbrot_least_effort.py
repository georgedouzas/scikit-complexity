"""
# Mandelbrot least-effort word frequencies

A language minimises the mean cost of its words over its entropy, and a word of rank `i` costs `log(i + 1)`. The minimum
gives the power law of Zipf, `p_i = (i + 1)^-lambda`, whose exponent makes the probabilities sum to 1.
"""

# %%
# The model solves the exponent and the probabilities of 200 words once, from the stationarity of the ratio.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.experiments import Experiment, GridDesign
from skcomplex.models import create
from skcomplex.models.social.language import mandelbrot_least_effort

MODEL = 'mandelbrot_least_effort'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, observers=[observer]).prepare().run(until=1.0)
probabilities = simulation.trajectory_.values('words.probability')[0]

# %%
# The exponent against the vocabulary
# -----------------------------------
# An `Experiment` sweeps the number of words. The exponent grows towards 1.73, where the zeta function is 2, the limit
# of an infinite vocabulary.

sizes = [5, 10, 20, 50, 100, 200, 400]
experiment = Experiment(
    mandelbrot_least_effort,
    GridDesign(),
    {'n_words': sizes},
    until=1.0,
    outputs=['language.exponent'],
)
exponents = experiment.prepare().run().result_.series('language.exponent')[:, 0]
ranks = np.arange(1, probabilities.size + 1)
fig, (law, trend) = plt.subplots(1, 2, figsize=(10, 4))
law.loglog(ranks + 1, probabilities, '.')
law.set_xlabel('rank i + 1')
law.set_ylabel('probability')
law.set_title('A power law of rank')
trend.semilogx(sizes, exponents, 'o-')
trend.axhline(1.7286, color='black', linestyle='--', label='infinite vocabulary')
trend.set_xlabel('number of words')
trend.set_ylabel('exponent lambda')
trend.set_title('The exponent grows with the vocabulary')
trend.legend()
fig.tight_layout()
plt.show()
