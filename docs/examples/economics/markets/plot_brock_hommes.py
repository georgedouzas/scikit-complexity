"""
# Brock-Hommes heuristic switching

Traders forecast the price of a risky asset either as fundamentalists, who expect the fundamental value, or as trend
chasers, who extrapolate the last deviation, and switch towards the heuristic that earned more. When traders switch
fast enough, the price no longer settles at its fundamental value but fluctuates around it in bubbles and crashes.
"""

# %%
# Two hundred traders choose a heuristic every step by a logit choice of intensity `beta` on its last realised profit.
# The fundamentalist heuristic costs `C = 1`, the trend chasers extrapolate with `g = 1.2`, and the riskless return is
# `R = 1.1`. A market maker clears the deviation of the price from the fundamental at `R x_t = sum_h n_h f_h + eps`,
# with a small noise of standard deviation 0.01. The blueprint shows the deviation for `beta = 6`, above the critical
# intensity.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb
from scipy.special import expit

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'brock_hommes'
RATE, STRENGTH, COST, NOISE = 1.1, 1.2, 1.0, 0.01

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/reporters',
        contents=['/reporters/pricing.deviation'],
        name='price deviation x',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, n_traders=200, intensity=6.0), observers=[observer], random_state=0).run(100)

# %%
# Bifurcation with the intensity of choice
# ----------------------------------------
# With infinitely many traders the shares are the logit probabilities and the deviation follows the Brock-Hommes
# map. At the fundamental steady state the share of trend chasers is `1 / (1 + exp(-beta C))`, which makes it unstable
# once it exceeds `R / g`, beyond `beta* = ln(R / (g - R)) / C`, about 2.4. The bifurcation diagram plots the
# deviations the noisy map visits for every `beta`: a narrow band around zero below `beta*`, and above it cycles of
# bubbles and crashes that widen with `beta`. The spread of the simulated deviations, at a few intensities, should
# follow it.


def mean_field(intensity, steps=600, seed=1):
    """Return the deviations of the Brock-Hommes map with infinitely many traders."""
    rng = np.random.default_rng(seed)
    history, previous, fitness, deviations = 0.1, None, np.zeros(2), []
    for _ in range(steps):
        share = expit(intensity * (fitness[1] - fitness[0]))
        forecasts = np.array([0.0, STRENGTH * history])
        price = (forecasts @ [1 - share, share] + rng.normal(0.0, NOISE)) / RATE
        fitness = np.zeros(2) if previous is None else (price - RATE * history) * (previous - RATE * history)
        fitness = fitness - [COST, 0.0]
        previous, history = forecasts, price
        deviations.append(price)
    return np.array(deviations)


critical = np.log(RATE / (STRENGTH - RATE)) / COST
intensities = np.linspace(0.0, 8.0, 81)
samples = {intensity: mean_field(intensity)[-300:] for intensity in intensities}
simulated = {
    intensity: Simulation(create(MODEL, n_traders=200, intensity=intensity), random_state=0)
    .run(100)
    .trajectory_['pricing.deviation'][-50:]
    for intensity in (1.0, 2.0, 4.0, 6.0)
}
fig, (diagram, spread) = plt.subplots(1, 2, figsize=(11, 4))
for intensity, values in samples.items():
    diagram.plot(np.full(len(values), intensity), values, ',', color='black', alpha=0.4)
diagram.axvline(critical, color='C0', linestyle='--', label='beta*')
diagram.set_xlabel('intensity of choice beta')
diagram.set_ylabel('deviation x')
diagram.set_title('Bifurcation diagram of the mean-field map')
diagram.legend()
spread.plot(intensities, [values.std() for values in samples.values()], color='black', label='mean-field map')
spread.plot(list(simulated), [values.std() for values in simulated.values()], 'o', color='C1', label='200 traders')
spread.axvline(critical, color='C0', linestyle='--', label='beta*')
spread.set_xlabel('intensity of choice beta')
spread.set_ylabel('standard deviation of x')
spread.set_title('The fundamental loses stability at beta*')
spread.legend()
fig.tight_layout()
plt.show()
