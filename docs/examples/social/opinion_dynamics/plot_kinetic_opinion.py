"""
# Kinetic opinion formation

A population is a distribution over seven opinion levels from -1 to 1. In every encounter an opinion moves one level
toward a close opinion and one level away from a distant one. With a narrow confidence and repulsion, an almost
uniform population polarises into the two extremes and the variance of the opinions grows toward one. With a wide
confidence it reaches a consensus and the variance falls to zero.
"""

# %%
# The population starts almost uniform, with a slight lean toward positive opinions, and is persuaded only by
# opinions at most two levels away. The blueprint shows the variance of the opinions, which grows from 0.44 toward 1.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe
from skcomplex.models.social.opinion_dynamics import opinion_grid

MODEL = 'kinetic_opinion'

# The distribution has no spatial view that shows polarisation, so the primary view is the variance over the run.
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin='/opinion_variance',
        name='variance of opinions',
        axis_x=rrb.TimeAxis(view_range=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
system = create(MODEL)
simulation = Simulation(system, observers=[observer], dt=0.5, observables=describe(MODEL).observables_of(system)).run(
    until=60,
)

# %%
# Consensus and polarisation
# --------------------------
# Encounters keep the total density, so every distribution sums to one. A confidence wide enough to reach every
# opinion should gather the population around the mean opinion, and narrower confidences with repulsion should push
# it into the extremes, so the final variance should grow as the confidence shrinks.


def final_distribution(confidence, repulsion):
    """Return the final distribution and variance of the opinions for a confidence and a repulsion."""
    model_system = create(MODEL, confidence=confidence, repulsion=repulsion)
    run = Simulation(model_system, dt=60.0, observables=describe(MODEL).observables_of(model_system)).run(until=60)
    return run.trajectory_['opinions.f'][-1], run.trajectory_['opinion_variance'][-1]


opinions = opinion_grid(7)
cases = {6: 0.0, 3: 0.1, 2: 0.1, 1: 0.2}
results = {confidence: final_distribution(confidence, repulsion) for confidence, repulsion in cases.items()}
fig, (distribution_axis, variance_axis) = plt.subplots(1, 2, figsize=(10, 4))
for confidence, (distribution, _) in results.items():
    distribution_axis.plot(opinions, distribution, marker='o', label=f'confidence {confidence} levels')
distribution_axis.set_xlabel('opinion')
distribution_axis.set_ylabel('density')
distribution_axis.set_title('Final distributions of opinions')
distribution_axis.legend()
variance_axis.plot(list(results), [variance for _, variance in results.values()], marker='o')
variance_axis.set_xlabel('confidence in levels')
variance_axis.set_ylabel('final variance of opinions')
variance_axis.set_title('Narrow confidence polarises, wide confidence agrees')
fig.tight_layout()
plt.show()
