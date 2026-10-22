"""
# Kinouchi-Copelli excitable network

A thousand excitable elements on a random network excite their quiescent neighbours with a probability set by the
branching ratio, the mean number of elements one excitation excites. At the critical branching ratio of 1, a weak
external stimulus sets off avalanches of every size, which neither die out at once nor sustain themselves. The view
shows the fraction of excited elements over time, rising in irregular bursts of very different heights.
"""

# %%
# The model is created by name from the catalog on an Erdős-Rényi graph of 1000 nodes with mean degree 10, at the
# branching ratio `sigma = 1`, with 5 states per cycle, every element quiescent at the start, and a weak stimulus of
# rate `r = 0.002` per step, so about two elements per step are excited from outside. The blueprint shows the standard
# observable `activity`, the fraction of excited elements, over the whole run of 280 steps.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'kinouchi_copelli'

graph = nx.gnp_random_graph(1000, 10 / 999, seed=0)
whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/activity', name='excited fraction', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL, graph=graph, sigma=1.0, stimulus=0.002, excited=0.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(
    system,
    dt=1,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(system),
).run(280)

# %%
# Response curves and the dynamic range
# -------------------------------------
# The response `F` is the mean firing rate, the fraction of excited elements per step, against the stimulus rate `r`.
# Its dynamic range `10 log10(r_0.9 / r_0.1)` is the number of decibels of stimulus between 10% and 90% of the span of
# the response. Kinouchi and Copelli (2006) found that it peaks at the critical branching ratio 1: below it, weak
# stimuli are not amplified, and above it, self-sustained activity masks them. The curve at `sigma = 1` should rise
# from the weakest stimuli, and the dynamic range should peak at `sigma = 1`.

rates = np.logspace(-5, 1, 13)
sigmas = (0.6, 0.8, 1.0, 1.2, 1.4)
responses = {}
for sigma in sigmas:
    responses[sigma] = []
    for rate in rates:
        run = Simulation(create(MODEL, graph=graph, sigma=sigma, stimulus=rate), dt=1, random_state=0).run(600)
        responses[sigma].append((run.trajectory_['neurons.phase'][100:] == 1).mean())


def dynamic_range(response):
    """Return the dynamic range of a response curve in decibels."""
    response = np.maximum.accumulate(response)
    low, high = response[0], response[-1]
    r_10, r_90 = (np.exp(np.interp(low + share * (high - low), response, np.log(rates))) for share in (0.1, 0.9))
    return 10 * np.log10(r_90 / r_10)


fig, (curves, ranges) = plt.subplots(1, 2, figsize=(10, 4))
for sigma in (0.6, 1.0, 1.4):
    fired = np.array(responses[sigma]) > 0
    curves.loglog(rates[fired], np.array(responses[sigma])[fired], 'o-', markersize=3, label=f'sigma = {sigma}')
curves.set_xlabel('stimulus rate r (per step)')
curves.set_ylabel('response F (excited fraction per step)')
curves.set_title('Response curves')
curves.legend()
ranges.plot(sigmas, [dynamic_range(responses[sigma]) for sigma in sigmas], 'o-')
ranges.axvline(1.0, color='gray', linewidth=0.8, linestyle='--')
ranges.set_xlabel('branching ratio sigma')
ranges.set_ylabel('dynamic range (dB)')
ranges.set_title('The dynamic range peaks at sigma = 1')
fig.tight_layout()
plt.show()
