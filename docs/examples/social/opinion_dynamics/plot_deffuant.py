"""
# Deffuant bounded confidence

Pairs of people meet at random and move their opinions towards each other, but only when the opinions already lie
within a confidence bound. Opinions that start spread uniformly settle into a few separate clusters, about
`1 / (2 d)` of them for the bound `d`, which no longer talk to each other.
"""

# %%
# Two hundred people start with uniform opinions and a confidence bound of 0.2. The system's blueprint shows the
# histogram of the opinions, which collapses into two or three peaks.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.experiments import calibrate
from skcomplex.systems.social.opinion_dynamics import DeffuantSystem

SYSTEM = 'deffuant'

system = DeffuantSystem(confidence=0.2, random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Number of clusters against the bound
# ------------------------------------
# Opinions within 0.01 of each other at the end belong to one cluster. Clusters of fewer than five people are left out,
# since a few stragglers stay between the large clusters. The number of large clusters should follow `1 / (2 d)`,
# rounded down, as the bound shrinks.


def clusters(confidence, seed):
    """Return the number of clusters of at least five people at the end of a run."""
    run = simulate(DeffuantSystem(confidence=confidence, random_state=seed), random_state=seed)
    opinions = np.sort(run.get_agents('people', len(run.times) - 1).values['opinion'])
    groups = np.split(opinions, np.flatnonzero(np.diff(opinions) > 0.01) + 1)
    return sum(len(group) >= 5 for group in groups)


bounds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
counts = np.array([[clusters(bound, seed) for seed in range(3)] for bound in bounds])
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(bounds, counts.mean(axis=1), yerr=counts.std(axis=1), marker='o', capsize=3, label='simulated')
grid = np.linspace(0.08, 0.42, 200)
ax.plot(grid, np.floor(1 / (2 * grid)), color='black', linestyle='--', label='floor(1 / (2 d))')
ax.set_xlabel('confidence bound d')
ax.set_ylabel('clusters of at least five people')
ax.set_title('Narrow bounds split opinions into more clusters')
ax.legend()
plt.show()

# %%
# Calibrating the bound from the encounters
# -----------------------------------------
# With a steepness, every encounter compromises with the probability `1 / (1 + exp(-rho (d - |x_u - x_v|)))` of
# Lenti et al., drawn by a named draw that a run records. Replaying the recorded outcomes under another bound scores
# them by their exact log-probability, so `calibrate` with `loss='events'` fits the bound to one run of 100 people
# over 50 steps. The log-likelihood should peak at the true bound 0.25, and the estimate should lie within 0.02 of it.

recorded = simulate(DeffuantSystem(100, confidence=0.25, steepness=50.0), until=50, random_state=0, record_draws=True)
fit = calibrate(
    DeffuantSystem(100, steepness=50.0),
    recorded,
    {'confidence': (0.05, 0.5)},
    loss='events',
    random_state=0,
)
grid = np.linspace(0.15, 0.35, 21)
scores = [
    simulate(
        DeffuantSystem(100, confidence=bound, steepness=50.0),
        until=50,
        random_state=0,
        replay=recorded,
    ).log_probability
    for bound in grid
]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, scores, marker='o', label='log-likelihood of the encounters')
ax.axvline(0.25, color='black', linestyle='--', label='true bound')
ax.axvline(fit.estimates['confidence'], color='tab:red', label=f'estimate {fit.estimates["confidence"]:.3f}')
ax.set_xlabel('confidence bound d')
ax.set_ylabel('log-likelihood')
ax.set_title('The recorded encounters pin down the bound')
ax.legend()
plt.show()
