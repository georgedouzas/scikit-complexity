"""
# SIS on hypergraphs with heterogeneous exposure

Participants meet in environments, the hyperedges of a random hypergraph, and an environment gives a susceptible member
a dose that grows with the square of the share of its infected members. With this supralinear kernel the healthy state
stays stable, so a small seed dies out while a large one settles at a high prevalence.
"""

# %%
# A thousand participants share environments of 10, 20, and 30 members, five per participant on average. The run starts
# from 30% infected participants, above the unstable prevalence, and the default blueprint shows the prevalence.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.spreading import HypergraphExposureSisSystem

SYSTEM = 'hypergraph_exposure_sis'
HYPERDEGREE, RATE, RECOVERY = 5.0, 0.06, 0.05

system = HypergraphExposureSisSystem(share=0.3, seed=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
trajectory = simulate(system, until=200, observers=[observer], random_state=0)

# %%
# Two stable prevalences
# ----------------------
# In mean field the prevalence `I` is endemic where `k rate I (1 - I) = recovery`, for `k` environments per participant.
# With these parameters the equation has two roots: the higher is stable and the lower separates the runs that die out
# from those that persist.


def prevalence(share):
    """Return the share of infected participants after every step of a run."""
    run = simulate(HypergraphExposureSisSystem(share=share, seed=0), random_state=0)
    infected = run.values('participants.infected')
    return infected.reshape(len(infected), -1).mean(axis=1)


roots = np.roots([1.0, -1.0, RECOVERY / (HYPERDEGREE * RATE)]).real
fig, ax = plt.subplots(figsize=(6, 4))
for share in (0.05, 0.1, 0.3, 0.6):
    ax.plot(prevalence(share), label=f'from {share:.0%} infected')
ax.axhline(roots.max(), color='black', label='stable mean-field root')
ax.axhline(roots.min(), color='black', linestyle=':', label='unstable mean-field root')
ax.set_xlabel('step')
ax.set_ylabel('prevalence')
ax.set_title('A supralinear kernel makes the epidemic bistable')
ax.legend()
plt.show()
