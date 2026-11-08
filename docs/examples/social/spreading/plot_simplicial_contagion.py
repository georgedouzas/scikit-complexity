"""
# Simplicial contagion

A susceptible node catches an infection from every infected neighbour, and also from every filled triangle whose two
other members are infected. The triangles reinforce the infection where it is already common, so above a strength of
the triangles the epidemic either dies out or settles at a high prevalence, depending on how many nodes it starts from.
"""

# %%
# Two hundred nodes form a random simplicial complex with 20 links and 6 triangles per node on average. The rescaled
# infectivity of the links, 0.75, lies below the epidemic threshold 1 of a network without triangles, and that of the
# triangles is 3. The run starts from 40% infected nodes, and the default blueprint shows the prevalence settling near
# the endemic value.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.spreading import SimplicialContagionSystem

SYSTEM = 'simplicial_contagion'

system = SimplicialContagionSystem(share=0.4, seed=0)
observables = system.get_run_settings()['observables']
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
trajectory = simulate(system, until=30, observables=observables, observers=[observer], random_state=0)

# %%
# Two stable prevalences
# ----------------------
# In the mean-field limit the prevalence follows `rho' = -rho + l rho (1 - rho) + l_T rho^2 (1 - rho)`, with the
# infectivities `l` of the links and `l_T` of the triangles. For `l_T = 3` its endemic branch exists from `l ~ 0.46`,
# below the threshold 1 where the healthy state loses stability, so the outcome depends on the initial share of
# infected nodes. Runs from 2% and from 40% infected nodes trace the two branches.


def prevalence(infectivity, share):
    """Return the mean share of infected nodes over the last ten time units of a run."""
    system_system = SimplicialContagionSystem(infectivity=infectivity, share=share, seed=0)
    run = simulate(system_system, until=30, random_state=0)
    infected = run.values('nodes.infected')
    return infected.reshape(len(infected), -1).mean(axis=1)[-20:].mean()


infectivities = np.linspace(0.3, 1.2, 7)
fig, ax = plt.subplots(figsize=(6, 4.5))
for share in (0.02, 0.4):
    ax.plot(infectivities, [prevalence(item, share) for item in infectivities], 'o', label=f'from {share:.0%} infected')
grid = np.linspace(0.3, 1.2, 200)
roots = [np.roots([-3.0, 3.0 - item, item - 1.0]) for item in grid]
for sign in (1, -1):
    branch = [np.max(root.real) if sign == 1 else np.min(root.real) for root in roots]
    real = [abs(root.imag).max() < 1e-9 for root in roots]
    shown = np.where(real & (np.asarray(branch) > 0), branch, np.nan)
    ax.plot(grid, shown, '-' if sign == 1 else ':', color='black')
ax.set_xlabel('link infectivity l')
ax.set_ylabel('prevalence')
ax.set_title('Triangles make the epidemic bistable')
ax.legend()
plt.show()
