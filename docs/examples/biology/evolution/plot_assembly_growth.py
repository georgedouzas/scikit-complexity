"""
# Assembly growth

Objects of each assembly index are discovered by joining objects one step below them. Without selection every object
ever built is reused and the deeper levels fill exponentially fast, while selection, reusing only a few objects, slows
the discovery of new objects.
"""

# %%
# The system is built from its class with ten assembly levels and no selection, and the observer saves every frame
# with the default blueprint, whose primary view is the time series of the levels.

import math

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import AssemblyGrowthSystem

SYSTEM = 'assembly_growth'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(AssemblyGrowthSystem(), observers=[observer])

# %%
# Selection slows discovery
# -------------------------
# Without selection the levels should follow `N_a = (k t)^a / a!`, and selection, `alpha < 1`, should leave the deep
# levels far emptier.

fig, ax = plt.subplots(figsize=(8, 4.5))
for alpha in (1.0, 0.7, 0.4):
    run = simulate(AssemblyGrowthSystem(alpha=alpha))
    ax.semilogy(range(1, 11), run.values('levels.N')[-1], 'o-', label=f'alpha = {alpha}')
ax.semilogy(range(1, 11), [10**a / math.factorial(a) for a in range(1, 11)], 'k--', label='(k t)^a / a!')
ax.set_xlabel('assembly index a')
ax.set_ylabel('objects N_a at t = 10')
ax.set_title('Selection slows the expansion of the assembly space')
ax.legend()
plt.show()
