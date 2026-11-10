"""
# Creative phase transition

Hanel, Kauffman, and Thurner's system of combinatorial innovation: existing goods combine in pairs to produce new
ones, and every new good opens new combinations, the adjacent possible. Below a critical density of productive rules
the economy stays sparse, while above it a creative explosion fills almost the whole space of possible goods. The
final diversity jumps discontinuously as the rule density crosses the critical line.
"""

# %%
# The economy starts with 10 percent of the possible goods and the rule density `r = 5`, above the critical density.
# The blueprint shows the diversity `x` and the newly produced share `d`, which stays small for a few steps and then
# bursts as the diversity takes off to fill the space.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.innovation import CreativePhaseTransitionSystem

SYSTEM = 'creative_phase_transition'
UNTIL = 50

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(CreativePhaseTransitionSystem(x=0.1, r=5.0), until=20, observers=[observer])

# %%
# Phase diagram
# -------------
# Iterating `x' = x + d`, `d' = r (1 - x') (x'^2 - x^2)` from `d_0 = r x_0^2 (1 - x_0)`, with no step producing more
# than the goods that are missing, gives the final diversity on a fine grid of rule densities. The simulated final
# diversity should lie on these curves, which jump from a sparse to an almost full economy at a critical density that
# falls as the initial diversity grows.


def final_diversity(x, r, steps=UNTIL):
    """Return the final diversity of the map iterated directly."""
    d = r * x**2 * (1 - x)
    for _ in range(steps):
        new = min(x + d, 1.0)
        d, x = min(r * (1 - new) * (new**2 - x**2), 1.0 - new), new
    return x


densities = np.linspace(0.5, 12.0, 400)
coarse = np.linspace(1.0, 12.0, 12)
fig, (series, diagram) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(trajectory.times, trajectory.get_values('system.x'), 'o-', ms=3, label='diversity x')
series.plot(trajectory.times, trajectory.get_values('system.d'), 's-', ms=3, label='new share d')
series.set_xlabel('step')
series.set_ylabel('share of possible goods')
series.set_title('Creative explosion for r = 5')
series.legend()
for index, start in enumerate((0.05, 0.1, 0.2)):
    diagram.plot(densities, [final_diversity(start, r) for r in densities], color=f'C{index}', label=f'x0 = {start}')
    finals = [
        simulate(CreativePhaseTransitionSystem(x=start, r=r), until=UNTIL).get_values('system.x')[-1] for r in coarse
    ]
    diagram.plot(coarse, finals, 'o', color=f'C{index}')
diagram.set_xlabel('rule density r')
diagram.set_ylabel('final diversity')
diagram.set_title('Map (lines) and simulation (dots)')
diagram.legend()
fig.tight_layout()
plt.show()
