"""
# Pomeau-Manneville intermittency map

The orbit creeps away from the marginal fixed point at 0 in long laminar phases, where `x` stays small for hundreds of
steps, interrupted by short chaotic bursts that reinject it near 0 at random. The laminar phases have no typical
length: their durations follow a power law.
"""

# %%
# The map runs with the tangency exponent `z = 2` for 2900 steps from `x = 0.01`. The default blueprint shows `x` as a
# time series, with the laminar plateaus near 0 and the bursts between them.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import ccdf, laminar_durations
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'intermittency_map'

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=10)
simulation = Simulation(create(MODEL), dt=1, observers=[observer]).run(until=2900)

# %%
# Durations of the laminar phases
# -------------------------------
# Near 0 the map is `x' ≈ x + x^z`, so an orbit reinjected uniformly at `x0` takes about `x0^(1 - z) / (z - 1)` steps to
# leave. The share of laminar phases, `x < 0.1`, that last at least `τ` steps should then fall as `τ^(-1 / (z - 1))`
# over 200000 steps: as `1 / τ` for `z = 2` and as `1 / τ²` for `z = 1.5`.

fig, (series, tails) = plt.subplots(1, 2, figsize=(11, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_['map.x'], linewidth=0.6)
series.set_xlabel('step')
series.set_ylabel('x')
series.set_title('Laminar phases and bursts, z = 2')
for z in (1.5, 2.0):
    values = Simulation(create(MODEL, z=z), dt=1).run(until=200_000).trajectory_['map.x']
    durations, share = ccdf(laminar_durations(values, threshold=0.1))
    (line,) = tails.loglog(durations, share, label=f'z = {z}')
    reference = durations[(durations > 20) & (durations < 2000)]
    anchor = share[np.searchsorted(durations, reference[0])]
    tails.loglog(
        reference,
        anchor * (reference / reference[0]) ** (-1 / (z - 1)),
        '--',
        color=line.get_color(),
        label=f'slope {-1 / (z - 1):.0f}',
    )
tails.set_xlabel(r'laminar duration $\tau$')
tails.set_ylabel(r'share of phases $\geq \tau$')
tails.set_title('Power-law laminar phases')
tails.legend()
fig.tight_layout()
plt.show()
