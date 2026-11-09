"""
# Generalized Lotka-Volterra microbial community

Three microbial species that compete weakly and limit themselves grow from low abundances and coexist at the fixed
point of their interaction matrix.
"""

# %%
# The system is built from its class, three species of the group `species` that start at 0.1. The interaction matrix
# is read through a total over the group weighted by its rows. The observer saves every second frame of the 30 time
# units with the default blueprint, whose primary view is the time series of the abundances.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.ecology import GeneralizedLotkaVolterraSystem

SYSTEM = 'generalized_lotka_volterra'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(GeneralizedLotkaVolterraSystem(), observers=[observer])

# %%
# Coexistence at the interior fixed point
# ---------------------------------------
# The fixed point with every species present solves `b + a x = 0`, so `x* = -a^-1 b`. The abundances should approach
# the dashed lines, and the distance to the fixed point should fall exponentially.

b = np.array([1.0, 0.8, 0.6])
a = np.array([[-1.0, -0.3, 0.2], [-0.2, -1.0, -0.3], [0.1, -0.2, -1.0]])
fixed = -np.linalg.solve(a, b)
abundances = trajectory.get_values('species.x')
fig, (series, distance) = plt.subplots(1, 2, figsize=(10, 4))
for index in range(3):
    line = series.plot(trajectory.times, abundances[:, index], label=f'species {index + 1}')[0]
    series.axhline(fixed[index], color=line.get_color(), linestyle='--')
series.set_xlabel('time')
series.set_ylabel('abundance')
series.set_title('The species settle at x* = -a⁻¹ b')
series.legend()
distance.semilogy(trajectory.times, np.linalg.norm(abundances - fixed, axis=1))
distance.set_xlabel('time')
distance.set_ylabel('distance to x*')
distance.set_title('The approach is exponential')
fig.tight_layout()
plt.show()
