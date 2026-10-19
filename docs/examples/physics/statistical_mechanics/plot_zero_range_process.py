"""
# Zero-range process

Above the critical density `1 / (b - 2)`, the particles of a zero-range process on a ring gather on one site, a
condensate that holds the excess particles while the other sites keep a fluid at the critical density. The space-time
image shows the occupation of the 50 sites over the last 64 frames, one row per frame with the earliest at the top:
one bright column, the condensate, holds about 125 particles while the other sites hold one or two.
"""

# %%
# The model is created by name from the catalog on 50 sites at density 3 with `u(n) = 1 + 4 / n`, so the critical
# density is 0.5 and the condensate should hold about `50 (3 - 0.5) = 125` particles. The observer saves every frame
# with the default blueprint, whose primary view is the space-time image of the occupations. The Numba backend runs
# the hops in a compiled loop.

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'zero_range_process'

system = create(MODEL, n_sites=50, density=3.0, b=4.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=12.0, observers=[observer], random_state=0, backend='numba').run(until=3000)

# %%
# Condensate and fluid
# --------------------
# The stationary weights of one site are `f(n) = prod_(k <= n) 1 / u(k) = n! b! / (n + b)!`, which decay as `n^-b`.
# At the critical point the fluid sites follow `p(n) = f(n) / sum_n f(n)`, of mean `1 / (b - 2)`. The occupations of
# all sites but the largest, after the condensate forms, should follow this law, and the largest occupation should
# approach `L (rho - rho_c) = 125`.

occupations = simulation.trajectory_['sites.n']
settled = np.sort(occupations[100:], axis=1)[:, :-1].ravel()
counts = np.arange(0, 12)
weights = np.exp(gammaln(counts + 1) + gammaln(5) - gammaln(counts + 5))
measured = np.array([(settled == count).mean() for count in counts])
fig, (growth, fluid) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(simulation.trajectory_.times, occupations.max(axis=1), label='largest site')
growth.axhline(125, color='black', linestyle='--', label='L (rho - rho_c)')
growth.set_xlabel('time')
growth.set_ylabel('particles')
growth.set_title('Growth of the condensate')
growth.legend()
fluid.semilogy(counts, measured, 'o', label='fluid sites')
fluid.semilogy(counts, weights / weights.sum(), color='black', label='f(n) / sum f')
fluid.set_xlabel('occupation n')
fluid.set_ylabel('probability')
fluid.set_title('The fluid stays at the critical density')
fluid.legend()
fig.tight_layout()
plt.show()
