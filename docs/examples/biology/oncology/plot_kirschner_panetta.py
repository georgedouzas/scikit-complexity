"""
# Kirschner-Panetta immunotherapy

Untreated, the tumour and the effector cells oscillate in long cycles in which the tumour nearly vanishes and regrows,
while adoptive therapy with IL-2 above the treatment levels of the book of Padmanabhan, Meskin, and Al Moustafa (2021)
clears the tumour.
"""

# %%
# The example builds `KirschnerPanettaSystem` with an antigenicity of 0.02, in the oscillatory range, and no treatment,
# from one unit of every population. The system is stiff, so it runs with LSODA. The observer saves every frame with the
# default blueprint, whose primary view is the time series of the effector cells, the tumour, and IL-2.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.oncology import KirschnerPanettaSystem

SYSTEM = 'kirschner_panetta'

solver = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-10)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(KirschnerPanettaSystem(alpha_a=0.02), dt=5.0, solver=solver, observers=[observer])

# %%
# Oscillations and clearance
# --------------------------
# The untreated tumour should rise to about `4 10^5` and fall by several orders of magnitude, again and again, with the
# effector cells lagging behind. With the adoptive therapy and the IL-2 injection 10% above the book's treatment
# factors, `u1 = u2 = 1.1`, the tumour should fall below one cell and stay there.

untreated = trajectory
treated = simulate(KirschnerPanettaSystem(alpha_a=0.02, u1=1.1, u2=1.1), dt=5.0, solver=solver)
fig, (cycles, therapy) = plt.subplots(1, 2, figsize=(10, 4))
cycles.semilogy(untreated.times, untreated.get_values('site.A'), label='tumour')
cycles.semilogy(untreated.times, untreated.get_values('site.E'), label='effector cells')
cycles.set_xlabel('time (days)')
cycles.set_ylabel('volume')
cycles.set_title('Untreated, tumour and effector cells oscillate')
cycles.legend()
therapy.semilogy(untreated.times, untreated.get_values('site.A'), label='untreated')
therapy.semilogy(treated.times, treated.get_values('site.A').clip(1e-12), label='u1 = u2 = 1.1')
therapy.axhline(1.0, color='grey', linestyle=':', label='one cell')
therapy.set_xlabel('time (days)')
therapy.set_ylabel('tumour volume')
therapy.set_title('Adoptive therapy with IL-2 clears the tumour')
therapy.legend()
fig.tight_layout()
plt.show()
