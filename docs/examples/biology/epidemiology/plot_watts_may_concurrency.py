"""
# Watts-May concurrency

An infection that spreads along sexual partnerships passes not only to new partners but also to current partners of
someone who turns infectious during the partnership. Partnerships that last longer give everyone more current
partners, so the same partner and infection rates spread the epidemic faster.
"""

# %%
# The example builds `WattsMayConcurrencySystem` with partnerships of mean duration 3, and the observer saves every
# other frame with the default blueprint, whose primary view is the time series of the compartments.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.epidemiology import WattsMayConcurrencySystem

SYSTEM = 'watts_may_concurrency'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(WattsMayConcurrencySystem(tau=3.0), observers=[observer])

# %%
# Longer partnerships spread faster
# ---------------------------------
# The infection rate of Watts and May grows with the mean duration `tau` of partnerships, so the infectious fraction
# should rise earlier and peak higher for longer partnerships.

fig, ax = plt.subplots(figsize=(7, 4.5))
for tau in (0.25, 1.0, 3.0):
    run = simulate(WattsMayConcurrencySystem(tau=tau))
    ax.plot(run.times, run.get_values('population.I'), label=f'tau = {tau}')
ax.set_xlabel('time (years)')
ax.set_ylabel('infectious fraction I')
ax.set_title('The epidemic grows faster with longer partnerships')
ax.legend()
plt.show()
