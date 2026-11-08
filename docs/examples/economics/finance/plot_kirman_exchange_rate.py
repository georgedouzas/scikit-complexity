"""
# Kirman exchange rate

Traders in a foreign exchange market are fundamentalists, who expect the rate to return to its fundamental value, or
chartists, who expect its last change to go on. Like Kirman's ants choosing between two food sources, they switch by
meeting and recruiting one another, so the market spends long periods dominated by one kind and the volatility of the
rate comes and goes with the chartists.
"""

# %%
# Thirty traders start evenly split, with `v = 0.5`, `g = 1`, `r = 0.01`, and small supply shocks, for 20000 steps.
# The default blueprint shows the deviation of the rate and the number of fundamentalists.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.finance import KirmanExchangeRateSystem

SYSTEM = 'kirman_exchange_rate'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
trajectory = simulate(KirmanExchangeRateSystem(), observers=[observer])

# %%
# Regimes and volatility
# ----------------------
# Recruitment dominates spontaneous switches, `epsilon < (1 - delta) / (N - 1)`, so the share of fundamentalists
# should pile up near 0 and 1. The changes of the deviation should be larger when chartists dominate, since their
# eigenvalues lie close to the unit circle (Eq. 1.38) while fundamentalists contract the deviation (Eq. 1.37).

share, deviation = trajectory.values('market.fundamentalists'), trajectory.values('market.x')
fig, (path, histogram) = plt.subplots(1, 2, figsize=(10, 4))
path.plot(trajectory.times, deviation, lw=0.5, label='deviation x')
path.plot(trajectory.times, share, lw=0.8, label='share of fundamentalists')
path.set_xlabel('step')
path.set_title('Deviation of the rate and the regimes')
path.legend()
changes = np.abs(np.diff(deviation))
histogram.hist(share, bins=20, density=True)
histogram.set_xlabel('share of fundamentalists')
histogram.set_ylabel('density')
ratio = changes[share[1:] < 0.2].mean() / changes[share[1:] > 0.8].mean()
histogram.set_title(f'Bimodal share, volatility ratio {ratio:.1f}')
fig.tight_layout()
plt.show()
