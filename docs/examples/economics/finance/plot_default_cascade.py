"""
# Default cascade

Banks lend to each other, and when one bank's losses exhaust its equity it defaults and its creditors write off their
claims on it, which can default them in turn. On random networks the cascades are large only in a window of
intermediate connectivity: sparse networks do not carry the shock far, and dense ones spread it too thin.
"""

# %%
# Two hundred and fifty banks of the network of Gai and Kapadia, with a fifth of their assets lent to four other
# banks on average and capital of 4 percent, see the external assets of one bank wiped out. The default blueprint shows
# the number of defaults round by round.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.economics.finance import DefaultCascadeSystem

SYSTEM = 'default_cascade'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulate(DefaultCascadeSystem(), observers=[observer])

# %%
# The window of contagion
# -----------------------
# Shocking one bank in each of ten networks per mean degree, the share of runs with a cascade above 5 percent of the
# banks and the mean extent of the cascades should be near zero below a degree of 1, large between about 2 and 6, and
# near zero again for dense networks, as in Fig. 1 of Gai and Kapadia (2010). A recovery rate of 0.5 shrinks the window.

degrees = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
fig, axis = plt.subplots(figsize=(6, 4))
for recovery in (0.0, 0.5):
    extents = np.array(
        [
            [
                simulate(
                    DefaultCascadeSystem(degree=degree, recovery=recovery, shocked=(seed,), random_state=seed),
                ).values('cascade.defaults')[-1]
                / 250
                for seed in range(10)
            ]
            for degree in degrees
        ],
    )
    axis.plot(degrees, (extents > 0.05).mean(axis=1), 'o-', label=f'frequency, R = {recovery}')
    axis.plot(degrees, extents.mean(axis=1), 's--', label=f'mean extent, R = {recovery}')
axis.set_xlabel('mean degree')
axis.set_ylabel('share')
axis.set_title('Contagion window')
axis.legend()
fig.tight_layout()
plt.show()
