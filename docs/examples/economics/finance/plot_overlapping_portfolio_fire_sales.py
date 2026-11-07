"""
# Fire sales of overlapping portfolios

Banks that invest in common assets are linked even without lending to each other. When one asset loses value, the
banks that fail or deleverage sell their holdings, the sales depress prices, and every bank holding those assets loses
in turn. The stronger the market impact, the further the losses travel.
"""

# %%
# A hundred banks with leverage 20 hold ten of a hundred assets each, and the first asset loses 60 percent of its
# price, with a market impact `alpha = 1` and banks that liquidate when their equity is gone. The default blueprint
# shows the defaults and the value sold round by round.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'overlapping_portfolio_fire_sales'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
create_simulation(SYSTEM, shock=0.6).set_params(observers=[observer]).prepare().run(until=30)

# %%
# Market impact and the size of the shock
# ---------------------------------------
# Without market impact only the holders of the shocked asset should default. As the impact grows, the cascade should
# reach banks that never held the shocked asset, and with leverage targeting banks should sell, and prices fall, even
# after shocks that passive banks absorb.

impacts = np.linspace(0.0, 1.0, 6)
shocks = [0.2, 0.4, 0.6]
fig, axis = plt.subplots(figsize=(6, 4))
for shock in shocks:
    defaults = [
        create_simulation(SYSTEM, shock=shock, alpha=alpha)
        .prepare()
        .run(until=30)
        .trajectory_.values('sales.defaults')[-1]
        for alpha in impacts
    ]
    axis.plot(impacts, defaults, 'o-', label=f'shock {shock}, thresholds')
defaults = [
    create_simulation(SYSTEM, shock=0.2, alpha=alpha, selling='leverage')
    .prepare()
    .run(until=30)
    .trajectory_.values(
        'sales.defaults',
    )[-1]
    for alpha in impacts
]
axis.plot(impacts, defaults, 's--', label='shock 0.2, leverage targeting')
axis.set_xlabel('market impact alpha')
axis.set_ylabel('defaulted banks')
axis.set_title('Cascades of fire sales')
axis.legend()
fig.tight_layout()
plt.show()
