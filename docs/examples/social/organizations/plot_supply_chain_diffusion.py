"""
# Namrata's Family

Chesney's model of the diffusion of good working conditions through a garment supply network. A buyer can force a
supplier to adopt them when the supplier needs the buyer more than the buyer needs the supplier, measured by the share
of each one's business that flows between them. Starting from a seed group of buyers, adoption spreads down the
network until it meets suppliers powerful enough to resist.
"""

# %%
# Three buyers and 40 factories form a random supply network, and two buyers adopt at the start. The default
# blueprint shows the adoption of every firm on the network.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.organizations import SupplyChainDiffusionSystem

SYSTEM = 'supply_chain_diffusion'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
trajectory = simulate(SupplyChainDiffusionSystem(), observers=[observer], random_state=0)

# %%
# Adoption over time and who resists
# ----------------------------------
# Adoption should spread one tier of suppliers per period and stop. A supplier resists when its outflow, the size of
# its business, is at least the inflow of each of its adopting buyers, so the firms that never adopt should be the
# larger suppliers.

adopted = [trajectory.get_agents('firms', frame).values['adopted'].sum() for frame in range(len(trajectory.times))]
final = trajectory.get_agents('firms', len(trajectory.times) - 1).values
fig, (spread, sizes) = plt.subplots(1, 2, figsize=(10, 4))
spread.step(trajectory.times, adopted, where='post')
spread.set_xlabel('time period')
spread.set_ylabel('firms with good conditions')
spread.set_title('Diffusion from the seed group')
factories = np.arange(final['adopted'].size) >= 3
for value, name in ((1.0, 'adopted'), (0.0, 'resisted')):
    chosen = factories & (final['adopted'] == value)
    sizes.hist(final['outflow'][chosen], bins=10, alpha=0.6, label=name)
sizes.set_xlabel('outflow of the factory')
sizes.set_ylabel('factories')
sizes.set_title('Larger factories resist')
sizes.legend()
fig.tight_layout()
plt.show()
