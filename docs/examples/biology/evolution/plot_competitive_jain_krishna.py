"""
# Competitive Jain-Krishna

Species with signed catalytic effects on each other compete for a fixed total population, and the least populated is
replaced every step by a random newcomer. Selection keeps the species that help each other, so the links of the
ecosystem turn from half positive to mostly positive, the emergence of cooperation.
"""

# %%
# The example builds `CompetitiveJainKrishnaSystem` with fifty species, and the observer saves every other step with the
# default blueprint, whose primary view is the count of species.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.evolution import CompetitiveJainKrishnaSystem

SYSTEM = 'competitive_jain_krishna'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(CompetitiveJainKrishnaSystem(), observers=[observer], random_state=0)

# %%
# Emergence of cooperation
# ------------------------
# The share of positive links between distinct species should rise well above its initial value of about one half.


def positive_share(links):
    matrix = np.array(links, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return (matrix[matrix != 0] > 0).mean()


frames = range(0, 401, 5)
shares = [positive_share(trajectory.get_agents('species', frame).values['links']) for frame in frames]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(list(frames), shares)
ax.axhline(0.5, color='gray', linestyle='--')
ax.set_xlabel('replacement step')
ax.set_ylabel('share of positive links')
ax.set_title('Selection favours cooperating species')
plt.show()
