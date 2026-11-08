"""
# Majority rule with willingness

An agent consults a few random others and adopts their majority opinion with the probability of its willingness.
Willing agents reach consensus, zealots of willingness zero holding both opinions prevent it, and a small group of
zealots of one opinion wins everyone over.
"""

# %%
# Two hundred agents consult groups of three, and a tenth of them are zealots of opinion `A`. The default blueprint
# shows the opinions.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.opinion_dynamics import WillingMajoritySystem

SYSTEM = 'willing_majority'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default')
system = WillingMajoritySystem(zealots=0.1, zealot_opinion=1.0)
trajectory = simulate(system, observers=[observer], random_state=0)

# %%
# Consensus and zealots
# ---------------------
# The share of opinion `A` should reach 0 or 1 when every agent is willing, stay between them with zealots of both
# opinions, and reach 1 with the zealots of `A`.

runs = {
    'all willing': WillingMajoritySystem(),
    'zealots of both opinions': WillingMajoritySystem(zealots=0.2),
    'zealots of A': system,
}
fig, (shares, groups) = plt.subplots(1, 2, figsize=(10, 4))
for name, item in runs.items():
    trajectory = simulate(item, until=50, random_state=0)
    share = [np.mean(trajectory.agents('voters', frame).values['opinion']) for frame in range(51)]
    shares.plot(trajectory.times, share, label=name)
shares.set_xlabel('sweeps')
shares.set_ylabel('share of opinion A')
shares.set_title('Who reaches consensus')
shares.legend()
fractions = np.linspace(0.0, 0.4, 9)
finals = []
for fraction in fractions:
    trajectory = simulate(WillingMajoritySystem(zealots=fraction), random_state=0)
    opinions = trajectory.agents('voters', 50).values['opinion']
    finals.append(min(opinions.mean(), 1 - opinions.mean()))
groups.plot(fractions, finals, 'o-')
groups.set_xlabel('fraction of zealots f')
groups.set_ylabel('share of the minority opinion')
groups.set_title('Zealots of both opinions keep a minority')
fig.tight_layout()
plt.show()
