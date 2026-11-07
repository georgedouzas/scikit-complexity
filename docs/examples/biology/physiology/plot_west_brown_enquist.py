"""
# West-Brown-Enquist

A space-filling distribution network of area-preserving vessels ends in capillaries of one size in every organism. The
number of capillaries, the metabolic rate, grows as the blood volume, the mass, to the power 3/4, Kleiber's law.
"""

# %%
# The static system is solved once for a tree of six branchings, and the observer saves its one frame with the default
# blueprint, whose primary view is the tree with the radii of its vessels.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import WestBrownEnquistSystem

SYSTEM = 'west_brown_enquist'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(WestBrownEnquistSystem(), until=1.0, observers=[observer])

# %%
# Kleiber's law
# -------------
# Trees of growing depth should give metabolic rates on a power law of the mass whose exponent rises towards 3/4.

depths = [2, 3, 4, 5, 6]
volumes, capillaries = [], []
for depth in depths:
    run = simulate(WestBrownEnquistSystem(depth=depth), until=1.0)
    volumes.append(run.values('blood_volume')[-1])
    capillaries.append(run.values('capillaries')[-1])
exponents = np.diff(np.log(capillaries)) / np.diff(np.log(volumes))
fig, (law, slope) = plt.subplots(1, 2, figsize=(11, 4.5))
law.loglog(volumes, capillaries, 'o-', label='trees')
law.loglog(volumes, capillaries[-1] * (np.array(volumes) / volumes[-1]) ** 0.75, '--', label='slope 3/4')
law.set_xlabel('blood volume (mass)')
law.set_ylabel('capillaries (metabolic rate)')
law.set_title('Metabolic rate against mass')
law.legend()
slope.plot(depths[1:], exponents, 'o-')
slope.axhline(0.75, color='black', linestyle='--')
slope.set_xlabel('depth of the tree')
slope.set_ylabel('local exponent')
slope.set_title('The exponent approaches 3/4 from below')
fig.tight_layout()
plt.show()
