"""
# Brothel movement

Trafficking victims are held in a few brothels and moved between them every month, while free women travel on their
own. The crowded brothels leave a trace that investigators can look for: their quarterly energy bills stand far above
those of ordinary houses.
"""

# %%
# The model is created by name from the catalog on a 30 by 30 grid of three cities with 40 houses, 3 brothels, 12
# victims, and 30 free women, and runs for one year, one step per day. The default blueprint shows the premises and the
# women, with brothels and victims bright and houses and free women dark: the bright points are the few brothel cells
# where all the victims are held.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'trafficking'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=360)

# %%
# Traces in the bills
# -------------------
# The bills of a premises grow with the number of people inside, so after every quarter the recorded bills of the three
# brothels should stand well above those of the houses, where at most a few free women stay.

trajectory = simulation.trajectory_
quarters = [90, 180, 270, 360]
fig, ax = plt.subplots(figsize=(7, 4.5))
for index, day in enumerate(quarters):
    frame = int(np.argmin(np.abs(trajectory.times - day)))
    premises = trajectory.agents('premises', frame).values
    brothel = np.asarray(premises['brothel']) > 0
    bills = np.asarray(premises['last_bills'])
    jitter = np.random.default_rng(index).uniform(-0.15, 0.15, len(bills))
    ax.plot(index + jitter[~brothel], bills[~brothel], '.', color='C0', label='houses' if index == 0 else None)
    ax.plot(index + jitter[brothel], bills[brothel], 'o', color='C3', label='brothels' if index == 0 else None)
ax.set_xticks(range(len(quarters)), labels=[f'quarter {quarter // 90}' for quarter in quarters])
ax.set_ylabel('energy bills of the last quarter')
ax.set_title('Brothels stand out in the energy bills')
ax.legend(loc='center right')
plt.show()
