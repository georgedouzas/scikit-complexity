"""
# Active walkers

Walkers cross a green between its corners. They prefer existing trails to the direct way, and trails fade unless they
are used, so instead of a lawn worn evenly a few shared trails emerge that bundle the routes between the corners.
"""

# %%
# The model is created by name from the catalog with twenty walkers on a green of 30 by 30 cells. The observer saves
# every second step with the default blueprint, whose primary view is the image of the trails.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import fields_of, gini
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'active_walkers'

system = create(MODEL, size=30, n_walkers=20, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=300)

# %%
# Trails concentrate the wear
# ---------------------------
# If the walkers spread evenly the wear of the green would be uniform, with a Gini coefficient near zero. As the
# trails form and the walkers share them, the Gini coefficient of the trail field should settle well above that.

trajectory = simulation.trajectory_
trails = fields_of(trajectory)
concentration = [gini(frame.ravel()) for frame in trails]
fig, (image, inequality) = plt.subplots(1, 2, figsize=(11, 4.5))
image.imshow(trails[-1], cmap='Greens', origin='lower')
image.set_title(f'Trails at step {trajectory.times[-1]:.0f}')
image.set_xlabel('x')
image.set_ylabel('y')
inequality.plot(trajectory.times, concentration)
inequality.set_ylim(0, 1)
inequality.set_xlabel('step')
inequality.set_ylabel('Gini coefficient of the trails')
inequality.set_title(f'Wear stays concentrated: Gini {concentration[-1]:.2f}')
fig.tight_layout()
plt.show()
