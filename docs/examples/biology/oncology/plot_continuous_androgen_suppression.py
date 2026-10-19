"""
# Prostate cancer under continuous androgen suppression

The suppression lowers the serum androgen, and the androgen-dependent cancer cells first grow and then decline, while
the androgen-independent cells grow throughout and cause the relapse.
"""

# %%
# The model is created by name from the catalog: ten dependent cells and 0.14 independent cells under a suppression
# of efficacy 0.9. The observer saves every frame with the default blueprint, whose primary view is the time series of
# the cells and the androgen.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'continuous_androgen_suppression'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=5.0, observers=[observer]).run(until=900)

# %%
# Androgen and relapse
# --------------------
# The serum androgen follows `U(t) = U0 (1 - a) + U0 a exp(-du t)` and settles at 2 ng/mL. The dependent cells grow
# while the androgen is still high and decline once it is low, and the independent cells overtake them.

trajectory = simulation.trajectory_
times = trajectory.times
fig, (androgen, cells) = plt.subplots(1, 2, figsize=(10, 4))
androgen.plot(times, trajectory['tumour.U'], label='simulated')
androgen.plot(times, 20.0 * 0.1 + 20.0 * 0.9 * np.exp(-0.08 * times), '--', color='black', label='closed form')
androgen.set_xlabel('time (days)')
androgen.set_ylabel('serum androgen (ng/mL)')
androgen.set_title('The suppression lowers the androgen')
androgen.legend()
cells.semilogy(times, trajectory['tumour.AD'], label='androgen-dependent')
cells.semilogy(times, trajectory['tumour.AI'], label='androgen-independent')
cells.set_xlabel('time (days)')
cells.set_ylabel('cells')
cells.set_title('Independent cells take over')
cells.legend()
fig.tight_layout()
plt.show()
