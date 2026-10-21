"""
# Mackey-Glass

Blood cells are produced with a delay after the signal that asks for them. With a delay of 17 days the delayed
feedback makes the concentration oscillate irregularly, a chaotic signal that never repeats.
"""

# %%
# The model is created by name from the catalog with the delay `tau = 17` and run for 500 days. The observer saves
# every fourth frame with the default blueprint, whose primary view is the time series of the concentration.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import delay_embedding
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'mackey_glass'

system = create(MODEL, tau=17.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.5, observers=[observer]).run(until=500)

# %%
# Periodic against chaotic
# ------------------------
# Plotting `x(t)` against `x(t - tau)` reconstructs the attractor. With `tau = 6` it should be one closed loop, a
# periodic orbit, while with `tau = 17` it should be a band of loops that never close, a strange attractor.

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, tau in zip(axes, [6.0, 17.0], strict=True):
    trajectory = Simulation(create(MODEL, tau=tau), dt=0.5).run(until=1000).trajectory_
    values = trajectory['cells.x'][trajectory.times > 200]
    vectors = delay_embedding(values, dimension=2, delay=int(tau / 0.5))
    ax.plot(vectors[:, 1], vectors[:, 0], linewidth=0.6)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t - tau)')
    ax.set_title(f'tau = {tau:.0f}: {"periodic" if tau < 10 else "chaotic"}')
fig.tight_layout()
plt.show()
