"""
# Mackey-Glass

Blood cells are produced with a delay after the signal that asks for them. With a delay of 17 days the delayed
feedback makes the concentration oscillate irregularly, a chaotic signal that never repeats.
"""

# %%
# The example builds `MackeyGlassSystem` with the delay `tau = 17` and runs it for 500 days. The observer saves every
# fourth frame with the default blueprint, whose primary view is the time series of the concentration.

import matplotlib.pyplot as plt

from skcomplex.core.analysis import embedding
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import MackeyGlassSystem

SYSTEM = 'mackey_glass'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(MackeyGlassSystem(tau=17.0), observers=[observer])

# %%
# Periodic against chaotic
# ------------------------
# Plotting `x(t)` against `x(t - tau)` reconstructs the attractor. With `tau = 6` it should be one closed loop, a
# periodic orbit, while with `tau = 17` it should be a band of loops that never close, a strange attractor.

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, tau in zip(axes, [6.0, 17.0], strict=True):
    trajectory = simulate(MackeyGlassSystem(tau=tau), until=1000)
    values = trajectory.get_values('cells.x')[trajectory.times > 200]
    vectors = embedding(values, delay=int(tau / 0.5), dimension=2).points
    ax.plot(vectors[:, 1], vectors[:, 0], linewidth=0.6)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('x(t - tau)')
    ax.set_title(f'tau = {tau:.0f}: {"periodic" if tau < 10 else "chaotic"}')
fig.tight_layout()
plt.show()
