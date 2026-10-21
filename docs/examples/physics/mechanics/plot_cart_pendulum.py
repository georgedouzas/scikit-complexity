"""
# Cart pendulum

A pendulum hanging from a cart that rolls freely on a track: as the bob swings one way the cart rolls the other, so
their common centre of mass never moves, and the pair swings faster than a pendulum on a fixed pivot.
"""

# %%
# Cart and bob have equal masses, the rod has length 1, and the bob starts at 0.5 radians at rest. The default
# blueprint draws the cart, the rod, and the bob, which swing in opposition.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'cart_pendulum'


system = create(MODEL, theta=0.5, y=0.0, m1=1.0, m2=1.0, l=1.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.02, observers=[observer]).run(until=10)

# %%
# Fixed centre of mass and a faster swing
# ---------------------------------------
# No horizontal force acts on the pair, so the centre of mass `(m1 y + m2 (y + l sin θ)) / (m1 + m2)` should stay at
# its initial value. For small swings the angle oscillates at `sqrt(g (m1 + m2) / (m1 l))`, `sqrt(2)` times faster than
# on a fixed pivot, so the measured frequency of runs with a lighter and lighter cart should follow that curve.

times = np.asarray(simulation.trajectory_.times)
cart = np.ravel(simulation.trajectory_['cart.y'])
bob = cart + np.sin(np.ravel(simulation.trajectory_['cart.theta']))
ratios = np.linspace(0.25, 4.0, 10)
frequencies = []
for ratio in ratios:
    run = Simulation(create(MODEL, theta=0.05, m1=ratio, m2=1.0, l=1.0), dt=0.01).run(until=20)
    velocity, steps = np.ravel(run.trajectory_['cart.theta_d1']), np.asarray(run.trajectory_.times)
    crossings = steps[1:][(velocity[:-1] < 0) & (velocity[1:] >= 0)]
    frequencies.append(2 * np.pi / np.diff(crossings).mean())
grid = np.linspace(0.2, 4.0, 200)
fig, (centre, swing) = plt.subplots(1, 2, figsize=(10, 4))
centre.plot(times, cart, label='cart')
centre.plot(times, bob, label='bob')
centre.plot(times, (cart + bob) / 2, color='black', label='centre of mass')
centre.set_xlabel('time')
centre.set_ylabel('horizontal position')
centre.set_title('The centre of mass stays put')
centre.set_ylim(-0.1, 0.7)
centre.legend(loc='upper center', ncol=3)
swing.plot(grid, np.sqrt(9.81 * (grid + 1) / grid), color='black', label='sqrt(g (m1 + m2) / (m1 l))')
swing.plot(ratios, frequencies, 'o', label='simulated')
swing.set_xlabel('cart mass m1 (bob mass 1)')
swing.set_ylabel('angular frequency of small swings')
swing.set_title('A lighter cart swings faster')
swing.legend()
fig.tight_layout()
plt.show()
