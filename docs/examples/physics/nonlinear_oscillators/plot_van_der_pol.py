"""
# Van der Pol

The oscillator pumps energy in at small amplitude and dissipates it at large amplitude, so every start ends on the same
limit cycle: a self-sustained oscillation of amplitude close to 2 that does not depend on the initial condition.
"""

# %%
# With `μ = 3` the oscillation is a relaxation oscillation, slow drifts broken by fast jumps. The mass starts close to
# the unstable rest point, and the default blueprint shows the position and the velocity growing onto the cycle.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.nonlinear_oscillators import VanDerPolSystem

SYSTEM = 'van_der_pol'

system = VanDerPolSystem(x=0.1, v=0.0, mu=3.0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(system, observers=[observer])

# %%
# Attraction to the limit cycle
# -----------------------------
# Orbits started inside and outside the cycle should spiral onto the same closed curve in the phase plane, whose
# largest displacement is close to 2 for every `μ`.

fig, ax = plt.subplots(figsize=(6, 4))
for x, v in [(0.1, 0.0), (4.0, 6.0), (-3.0, -8.0)]:
    run = simulate(VanDerPolSystem(x=x, v=v, mu=3.0), until=30, dt=0.02)
    ax.plot(
        run.get_values('oscillator.x'),
        run.get_values('oscillator.x_d1'),
        linewidth=1,
        label=f'x0 = {x}',
    )
ax.set_xlabel('x')
ax.set_ylabel("x'")
ax.set_title('Three starts, one limit cycle (μ = 3)')
ax.legend()
plt.show()
