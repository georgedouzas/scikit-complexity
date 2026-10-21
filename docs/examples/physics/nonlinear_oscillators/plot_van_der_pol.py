"""
# Van der Pol

The oscillator pumps energy in at small amplitude and dissipates it at large amplitude, so every start ends on the same
limit cycle: a self-sustained oscillation of amplitude close to 2 that does not depend on the initial condition.
"""

# %%
# With `μ = 3` the oscillation is a relaxation oscillation, slow drifts broken by fast jumps. The mass starts close to
# the unstable rest point, and the default blueprint shows the position and the velocity growing onto the cycle.

import matplotlib.pyplot as plt

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'van_der_pol'

system = create(MODEL, x=0.1, v=0.0, mu=3.0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=4)
simulation = Simulation(system, dt=0.05, observers=[observer]).run(until=50)

# %%
# Attraction to the limit cycle
# -----------------------------
# Orbits started inside and outside the cycle should spiral onto the same closed curve in the phase plane, whose
# largest displacement is close to 2 for every `μ`.

fig, ax = plt.subplots(figsize=(6, 4))
for x, v in [(0.1, 0.0), (4.0, 6.0), (-3.0, -8.0)]:
    run = Simulation(create(MODEL, x=x, v=v, mu=3.0), dt=0.02).run(until=30)
    ax.plot(run.trajectory_['oscillator.x'], run.trajectory_['oscillator.x_d1'], linewidth=1, label=f'x0 = {x}')
ax.set_xlabel('x')
ax.set_ylabel("x'")
ax.set_title('Three starts, one limit cycle (μ = 3)')
ax.legend()
plt.show()
