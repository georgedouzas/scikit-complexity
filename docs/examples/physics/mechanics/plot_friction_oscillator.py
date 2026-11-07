"""
# Mass-spring oscillator with static and Coulomb friction

A mass on a spring slides on a rough floor. Dry friction takes the same amount of energy every swing, so the swings
shrink linearly rather than exponentially, and the mass sticks after finitely many swings anywhere inside a band of
equilibria.
"""

# %%
# The spring has unit stiffness, the kinetic friction is `mu_k = 0.05` and the static friction `mu_s = 0.06`, and the
# mass starts at rest at `x = 1.03`. The default blueprint shows the position and the velocity.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.mechanics import FrictionOscillatorSystem

SYSTEM = 'friction_oscillator'

system = FrictionOscillatorSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
trajectory = simulate(system, observers=[observer])

# %%
# Dry and viscous friction
# ------------------------
# In the phase plane, dry friction should give half ellipses around the shifted centres `x = ∓ mu_k`, every one
# `2 mu_k` smaller than the one before, ending on the segment of equilibria `|x| <= mu_s`, shaded. Viscous friction
# alone should give the familiar spiral that shrinks exponentially and only reaches the origin.

fig, ax = plt.subplots(figsize=(6, 5))
for label, settings in [('dry', {}), ('viscous', {'mu_s': 0.0, 'mu_k': 0.0, 'c': 0.05})]:
    run = simulate(FrictionOscillatorSystem(**settings))
    ax.plot(np.ravel(run.values('mass.x')), np.ravel(run.values('mass.v')), label=label)
ax.axvspan(-0.06, 0.06, color='grey', alpha=0.3, label='equilibria')
ax.set_xlabel('x')
ax.set_ylabel('v')
ax.set_title('Stick-slip under dry friction')
ax.legend()
plt.show()
