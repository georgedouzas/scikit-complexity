"""
# Rössler-driven Lorenz and generalized synchronization

A chaotic Rössler system drives a Lorenz-63 system that has nothing in common with it. Above a critical coupling the
Lorenz system forgets its own initial state and becomes a function of the state of its drive, generalized
synchronization.
"""

# %%
# The coupling is `k = 8`, above the critical `k ≈ 6.66`, and a second copy of the Lorenz system starts from another
# state under the same drive. The default blueprint shows the drive, the response, and its copy.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.synchronization import RosslerDrivenLorenzSystem

SYSTEM = 'rossler_driven_lorenz'

system = RosslerDrivenLorenzSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
simulation = simulate(system, dt=0.05, observers=[observer])

# %%
# Auxiliary system method
# -----------------------
# The distance between the two copies of the response, averaged over the second half of the run, should fall to zero
# above the critical coupling, dashed, and stay of the size of the attractor below it.

couplings = np.linspace(0.0, 12.0, 7)
errors = []
for k in couplings:
    trajectory = simulate(RosslerDrivenLorenzSystem(k=k), dt=0.05)
    squares = [
        (np.ravel(trajectory.values(f'response.{name}')) - np.ravel(trajectory.values(f'auxiliary.{name}'))) ** 2
        for name in 'xyz'
    ]
    errors.append(np.sqrt(sum(squares))[len(squares[0]) // 2 :].mean())
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(couplings, np.maximum(errors, 1e-12), 'o-')
ax.axvline(6.66, linestyle='--', color='black')
ax.set_xlabel('coupling k')
ax.set_ylabel('mean distance of the two responses')
ax.set_title('Generalized synchronization')
plt.show()
