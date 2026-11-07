"""
# Beam-beam map

A particle of an accelerator beam turns by the tune once per revolution and is kicked by the charge of the opposite
beam where the beams cross. The kick is nonlinear, so the phase space of this area-preserving map shows invariant
curves around the centre and chains of islands further out.
"""

# %%
# The tune is `nu = 0.3` and the strength of the kick `c = 0.05`, for which the origin is stable. The default blueprint
# shows the position and the momentum turn by turn.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system

SYSTEM = 'beam_beam_map'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
simulation = Simulation(system, dt=1, observers=[observer]).prepare().run(until=2000)

# %%
# Phase portrait
# --------------
# Orbits started at growing distances from the origin trace closed invariant curves near it. Further out, a resonance
# of the tune should break the curves into a chain of islands, as in the standard map.

fig, ax = plt.subplots(figsize=(6, 6))
for x in np.linspace(0.2, 4.0, 16):
    run = Simulation(create_system(SYSTEM, x=x, y=0.0), dt=1).prepare().run(until=1500)
    ax.plot(np.ravel(run.trajectory_.values('particle.x')), np.ravel(run.trajectory_.values('particle.y')), ',')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Beam-beam map, nu = 0.3, c = 0.05')
plt.show()
