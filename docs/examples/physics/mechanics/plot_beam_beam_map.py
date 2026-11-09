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

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.mechanics import BeamBeamMapSystem

SYSTEM = 'beam_beam_map'

system = BeamBeamMapSystem()
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=20)
trajectory = simulate(system, observers=[observer])

# %%
# Phase portrait
# --------------
# Orbits started at growing distances from the origin trace closed invariant curves near it. Further out, a resonance
# of the tune should break the curves into a chain of islands, as in the standard map.

fig, ax = plt.subplots(figsize=(6, 6))
for x in np.linspace(0.2, 4.0, 16):
    run = simulate(BeamBeamMapSystem(x=x, y=0.0), until=1500)
    ax.plot(np.ravel(run.get_values('particle.x')), np.ravel(run.get_values('particle.y')), ',')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Beam-beam map, nu = 0.3, c = 0.05')
plt.show()
