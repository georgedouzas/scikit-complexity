"""
# Lennard-Jones fluid

Particles that repel at short range and attract weakly at longer range, the Lennard-Jones system of simple atoms like
argon, start on a square lattice with random velocities. The lattice melts within a few collision times and the
particles move as a dense fluid in which every particle sits in a cage of neighbours at about the distance of the
potential minimum. The view shows 400 particles in a periodic box at density 0.5.
"""

# %%
# The example builds `LennardJonesFluidSystem` at constant energy. The velocity Verlet stepper takes steps of 0.005
# and the observer saves every second step with the default blueprint, whose primary view shows the particles.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import neighbors
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.statistical_mechanics import LennardJonesFluidSystem

SYSTEM = 'lennard_jones_fluid'

system = LennardJonesFluidSystem(random_state=0)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(system, until=2.5, observers=[observer])

# %%
# Structure of a gas and a liquid
# -------------------------------
# The radial distribution function `g(r)` is the density of neighbours at the distance `r` relative to an ideal gas.
# It vanishes inside the repulsive core and peaks near the potential minimum `2^(1/6) = 1.12`. With a thermostat at
# `T = 1`, the dilute fluid at density 0.2 should show a single peak of close pairs and little structure beyond it,
# and the dense liquid at density 0.8 a first peak followed by damped shells near 2.1 and 3.2, with `g` tending to one
# far away in both.

fig, ax = plt.subplots(figsize=(7, 4))
for density in (0.2, 0.8):
    side = np.sqrt(400 / density)
    run = simulate(
        LennardJonesFluidSystem(density=density, temperature=1.0, random_state=1).prepare().system_,
        until=20,
        dt=0.05,
        random_state=1,
    )
    positions = np.mod(run.get_values('fluid.x')[100:], side)
    pairs = neighbors(positions, kind='radial-distribution', box=[side, side], bins=120, r_max=5.0)
    distances, g = pairs.radii, pairs.distribution
    ax.plot(distances, g, label=f'density {density}')
ax.axvline(2 ** (1 / 6), color='gray', linestyle='--', label='potential minimum 2^(1/6)')
ax.axhline(1.0, color='lightgray')
ax.set_xlabel('distance r')
ax.set_ylabel('radial distribution g(r)')
ax.set_title('Neighbour shells of a dilute fluid and a dense liquid')
ax.legend()
fig.tight_layout()
plt.show()
