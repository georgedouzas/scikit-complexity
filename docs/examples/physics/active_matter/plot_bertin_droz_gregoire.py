"""
# Bertin-Droz-Grégoire kinetic flocking

Bertin, Droz, and Grégoire turned the Vicsek rules into a kinetic equation for the density of particles over position
and orientation. Particles tumble at a fixed rate by a noisy angle, which loses polar order, and collide in pairs and
leave at their mean angle with the same noise, which gains it. Above a transition density the gain wins, so a nearly
isotropic gas orders spontaneously. The chart shows the polar order of a homogeneous gas at twice the transition
density, which grows exponentially and saturates.
"""

# %%
# The system is built from its class on a single position with 32 orientations, since the initial state of
# the system is homogeneous and streaming therefore leaves it so. With the noise 0.5, the interaction range 0.5, and the
# unit speed and tumbling rate of the defaults, the transition density is close to 0.43. The observable `polar_order` is
# computed from the density at every frame of 1 up to `t = 60`, and the default blueprint shows the observables first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.physics.active_matter import BertinDrozGregoireSystem

SYSTEM = 'bertin_droz_gregoire'
ANGLES = 32
NOISE = 0.5
SURVIVAL = np.exp(-(NOISE**2) / 2)
TRANSITION = np.pi * (1 - SURVIVAL) / (8 * 0.5 * (SURVIVAL - 2 / 3))
THETA = 2 * np.pi * (np.arange(ANGLES) + 0.5) / ANGLES


def polar_order(_, values):
    """Return the modulus of the mean of the unit vectors of the orientations."""
    density = np.ravel(values['flock.f'])
    return np.abs(density @ np.exp(1j * THETA)) / density.sum()


observables = [ObservableSpec('polar_order', function=polar_order)]
system = BertinDrozGregoireSystem(positions=1, angles=ANGLES, density=2 * TRANSITION)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='observables', interval=1)
solver = RK4Solver(dt=0.2)
trajectory = simulate(system, until=60, solver=solver, dt=1.0, observers=[observer], observables=observables)

# %%
# The transition to polar order
# -----------------------------
# Linearizing the equation about the isotropic gas gives the growth rate
# `8 interaction_range speed (exp(-noise^2 / 2) - 2 / 3) (density - transition) / pi` of a small polar order, which is
# negative below the transition density and positive above it. The rate measured over the first 10 units of time should
# follow this line. Started from a larger polar order, so that the slow growth near the transition does not hide it, the
# polar order at long times should rise steeply from zero above the transition, where the chart starts it, and level off
# at larger densities.

ratios = np.array([0.5, 0.8, 1.2, 1.5, 2.0, 3.0, 4.0])
rates = []
for ratio in ratios:
    gas = BertinDrozGregoireSystem(positions=1, angles=ANGLES, density=ratio * TRANSITION)
    run = simulate(gas, until=10, solver=solver, dt=10.0, observables=observables)
    order = run.values('polar_order')
    rates.append(np.log(order[-1] / order[0]) / 10)
above = ratios[ratios > 1]
orders = []
for ratio in above:
    gas = BertinDrozGregoireSystem(positions=1, angles=ANGLES, density=ratio * TRANSITION, perturbation=0.5)
    run = simulate(gas, until=150, solver=solver, dt=150.0, observables=observables)
    orders.append(run.values('polar_order')[-1])
grid = np.linspace(ratios[0], ratios[-1], 100)
theory = 8 * 0.5 * (SURVIVAL - 2 / 3) * (grid - 1) * TRANSITION / np.pi

fig, (growth, saturation) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(ratios, rates, 'o', label='simulation')
growth.plot(grid, theory, 'k--', label='linear theory')
growth.axhline(0, color='gray', linewidth=0.8)
growth.axvline(1, color='gray', linewidth=0.8)
growth.set_xlabel('density / transition density')
growth.set_ylabel('growth rate of the polar order')
growth.set_title('The isotropic gas turns unstable at the transition')
growth.legend()
saturation.plot([1.0, *above], [0.0, *orders], 'o-')
saturation.axvline(1, color='gray', linewidth=0.8)
saturation.set_xlabel('density / transition density')
saturation.set_ylabel('polar order at t = 150')
saturation.set_title('Polar order rises from zero above it')
fig.tight_layout()
plt.show()
