"""
# Vlasov-Poisson plasma

Electrons stream through a neutralizing background and feel only the electric field of their own density, with no
collisions at all. Two beams passing through each other are unstable: a small density wave bunches the beams, the
bunches strengthen the wave, and the wave grows until it traps the electrons, which roll up into a vortex in the plane
of position and velocity. The image shows the density in that plane, with position down and velocity across.
"""

# %%
# The system is built from its class in its two-stream set-up, with beams of temperature 0.25 at the
# velocities -1.5 and 1.5, a density wave of relative amplitude 1e-4, a periodic position axis of 64 cells, and a
# velocity axis of 128. The catalog registers the semi-Lagrangian stepper for this system, which shifts the density
# along its characteristics. The observer saves every frame of 0.25 up to `t = 50`, with the default blueprint and the
# image of the density first.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import wofz

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import SemiLagrangianSolver
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.physics.fluids import VlasovEquationSystem

SYSTEM = 'vlasov_equation'
WAVE_NUMBER = 0.5

system = VlasovEquationSystem(positions=64, velocities=128, setup='two-stream', amplitude=1e-4)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', primary='gas/f', interval=1)
solver = SemiLagrangianSolver(0.25)
simulation = simulate(system, until=50, solver=solver, observers=[observer])

# %%
# Damping and growth of the density wave
# --------------------------------------
# With no collisions to dissipate it, a density wave can still decay or grow, by exchanging energy with the electrons
# that move with it. Linear theory gives the rates. On a single Maxwellian, the wave of wave number 0.5 should oscillate
# inside an envelope that decays at the Landau damping rate 0.1533. On the two beams, it should grow at the rate at
# which the kinetic dispersion relation vanishes, on the imaginary frequency axis, until trapping saturates it.


def dispersion(rate):
    """Return the dielectric function of the two beams at a purely growing frequency."""
    response = 0.0
    for beam in (-1.5, 1.5):
        argument = (1j * rate / WAVE_NUMBER - beam) / np.sqrt(2 * 0.25)
        plasma = 1j * np.sqrt(np.pi) * wofz(argument)
        response += 0.5 * (1 + argument * plasma) / (WAVE_NUMBER**2 * 0.25)
    return np.real(1 + response)


def wave_amplitude(trajectory, positions, velocities):
    """Return the amplitude of the density wave of every frame."""
    density = np.asarray(trajectory.values('gas.x')).reshape(-1, positions, velocities).sum(axis=2)
    return np.abs(np.fft.rfft(density, axis=1)[:, 1])


growth_rate = brentq(dispersion, 0.01, 1.0)
times = np.asarray(simulation.times)
amplitude = wave_amplitude(simulation, 64, 128)
landau = simulate(VlasovEquationSystem(), solver=solver)
landau_times = np.asarray(landau.times)
landau_amplitude = wave_amplitude(landau, 32, 64)

fig, (damping, growth) = plt.subplots(1, 2, figsize=(10, 4))
damping.semilogy(landau_times, landau_amplitude, label='density wave, simulation')
damping.semilogy(landau_times, landau_amplitude[0] * np.exp(-0.1533 * landau_times), 'k--', label='rate 0.1533')
damping.set_xlabel('time t')
damping.set_ylabel('amplitude of the density wave')
damping.set_title('A Maxwellian damps the wave')
damping.legend()
growth.semilogy(times, amplitude, label='density wave, simulation')
anchor = np.searchsorted(times, 20.0)
linear = amplitude[anchor] * np.exp(growth_rate * (times - times[anchor]))
growth.semilogy(times, linear, 'k--', label=f'rate {growth_rate:.3f}')
growth.set_ylim(amplitude.min() / 2, amplitude.max() * 2)
growth.set_xlabel('time t')
growth.set_ylabel('amplitude of the density wave')
growth.set_title('Two beams grow it until trapping saturates it')
growth.legend()
fig.tight_layout()
plt.show()
