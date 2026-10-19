"""
# Closed traffic density equation with a jam instability

The space-time diagram of the density on a ring road shows uniform traffic at the density 0.5 that is unstable: small
density waves grow at a preferred wavelength and form a train of dense jams separated by light traffic.
"""

# %%
# The density and the flux start uniform, with relative perturbations of `1e-4` in every cell, on a ring of length 50
# with 100 cells. The threshold density `D1 gamma rho_max^2 / u0^2` is 0.2 here, so the flow at 0.5 is unstable. The
# default blueprint draws each field of a one-dimensional lattice as a space-time diagram, with the density first.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'traffic_density_equation'

system = create(MODEL, perturbation=1e-4)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default')
simulation = Simulation(system, dt=4.0, solver=RK4(dt=0.1), observers=[observer]).run(until=400)

# %%
# Growth rates of the density waves
# ---------------------------------
# Linearizing the moment equations around the uniform flow gives a 2x2 matrix for each wavenumber `q`, and the larger
# real part of its eigenvalues is the growth rate of the density wave. While the waves are small, the logarithm of each
# Fourier amplitude of the simulated density should grow at that rate, with the fastest growth near `q = 0.9`.

times, rho = simulation.trajectory_.times, simulation.trajectory_['road.rho']
length, D1, gamma, density = 50.0, 0.2, 1.0, 0.5
speed, slope = 1 - density, 1 - 2 * density
amplitudes = np.abs(np.fft.rfft(rho - rho.mean(axis=1, keepdims=True), axis=1))
wavenumbers = 2 * np.pi * np.arange(amplitudes.shape[1]) / length


def growth_rate(q: float) -> float:
    """Return the growth rate of the density wave of wavenumber `q` of the linearized moment equations."""
    diffusion = D1 * density
    matrix = np.array(
        [
            [-diffusion * q**2, -1j * q],
            [1j * q * speed**2 + gamma * slope, -2j * q * speed - gamma - diffusion * q**2],
        ],
    )
    return float(np.linalg.eigvals(matrix).real.max())


grid = np.linspace(0.01, 2.0, 200)
predicted = np.array([growth_rate(q) for q in wavenumbers])
unstable = (predicted > 0.005) & (wavenumbers > 0)
linear = (times >= 20) & (times <= 150)
measured = np.polyfit(times[linear], np.log(amplitudes[linear][:, unstable]), 1)[0]
fig, (spectrum, profile) = plt.subplots(1, 2, figsize=(10, 4))
spectrum.plot(grid, [growth_rate(q) for q in grid], color='black', label='linear theory')
spectrum.plot(wavenumbers[unstable], measured, 'o', label='simulation')
spectrum.axhline(0.0, color='grey', linewidth=0.5)
spectrum.set_xlabel('wavenumber q')
spectrum.set_ylabel('growth rate')
spectrum.set_title('Growth of density waves at rho = 0.5')
spectrum.legend()
cells = np.arange(rho.shape[1]) * length / rho.shape[1]
profile.plot(cells, rho[-1])
profile.axhline(density, color='grey', linestyle='--', label='uniform density')
profile.set_xlabel('position')
profile.set_ylabel('density')
profile.set_title(f'Jams at t = {times[-1]:.0f}')
profile.legend()
fig.tight_layout()
plt.show()
