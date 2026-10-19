"""
# Langevin equation with colored noise

The velocity of the particle fluctuates around zero, driven by a noise that keeps its value for about its correlation
time, and it follows the noise smoothly with a smaller amplitude.
"""

# %%
# The model is created by name from the catalog with one particle, unit friction and noise intensity, and a noise
# correlation time `tau = 1`, and simulated for 60 relaxation times. The observer saves every frame with the default
# blueprint, whose primary view holds the velocity `v` and the noise `xi` of the particle.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'colored_noise_langevin'

system = create(MODEL, n_particles=1, gamma=1.0, D=1.0, tau=1.0, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.2, solver=EulerMaruyama(dt=0.01), observers=[observer], random_state=0)
simulation.run(until=59.8)

# %%
# Velocity variance and noise correlation
# ---------------------------------------
# For the noise correlation `<xi(t) xi(t')> = D / tau exp(-|t - t'| / tau)`, the stationary velocity variance is
# `<v^2> = D / (gamma (1 + gamma tau))`, which tends to the white noise value `D / gamma` as `tau` goes to 0. The
# variances measured in ensembles of 500 particles over a range of correlation times should follow this curve, and the
# correlation of the noise in the ensemble at `tau = 1` should decay as `exp(-t / tau)`.

taus = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
variances = []
for tau in taus:
    run = Simulation(
        create(MODEL, n_particles=500, tau=tau, random_state=0),
        dt=0.5,
        solver=EulerMaruyama(dt=min(0.01, tau / 20)),
        random_state=0,
    ).run(until=20)
    variances.append(np.mean(run.trajectory_['particles.v'][20:] ** 2))
    if tau == 1.0:
        noise = run.trajectory_['particles.xi']

lags = np.arange(11) * 0.5
correlation = np.array([np.mean(noise[: len(noise) - lag] * noise[lag:]) for lag in range(11)]) / np.mean(noise**2)

fig, (variance, decay) = plt.subplots(1, 2, figsize=(10, 4))
grid = np.logspace(-1.5, 0.7, 100)
variance.semilogx(grid, 1 / (1 + grid), color='black', label='D / (gamma (1 + gamma tau))')
variance.semilogx(taus, variances, 'o', label='simulated')
variance.axhline(1.0, color='grey', linestyle='--', label='white noise D / gamma')
variance.set_xlabel('correlation time tau')
variance.set_ylabel('stationary <v^2>')
variance.set_title('Colored noise lowers the velocity variance')
variance.legend()
decay.plot(lags, correlation, 'o', label='simulated')
decay.plot(lags, np.exp(-lags), color='black', label='exp(-t / tau)')
decay.set_xlabel('lag t')
decay.set_ylabel('<xi(0) xi(t)> / <xi^2>')
decay.set_title('The noise forgets in its correlation time')
decay.legend()
fig.tight_layout()
plt.show()
