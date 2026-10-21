"""
# Stochastic resonance

A particle sits in one well of the double well `U = x^4 / 4 - x^2 / 2`, which a weak periodic force tilts back and
forth. The force alone is far too weak to push the particle over the barrier, but noise of the right intensity makes
it hop between the wells about once every half period, so the hops lock to the forcing. The view shows the position
of the particle jumping between `-1` and `1` in step with the slow forcing of period 100.
"""

# %%
# The model is created by name from the catalog with forcing amplitude `A = 0.1`, frequency `f = 0.01`, and noise
# intensity `D = 0.1`. The Euler-Maruyama stepper takes steps of 0.05 and the observer saves every frame of 4 with the
# default blueprint, whose primary view is the time series of the position.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import EulerMaruyama
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'stochastic_resonance'

system = create(MODEL, D=0.1)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
solver = EulerMaruyama(dt=0.05)
simulation = Simulation(system, solver=solver, dt=4.0, observers=[observer], random_state=0).run(until=1000)

# %%
# Resonance in the noise intensity
# --------------------------------
# The response of the particle to the forcing is the amplitude of the Fourier component of its well, `sign(x)`, at
# the forcing frequency. Too little noise and the particle stays in one well; too much and it hops at random. The
# response should peak at an intermediate intensity, near the one at which the Kramers time
# `2 pi / sqrt(2) exp(1 / (4 D))` between hops equals half the forcing period.

intensities = np.array([0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.22, 0.3])
responses = []
for intensity in intensities:
    run = Simulation(create(MODEL, D=intensity), solver=solver, dt=1.0, random_state=1).run(until=4000)
    times = run.trajectory_.times
    well = np.sign(run.trajectory_['particle.x'])
    responses.append(2 * np.abs(np.mean(well * np.exp(-2j * np.pi * 0.01 * times))))
matching = 1 / (4 * np.log(50 * np.sqrt(2) / (2 * np.pi)))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(intensities, responses, 'o-', label='simulation')
ax.axvline(matching, color='gray', linestyle='--', label='Kramers time = half period')
ax.set_xlabel('noise intensity D')
ax.set_ylabel('response of sign(x) at the forcing frequency')
ax.set_title('Noise of the right strength amplifies a weak signal')
ax.legend()
fig.tight_layout()
plt.show()
