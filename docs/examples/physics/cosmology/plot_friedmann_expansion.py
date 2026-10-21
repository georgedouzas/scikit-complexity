"""
# Friedmann expansion

A flat universe of matter and dark energy expands ever faster: matter brakes the expansion while the universe is
small, and once it has grown enough the cosmological constant takes over and the expansion accelerates towards an
exponential.
"""

# %%
# The run starts in the past, at the scale factor `a = 0.3`, with today's mixture `Ω_m = 0.3` and `Ω_Λ = 0.7` and
# `H0 = 1`. The factory sets the initial rate `a'` from the Friedmann equation. The default blueprint shows `a` and
# `a'`: the rate first falls while matter dominates and then rises.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'friedmann_expansion'

system = create(MODEL, a=0.3, H0=1.0, omega_matter=0.3, omega_lambda=0.7)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
simulation = Simulation(system, dt=0.01, observers=[observer]).run(until=2.0)

# %%
# Exact solution
# --------------
# The flat universe of matter and a cosmological constant has the exact scale factor
# `a(τ) = (Ω_m / Ω_Λ)^(1/3) sinh^(2/3)(3 sqrt(Ω_Λ) H0 τ / 2)`, with `τ` the time since the big bang. The simulated
# scale factor should lie on it, and the deceleration should turn into acceleration at `a = (Ω_m / (2 Ω_Λ))^(1/3)`.

matter, vacuum = 0.3, 0.7


def exact(age):
    """Return the scale factor of the flat matter and vacuum universe at an age."""
    return (matter / vacuum) ** (1 / 3) * np.sinh(1.5 * np.sqrt(vacuum) * age) ** (2 / 3)


start = 2 / (3 * np.sqrt(vacuum)) * np.arcsinh(np.sqrt(vacuum / matter * 0.3**3))
times = np.asarray(simulation.trajectory_.times)
scale = np.ravel(simulation.trajectory_['universe.a'])
rate = np.ravel(simulation.trajectory_['universe.a_d1'])
fig, (growth, speed) = plt.subplots(1, 2, figsize=(10, 4))
growth.plot(times, scale, label='simulated')
growth.plot(times, exact(start + times), '--', color='black', label='exact')
growth.set_xlabel('time')
growth.set_ylabel('scale factor a')
growth.set_title('Scale factor against the exact solution')
growth.legend()
speed.plot(scale, rate)
speed.axvline((matter / (2 * vacuum)) ** (1 / 3), linestyle='--', color='grey', label='start of acceleration')
speed.set_xlabel('scale factor a')
speed.set_ylabel("expansion rate a'")
speed.set_title('Deceleration, then acceleration')
speed.legend()
fig.tight_layout()
plt.show()
