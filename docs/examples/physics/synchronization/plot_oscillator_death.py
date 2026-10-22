"""
# Oscillator death

Fifty limit-cycle oscillators turn on their unit circles until the strong global coupling is switched on, and then all
of them come to rest at 0. Every oscillator would oscillate alone, but with frequencies spread wider than the coupling
can pull together, the coupling kills the oscillations instead of synchronizing them.
"""

# %%
# The frequencies of the oscillators follow a Lorentzian of width `ω0 = 1.5`, above the critical width 1. The schedule
# keeps them uncoupled for 5 time units and then sets the coupling to `K = 3`. The default blueprint shows the real
# parts `x_j` of all oscillators as time series.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'oscillator_death'

system = create(MODEL, n_oscillators=50, K=3.0, width=1.5, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)


def coupling_at(time):
    """Return no coupling for the first 5 time units and the coupling 3 afterwards."""
    return 0.0 if time < 5.0 else 3.0


simulation = Simulation(
    system,
    dt=0.05,
    observers=[observer],
    schedule={'coupling.K': coupling_at},
    observables=describe(MODEL).observables_of(system),
).run(until=14)

# %%
# Decay rate against the width of the frequencies
# -----------------------------------------------
# For a Lorentzian density, Bertin's linear analysis gives one collective mode that grows at the rate `1 - ω0`,
# whatever `K > 1`. Below the width 1 the oscillations should survive, and above it the mean amplitude should decay as
# `exp((1 - ω0) t)`, shown dashed, while the other modes decay faster at the rate `1 - K`.

widths = [0.5, 1.25, 1.5, 2.0]
fig, ax = plt.subplots(figsize=(6, 4))
for width in widths:
    model_system = create(MODEL, n_oscillators=100, K=3.0, width=width)
    run = Simulation(model_system, dt=0.1, observables=describe(MODEL).observables_of(model_system)).run(until=20)
    times, amplitude = np.asarray(run.trajectory_.times), np.asarray(run.trajectory_['amplitude'])
    line = ax.semilogy(times, amplitude, label=f'ω0 = {width}')[0]
    if width > 1:
        anchor = np.searchsorted(times, 8.0)
        predicted = amplitude[anchor] * np.exp((1 - width) * (times - times[anchor]))
        ax.semilogy(times, predicted, '--', color=line.get_color())
ax.set_ylim(1e-6, 2)
ax.set_xlabel('time')
ax.set_ylabel('mean amplitude')
ax.set_title('Oscillator death above the critical width 1, K = 3')
ax.legend()
plt.show()
