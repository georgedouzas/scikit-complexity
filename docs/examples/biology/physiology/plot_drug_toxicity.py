"""
# Cumulative drug toxicity

Six weekly doses raise the plasma concentration in spikes, the toxicity climbs in steps with every dose, and once the
doses stop the metabolism lowers it.
"""

# %%
# The model is created by name from the catalog: a one-day infusion every week, a drug eliminated at the rate 0.5 a
# day, and a toxicity cleared at the rate 0.05 a day. The observer saves every second frame with the default blueprint,
# whose primary view is the time series of the concentration and the toxicity.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'drug_toxicity'

system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.2, observers=[observer]).run(until=84)

# %%
# Toxicity and exposure
# ---------------------
# Without clearance the toxicity is the integral of the concentration, which the run without clearance should match.
# With clearance at the rate `eta`, the toxicity should decay as `exp(-eta t)` after the last dose.

times = simulation.trajectory_.times
cumulative = Simulation(create(MODEL, eta=0.0), dt=0.2).run(until=84).trajectory_
exposure = np.concatenate(
    ([0.0], np.cumsum(np.diff(times) * (cumulative['patient.c'][1:] + cumulative['patient.c'][:-1]) / 2)),
)
toxicity = simulation.trajectory_['patient.T']
late = times >= 50.0
fig, (accumulation, decay) = plt.subplots(1, 2, figsize=(10, 4))
accumulation.plot(times, cumulative['patient.T'], label='toxicity, eta = 0')
accumulation.plot(times, exposure, '--', color='black', label='integral of c')
accumulation.plot(times, toxicity, label='toxicity, eta = 0.05')
accumulation.set_xlabel('time (days)')
accumulation.set_ylabel('toxicity')
accumulation.set_title('Toxicity integrates the exposure')
accumulation.legend()
decay.semilogy(times[late], toxicity[late], label='simulated')
decay.semilogy(
    times[late],
    toxicity[late][0] * np.exp(-0.05 * (times[late] - times[late][0])),
    '--',
    color='black',
    label='exp(-eta t)',
)
decay.set_xlabel('time (days)')
decay.set_ylabel('toxicity')
decay.set_title('Clearance after the last dose')
decay.legend()
fig.tight_layout()
plt.show()
