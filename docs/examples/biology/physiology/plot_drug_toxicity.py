"""
# Cumulative drug toxicity

Six weekly doses raise the plasma concentration in spikes, the toxicity climbs in steps with every dose, and once the
doses stop the metabolism lowers it.
"""

# %%
# The system is built from its class: a one-day infusion every week, a drug eliminated at the rate 0.5 a day, and a
# toxicity cleared at the rate 0.05 a day. The observer saves every second frame with the default blueprint, whose
# primary view is the time series of the concentration and the toxicity.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import DrugToxicitySystem

SYSTEM = 'drug_toxicity'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(DrugToxicitySystem(), observers=[observer])

# %%
# Toxicity and exposure
# ---------------------
# Without clearance the toxicity is the integral of the concentration, which the run without clearance should match.
# With clearance at the rate `eta`, the toxicity should decay as `exp(-eta t)` after the last dose.

times = trajectory.times
cumulative = simulate(DrugToxicitySystem(eta=0.0))
exposure = np.concatenate(
    (
        [0.0],
        np.cumsum(
            np.diff(times) * (cumulative.get_values('patient.c')[1:] + cumulative.get_values('patient.c')[:-1]) / 2,
        ),
    ),
)
toxicity = trajectory.get_values('patient.T')
late = times >= 50.0
fig, (accumulation, decay) = plt.subplots(1, 2, figsize=(10, 4))
accumulation.plot(times, cumulative.get_values('patient.T'), label='toxicity, eta = 0')
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
