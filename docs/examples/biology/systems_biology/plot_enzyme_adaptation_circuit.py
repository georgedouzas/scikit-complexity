"""
# Enzyme adaptation circuit

Three enzymes in a negative feedback loop with a buffer node respond to a step of their input with a sharp transient of
the output, then return to the output they had before, perfect adaptation as in chemotaxis and sensory systems.
"""

# %%
# The system is built from its class with the adapting loop and a step of the input from 0.5 to 0.6 at time 300. The
# observer saves every frame with the default blueprint, whose primary view is the time series of the three enzymes.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import EnzymeAdaptationCircuitSystem

SYSTEM = 'enzyme_adaptation_circuit'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=4)
trajectory = simulate(EnzymeAdaptationCircuitSystem(), observers=[observer])

# %%
# Sensitivity, precision, and robustness
# --------------------------------------
# Ma et al. call a circuit adaptive when its sensitivity, the relative peak over the relative input change, exceeds 1
# and its precision, the inverse relative error of the return, exceeds 10. Random changes of up to 20% of all rate
# constants should mostly keep the loop adaptive.

rates = np.array([[0.419, 0.3, 5.061], [7.809, 9.355, 4.852], [0.735, 1.176, 1.928], [7.282, 0.182, 7.914]])
rng = np.random.default_rng(0)
scores = []
for _ in range(30):
    factors = rng.uniform(0.8, 1.2, (4, 3))
    run = simulate(EnzymeAdaptationCircuitSystem(k=(rates * factors).tolist()))
    output, times = run.get_values('C.X'), run.times
    before, after = output[times < 300][-1], output[times >= 300]
    peak = after[np.argmax(np.abs(after - before))]
    scores.append((abs(peak - before) / before / 0.2, 0.2 * before / max(abs(output[-1] - before), 1e-12)))
fig, (series, robustness) = plt.subplots(1, 2, figsize=(11, 4.5))
for node in ('A', 'B', 'C'):
    series.plot(trajectory.times, trajectory.get_values(f'{node}.X'), label=node)
series.axvline(300, color='gray', linestyle='--', label='input step')
series.set_xlabel('time')
series.set_ylabel('active fraction')
series.set_title('Transient response and return of the output C')
series.legend()
sensitivity, precision = np.array(scores).T
robustness.loglog(sensitivity, precision, 'o')
robustness.axvline(1, color='gray', linestyle='--')
robustness.axhline(10, color='gray', linestyle='--')
robustness.set_xlabel('sensitivity')
robustness.set_ylabel('precision')
robustness.set_title(f'{np.mean((sensitivity > 1) & (precision > 10)):.0%} of perturbed loops adapt')
fig.tight_layout()
plt.show()
