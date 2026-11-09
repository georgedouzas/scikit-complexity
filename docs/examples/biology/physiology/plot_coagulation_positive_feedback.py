"""
# Coagulation positive feedback

An injury of one time unit starts the coagulation cascade. Thrombin activates its own production, so the small
trigger is amplified into a burst of thrombin, which lays down a clot of fibrin. Thrombin also activates protein C,
which ends the burst, and the fibrin activates plasmin, which dissolves the clot. The view shows the sharp burst of
thrombin, the larger clot of fibrin that follows it and dissolves, and the slow decay of protein C and plasmin.
"""

# %%
# The system is built from its class with an injury of rate 1 from time 1 to time 2 and the feedback `ka = 3`, above
# the bistability threshold 2. The observer saves every second frame of the run of 40 time units with the default
# blueprint, whose primary view is the time series of thrombin `T`, protein C `C`, fibrin `F`, and plasmin `P`.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.physiology import CoagulationPositiveFeedbackSystem

SYSTEM = 'coagulation_positive_feedback'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=2)
trajectory = simulate(CoagulationPositiveFeedbackSystem(), observers=[observer], until=40)

# %%
# All or nothing
# --------------
# The feedback `ka T^2 / (1 + T^2) - T` has an unstable state at `T = (3 - sqrt(5)) / 2 = 0.38` for `ka = 3`. An
# injury that lifts thrombin past it should set off a full burst, while a weaker one should fade, so the peak of
# thrombin should jump at a threshold of injury and then hardly grow. Every burst should leave a clot that plasmin
# dissolves.

injuries = np.linspace(0.05, 2.0, 40)
peaks, clots = [], []
for injury in injuries:
    trajectory = simulate(CoagulationPositiveFeedbackSystem(injury=injury))
    peaks.append(trajectory.get_values('blood.T').max())
    clots.append(trajectory.get_values('blood.F').max())
fig, (response, clot) = plt.subplots(1, 2, figsize=(10, 4))
response.plot(injuries, peaks, 'o-', markersize=3)
response.axhline((3 - np.sqrt(5)) / 2, color='black', linestyle=':', label='unstable state of the feedback')
response.set_xlabel('injury rate')
response.set_ylabel('peak thrombin')
response.set_title('The burst of thrombin is all or nothing')
response.legend()
clot.plot(trajectory.times, trajectory.get_values('blood.F'), label='fibrin')
clot.plot(trajectory.times, trajectory.get_values('blood.P'), label='plasmin')
clot.set_xlabel('time')
clot.set_ylabel('level')
clot.set_title('Fibrinolysis dissolves the clot')
clot.legend()
fig.tight_layout()
plt.show()
