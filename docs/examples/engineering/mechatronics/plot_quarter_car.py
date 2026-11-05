"""
# Quarter car

A body rests on a spring and a damper above a wheel, and a tire spring joins the wheel to the road. When the road steps
up, the wheel responds first and fast, the body follows slowly, and both settle at the height of the step.
"""

# %%
# The road rises by 5 cm at half a second. The default blueprint shows both displacements and the energy of the springs.

import matplotlib.pyplot as plt

from skcomplex.engine.runtime import Simulation
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'quarter_car'

system = create(MODEL, height=0.05)
observables = describe(MODEL).observables_of(system)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.005, observers=[observer], observables=observables).prepare().run(until=3.0)

# %%
# The step response
# -----------------
# The wheel reaches the height of the road first on the stiff tire. The body rises later and overshoots by more than
# half the step, and both oscillate until the damper has taken out the energy.

trajectory = simulation.trajectory_
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(trajectory.times, trajectory.values('car.y1'), label='body')
ax.plot(trajectory.times, trajectory.values('car.y2'), label='wheel')
ax.axhline(0.05, color='black', linestyle='--', label='road')
ax.set_xlabel('time')
ax.set_ylabel('displacement')
ax.set_title('A road step through the suspension')
ax.legend()
plt.show()
