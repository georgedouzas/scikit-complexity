"""
# Insider sabotage

Employees wander through an office under the eyes of three supervisors. Stress and disgruntlement accumulate over
the days of the year, so the risk that an employee sabotages the organization creeps up for everyone, and a few
employees drift into a high-risk tail.
"""

# %%
# The model is created by name from the catalog with 130 employees and 3 supervisors on a 20 by 20 office and a yearly
# turnover of 15%, and runs for one year, one step per day. The default blueprint shows the employees colored by risk,
# from 0 (dark) to 5 (bright), with the supervisors.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import histogram
from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'insider_sabotage'

system = create(MODEL, random_state=0)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=1, observers=[observer], random_state=0).run(until=365)

# %%
# Risk builds up over the year
# ----------------------------
# The mean risk of the employees should rise steadily over the year, and the upper tail should rise faster than the
# mean, so the distribution of risk at the end of the year should be wider and shifted to the right of the first day.

trajectory = simulation.trajectory_
risks = [np.asarray(trajectory.agents('employees', frame).values['risk']) for frame in range(len(trajectory.times))]
fig, (trend, spread) = plt.subplots(1, 2, figsize=(10, 4))
trend.plot(trajectory.times, [risk.mean() for risk in risks], label='mean')
trend.plot(trajectory.times, [np.percentile(risk, 90) for risk in risks], label='90th percentile')
trend.set_xlabel('day')
trend.set_ylabel('risk of sabotage')
trend.set_title('Risk creeps up over the year')
trend.legend()
for frame, label in [(0, 'first day'), (len(risks) - 1, 'last day')]:
    edges, density = histogram(risks[frame], bins=20)
    spread.stairs(density, edges, fill=True, alpha=0.5, label=label)
spread.set_xlabel('risk of sabotage')
spread.set_ylabel('probability density')
spread.set_title('The distribution widens and shifts')
spread.legend()
fig.tight_layout()
plt.show()
