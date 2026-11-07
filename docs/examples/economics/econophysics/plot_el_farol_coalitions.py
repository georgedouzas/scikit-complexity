"""
# El Farol with coalitions

Patrons decide whether to go to the El Farol bar by predicting the attendance from the last few weeks. In Collins'
version they can form coalitions that use the best predictor of all their members, splitting after crowded nights and
merging after pleasant ones. Better predictors do not help: coalitions go together, crowd the bar together, and enjoy
fewer nights than lone patrons.
"""

# %%
# A hundred patrons with ten strategies of memory five go when they predict at most 60 patrons, for 500 nights. The
# default blueprint shows the attendance, the number of coalitions, and the mean return.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'el_farol_coalitions'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
grouped = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run(until=500).trajectory_
alone = create_simulation(SYSTEM, coalitions=False).prepare().run(until=500).trajectory_

# %%
# Lone patrons and coalitions
# ---------------------------
# As in Table 1 of Collins, lone patrons should keep the attendance near the capacity and enjoy about half their
# nights, while coalitions make the attendance swing between empty and crowded nights and lower the returns of all.

fig, (attendance, returns) = plt.subplots(1, 2, figsize=(10, 4))
for name, trajectory in (('lone patrons', alone), ('coalitions', grouped)):
    times = np.asarray(trajectory.times)
    attendance.plot(times, trajectory.values('bar.attendance'), lw=0.7, label=name)
    returns.plot(times, trajectory.values('bar.mean_score'), label=name)
attendance.axhline(60, color='black', linestyle=':', lw=0.8)
attendance.set_xlabel('night')
attendance.set_ylabel('attendance')
attendance.set_title('Attendance')
attendance.legend()
returns.set_xlabel('night')
returns.set_ylabel('mean smoothed return')
returns.set_title('Returns of the patrons')
returns.legend()
fig.tight_layout()
plt.show()
