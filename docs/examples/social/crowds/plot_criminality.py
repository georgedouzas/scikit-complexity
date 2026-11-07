"""
# Criminality and security

Citizens are a distribution over criminality classes and security agents a distribution over efficacy classes.
Encounters between citizens spread crime, encounters with security agents deter it, and the security forces grow in
proportion to the mean criminality and retire. Crime and security oscillate in turn and settle at a balance where some
crime persists.
"""

# %%
# A fifth of the citizens start in the most criminal class. The blueprint shows the mean criminality and the density
# of the security forces, which rise and fall in turn toward their balance.

import matplotlib.pyplot as plt
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.social.crowds import CriminalitySystem

SYSTEM = 'criminality'

# The distributions have no spatial view that shows the balance, so the views are the two series over the run.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/mean_criminality', name='mean criminality', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/security_forces', name='security forces', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
system = CriminalitySystem()
trajectory = simulate(system, until=100, observers=[observer])

# %%
# The cycle of crime and security
# -------------------------------
# Recruitment reads the mean criminality, a moment of the citizens, so the security forces lag behind crime. The path
# in the plane of mean criminality and security forces should spiral into the balance, and without deterrence and
# recruitment crime should spread through most of the society.

unchecked = simulate(CriminalitySystem(deterrence=0.0, recruitment=0.0), until=100, dt=0.5)
unchecked_criminality = unchecked.values('mean_criminality')
fig, (phase_axis, time_axis) = plt.subplots(1, 2, figsize=(10, 4))
phase_axis.plot(trajectory.values('mean_criminality'), trajectory.values('security_forces'))
phase_axis.plot(trajectory.values('mean_criminality')[0], trajectory.values('security_forces')[0], 'o', label='start')
phase_axis.set_xlabel('mean criminality')
phase_axis.set_ylabel('density of security forces')
phase_axis.set_title('Crime and security spiral into a balance')
phase_axis.legend()
time_axis.plot(trajectory.times, trajectory.values('mean_criminality'), label='with security forces')
time_axis.plot(trajectory.times, unchecked_criminality, label='without deterrence or recruitment')
time_axis.set_xlabel('time')
time_axis.set_ylabel('mean criminality')
time_axis.set_title('Security forces hold crime down')
time_axis.legend()
fig.tight_layout()
plt.show()
