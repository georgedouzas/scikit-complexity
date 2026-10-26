"""
# Shifting the burden

A system falls short of its goal, and an outside intervention relieves the symptom quickly. The system's own
capability would close the gap too, but slowly, and the intervention erodes it as a side effect. The intervention
takes over the whole burden and the capability withers away, an addiction.
"""

# %%
# The system delivers half of its goal on its own and starts without intervention. The run lasts 60 years. The
# blueprint shows the capability above the intervention.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.engine.analysis import loops
from skcomplex.engine.viz import RerunObserver
from skcomplex.models import create, run

MODEL = 'shifting_the_burden'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/burden/capability', name='capability'),
        rrb.TimeSeriesView(origin='/burden/intervention', name='intervention'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = run(MODEL, settings={'until': 60, 'dt': 0.5, 'observers': [observer]})

# %%
# With and without the quick fix
# ------------------------------
# The loop analysis finds the fast balancing loop of the fix, the slow balancing loop of the fundamental solution, and
# the reinforcing loops through the intervention, in which more intervention means less capability and a larger
# symptom.
# Without the fix, `fix_rate = 0`, the capability grows to the goal on its own.

polarities = [loop.polarity for loop in loops(create(MODEL)).loops]
summary = ', '.join(f'{polarities.count(kind)} {kind}' for kind in ('reinforcing', 'balancing'))

trajectory = simulation.trajectory_
unaided = run(MODEL, settings={'until': 60, 'dt': 0.5}, fix_rate=0.0).trajectory_
times = np.asarray(trajectory.times)
fig, (fixed, alone) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axis, scenario, title in ((fixed, trajectory, 'With the quick fix'), (alone, unaided, 'Without the quick fix')):
    axis.plot(times, scenario.values('burden.capability'), label='capability')
    axis.plot(times, scenario.values('burden.intervention'), '--', label='intervention')
    axis.plot(times, scenario.values('burden.symptom'), ':', label='symptom')
    axis.set_xlabel('years')
    axis.set_title(title)
    axis.legend()
fixed.set_ylabel('share of the goal')
fig.suptitle(f'Feedback loops: {summary}')
fig.tight_layout()
plt.show()
