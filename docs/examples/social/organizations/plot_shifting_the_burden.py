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

from skcomplex.core.analysis import LoopAnalysis
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation, create_system

SYSTEM = 'shifting_the_burden'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/burden/capability', name='capability'),
        rrb.TimeSeriesView(origin='/burden/intervention', name='intervention'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
simulation = create_simulation(SYSTEM).set_params(dt=0.5, observers=[observer]).prepare().run(until=60)

# %%
# With and without the quick fix
# ------------------------------
# The loop analysis finds the fast balancing loop of the fix, the slow balancing loop of the fundamental solution, and
# the reinforcing loops through the intervention, in which more intervention means less capability and a larger
# symptom.
# Without the fix, `fix_rate = 0`, the capability grows to the goal on its own.

polarities = LoopAnalysis().prepare(create_system(SYSTEM)).run().polarities_
summary = ', '.join(f'{polarities.count(kind)} {kind}' for kind in ('reinforcing', 'balancing'))

trajectory = simulation.trajectory_
unaided = create_simulation(SYSTEM, fix_rate=0.0).set_params(dt=0.5).prepare().run(until=60).trajectory_
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
