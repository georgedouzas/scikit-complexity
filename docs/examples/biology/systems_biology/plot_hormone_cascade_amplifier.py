"""
# Hormone cascade amplifier

A hormone that reaches a liver cell starts a cascade of enzymes, each of which activates many molecules of the next:
the G protein switches on adenylate cyclase, which makes cyclic AMP, which activates protein kinase A, phosphorylase
kinase, and finally glycogen phosphorylase. The G protein switches itself off by hydrolysing its GTP, a timer that ends
the signal soon after the hormone is gone. The time series shows active phosphorylase: one tall, smooth pulse that
rises after the one-second hormone pulse and decays back to zero within the run.
"""

# %%
# The model is created by name from the catalog: a hormone pulse of height 1 from `t = 1` to `t = 2`, the GTP hydrolysis
# rate `k_gtp = 0.5`, and the gains `alpha / beta` of 10, 50, 2, 10, and 10 of the five stages. The view shows the
# active phosphorylase over the whole run, and a second view shows the active G protein.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'hormone_cascade_amplifier'
STAGES = ('g_protein', 'adenylate_cyclase', 'camp', 'protein_kinase_a', 'phosphorylase_kinase', 'phosphorylase')

# The views show the whole run, not the default window that follows the latest frame.
whole_run = rrb.TimeAxis(
    view_range=rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()),
)
blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.TimeSeriesView(origin='/phosphorylase/x', name='active phosphorylase', axis_x=whole_run),
        rrb.TimeSeriesView(origin='/g_protein/g', name='active G protein', axis_x=whole_run),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=0.2, observers=[observer], random_state=0).run(until=30)

# %%
# Amplification and the G-protein timer
# -------------------------------------
# Every stage is activated in proportion to the stage above it and inactivated at a constant rate, so over the whole
# signal the integral of every stage is the integral of the stage above it times its gain `alpha / beta`. The integral
# of phosphorylase should be 100,000 times that of the G protein. Blocking the timer, as cholera toxin does by locking
# the G protein in its active state, is modelled by a ten times slower hydrolysis: the signal should last longer and
# reach a higher peak.

precise = ScipyODE(rtol=1e-9, atol=1e-12)
runs = {
    k_gtp: Simulation(create(MODEL, k_gtp=k_gtp), dt=0.05, solver=precise).run(until=60).trajectory_
    for k_gtp in (0.5, 0.05)
}
trajectory = runs[0.5]
integrals = [
    np.trapezoid(trajectory[f'{stage}.{"g" if stage == "g_protein" else "x"}'], trajectory.times) for stage in STAGES
]

fig, (stages, timer) = plt.subplots(1, 2, figsize=(10, 4))
stages.semilogy(range(len(STAGES)), integrals, 'o-', label='integral of the active form')
stages.semilogy(
    range(len(STAGES)),
    integrals[0] * np.cumprod([1, 10, 50, 2, 10, 10]),
    'k--',
    label='product of the gains',
)
stages.set_xticks(range(len(STAGES)), ['G', 'AC', 'cAMP', 'PKA', 'PhK', 'Ph'])
stages.set_xlabel('stage of the cascade')
stages.set_ylabel('integral over the signal')
stages.set_title(f'Amplification {integrals[-1] / integrals[0]:,.0f}')
stages.legend()
for k_gtp, label in [(0.5, 'k_gtp = 0.5'), (0.05, 'k_gtp = 0.05, toxin')]:
    timer.plot(runs[k_gtp].times, runs[k_gtp]['phosphorylase.x'], label=label)
timer.set_xlabel('time')
timer.set_ylabel('active phosphorylase')
timer.set_title('A slower timer prolongs the signal')
timer.legend()
fig.tight_layout()
plt.show()
