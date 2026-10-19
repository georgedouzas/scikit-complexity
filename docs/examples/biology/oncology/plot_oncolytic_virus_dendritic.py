"""
# Oncolytic virus and dendritic cells

The virus infects and lyses most of the tumour within days, but the T cells that the injected dendritic cells raise
dilute the infection, and the tumour regrows past its initial volume.
"""

# %%
# The model is created by name from the catalog with the book's rates of virus and dendritic cells and a tumour of
# 60 mm³, and runs for 20 days with the stiff solver of its catalog entry. The virions and the T cells number in the
# billions and millions, so the blueprint shows only the uninfected and infected tumour volumes, in one time series
# view over the whole run.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'oncolytic_virus_dendritic'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
volumes = rrb.TimeSeriesView(
    origin='/',
    contents=['/site/AU', '/site/AV'],
    name='tumour volumes',
    defaults=[rr.SeriesLines.from_fields(widths=2.5)],
    axis_x=rrb.TimeAxis(view_range=whole_run),
)
blueprint = rrb.Blueprint(
    volumes,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
solver = ScipyODE(method='LSODA')
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=2)
simulation = Simulation(system, dt=0.1, solver=solver, observers=[observer]).run(until=20)

# %%
# Each therapy alone and both together
# ------------------------------------
# Untreated, the tumour grows as `60 exp(r1 t)`. The virus alone clears it. Dendritic cells alone do nothing, since
# the T cells kill only through the infected cells when `a0 = 0`. With both, the tumour should first fall with the
# virus alone and then regrow, because the T cells raise the total `Atot` that divides the infection.

arms = {'untreated': create(MODEL, uv=0.0, ud=0.0), 'virus alone': create(MODEL, ud=0.0), 'both': system}
fig, (volume, cells) = plt.subplots(1, 2, figsize=(10, 4))
for name, arm in arms.items():
    trajectory = Simulation(arm, dt=0.1, solver=solver).run(until=20).trajectory_
    volume.semilogy(trajectory.times, (trajectory['site.AU'] + trajectory['site.AV']).clip(1e-6), label=name)
times = simulation.trajectory_.times
volume.semilogy(times, 60 * np.exp(0.3198 * times), ':', color='black', label='60 exp(r1 t)')
volume.set_ylim(1e-6, 1e6)
volume.set_xlabel('time (days)')
volume.set_ylabel('tumour volume (mm³)')
volume.set_title('The virus clears the tumour, the T cells dilute it')
volume.legend()
cells.semilogy(times, simulation.trajectory_['site.V'], label='virions')
cells.semilogy(times, simulation.trajectory_['site.E'], label='activated T cells')
cells.set_xlabel('time (days)')
cells.set_ylabel('count')
cells.set_title('Virions and T cells under both therapies')
cells.legend()
fig.tight_layout()
plt.show()
