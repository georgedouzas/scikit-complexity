"""
# Protein corona on a nanoparticle

Albumin, the most abundant blood protein, covers the nanoparticles within seconds, and lipoprotein, which binds more
tightly, displaces it over the next hours: the Vroman effect.
"""

# %%
# The model is created by name from the catalog with the book's concentrations of nanoparticles and of albumin,
# high-density lipoprotein, and fibrinogen, and runs for three hours. Free albumin is a hundred thousand times more
# concentrated than the corona, so the blueprint shows only the nanoparticles bound to each protein, in one time series
# view over the whole run.

import matplotlib.pyplot as plt
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import ScipyODE
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'nanoparticle_corona'
PROTEINS = ('albumin', 'lipoprotein', 'fibrinogen')

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
corona = rrb.TimeSeriesView(
    origin='/',
    contents=[f'/{name}_corona/**' for name in PROTEINS],
    name='corona',
    defaults=[rr.SeriesLines.from_fields(widths=2.5)],
    axis_x=rrb.TimeAxis(view_range=whole_run),
)
blueprint = rrb.Blueprint(
    corona,
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
system = create(MODEL)
solver = ScipyODE(method='LSODA', rtol=1e-8, atol=1e-12)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(system, dt=60.0, solver=solver, observers=[observer]).run(until=10800)

# %%
# Composition of the corona and its equilibrium
# ---------------------------------------------
# Every complex tends to the mass-action equilibrium `B_i = n_i alpha_i U P_i / delta_i`. Lipoprotein has the largest
# affinity `n_i alpha_i P_i / delta_i`, so it should end up holding nearly all the nanoparticles, after albumin held
# most of them first. A run of ten hours with a logarithmic time axis shows both phases.

long_run = Simulation(create(MODEL), dt=10.0, solver=solver).run(until=36000).trajectory_
fig, (shares, balance) = plt.subplots(1, 2, figsize=(10, 4))
total = 36.0
hours = long_run.times[1:] / 3600
for name in PROTEINS:
    shares.semilogx(hours, long_run[f'{name}_corona.n'][1:] / total, label=name)
shares.set_xlabel('time (hours)')
shares.set_ylabel('share of the nanoparticles')
shares.set_title('Lipoprotein displaces albumin')
shares.legend()
sites = {'albumin': 4 * 40**2 / 5**2, 'lipoprotein': 4 * 39**2 / 4**2, 'fibrinogen': 4 * 43.3**2 / 8.3**2}
rates = {'albumin': (2.4e3, 2e-3), 'lipoprotein': (3e4, 3e-5), 'fibrinogen': (2e3, 2e-3)}
free = long_run['nanoparticle.n'][-1]
simulated, predicted = [], []
for name in PROTEINS:
    association, dissociation = rates[name]
    simulated.append(long_run[f'{name}_corona.n'][-1])
    predicted.append(sites[name] * association * 1e-9 * free * long_run[f'{name}.n'][-1] / dissociation)
balance.loglog(predicted, simulated, 'o')
balance.loglog(predicted, predicted, ':', color='black', label='equilibrium')
for name, x, y in zip(PROTEINS, predicted, simulated, strict=True):
    balance.annotate(name, (x, y))
balance.set_xlabel('mass-action equilibrium (nM)')
balance.set_ylabel('simulated complex after 10 h (nM)')
balance.set_title('The corona reaches its equilibrium')
balance.legend()
fig.tight_layout()
plt.show()
