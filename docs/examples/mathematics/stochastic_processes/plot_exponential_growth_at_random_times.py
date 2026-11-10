"""
# Exponential growth at random times

A quantity grows exponentially, `x0 exp(mu t)`, and is observed and restarted at times drawn from an exponential
distribution of rate `lam`. Two exponentials make a power law: the observed values are distributed as
`x^(-lam / mu - 1)`, with no scale of their own.
"""

# %%
# Two thousand samples grow at the rate `mu = 1` and are observed at the rate `lam = 1.5` for 20 time units. The system
# logs the observations and the group size, and the values of a few samples are logged as time series, which the
# blueprint shows.

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.mathematics.stochastic_processes import ExponentialGrowthAtRandomTimesSystem

SYSTEM = 'exponential_growth_at_random_times'

whole_run = rrb.TimeRange(start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(origin='/samples', name='log x of five samples', axis_x=rrb.TimeAxis(view_range=whole_run)),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
trajectory = simulate(
    ExponentialGrowthAtRandomTimesSystem(n_samples=2000),
    dt=0.05,
    observers=[observer],
    random_state=0,
)
x = trajectory.get_values('samples.x')
for index in range(5):
    observer.recording_.send_columns(
        f'samples/sample_{index}',
        indexes=[rr.TimeColumn('sim_time', duration=trajectory.times)],
        columns=rr.Scalars.columns(scalars=np.log(x[:, index])),
    )
observer.recording_.flush()

# %%
# Power-law tails
# ---------------
# The complementary distribution of the values at the end of the run should fall as `x^(-lam / mu)`, a straight line
# of slope `-lam / mu` on logarithmic axes, steeper when the observations come faster than the growth.

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for mu, lam in ((1.0, 0.8), (1.0, 1.5), (1.0, 3.0)):
    run = simulate(ExponentialGrowthAtRandomTimesSystem(n_samples=4000, mu=mu, lam=lam), random_state=0)
    values = np.sort(run.get_values('samples.x')[-1])
    survival = 1 - np.arange(values.size) / values.size
    (line,) = ax.loglog(values, survival, label=f'lam / mu = {lam / mu:g}')
    ax.loglog(values, values ** (-lam / mu), ':', color=line.get_color())
ax.set_xlabel('value x')
ax.set_ylabel('share of samples above x')
ax.set_title('Tails x^(-lam/mu) (dotted)')
ax.set_ylim(1e-4, 1.2)
ax.legend()
plt.show()
