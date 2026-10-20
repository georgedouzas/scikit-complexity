"""
# Bass diffusion

A new product spreads through a market of 1000 people. Potential adopters adopt through advertising, at the rate of
innovation `p`, and through word of mouth, at the rate of imitation `q` times the share of adopters. The adopters
follow an S-shaped curve and the adoption rate rises to a peak before the market saturates.
"""

# %%
# The market starts without adopters, with the mean coefficients `p = 0.03` and `q = 0.38` per year, and evolves for
# 20 years. The blueprint shows the potential adopters above the adopters.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'bass_diffusion'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/market/P', name='potential adopters P'),
        rrb.TimeSeriesView(origin='/market/A', name='adopters A'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL), dt=0.25, observers=[observer]).run(20)

# %%
# The closed-form Bass curve
# --------------------------
# Without initial adopters the adopters are `A(t) = N (1 - exp(-(p + q) t)) / (1 + (q / p) exp(-(p + q) t))`, and the
# adoption peaks at `log(q / p) / (p + q)`, about 6.2 years.

p, q, market = 0.03, 0.38, 1000.0
trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
decay = np.exp(-(p + q) * times)
fig, (adopters, adoption) = plt.subplots(1, 2, figsize=(10, 4))
adopters.plot(times, trajectory['market.A'], label='simulation')
adopters.plot(times, market * (1 - decay) / (1 + q / p * decay), '--', color='black', label='closed form')
adopters.set_xlabel('years')
adopters.set_ylabel('adopters')
adopters.set_title('The S-shaped diffusion')
adopters.legend()
adoption.plot(times, trajectory['market.adoption'])
adoption.axvline(np.log(q / p) / (p + q), color='grey', linestyle=':', label='peak time')
adoption.set_xlabel('years')
adoption.set_ylabel('adopters per year')
adoption.set_title('The adoption flow')
adoption.legend()
fig.tight_layout()
plt.show()
