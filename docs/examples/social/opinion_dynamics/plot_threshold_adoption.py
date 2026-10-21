"""
# Threshold adoption

People scattered over a torus talk only to those within their social circle. An innovation starts with the few most
eager adopters and spreads either by infection, where one adopter in the circle is enough, or by influence, where the
share of adopters in the circle must exceed a personal threshold. Infection reaches everyone once the circles link up
into one network, a percolation transition in the size of the circles, while thresholds stall the spread early.
"""

# %%
# Three hundred people live on a torus of side 175 with circles of radius 9, below the percolation threshold, and the
# innovation spreads by infection from the 2.5 percent with the lowest thresholds. The blueprint shows the people
# after 30 steps on a grey background, adopters in yellow: whole linked groups have adopted while the people in
# groups without an innovator, in purple, never heard of it.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'threshold_adoption'
N_AGENTS, SIZE, STEPS = 300, 175.0, 30

blueprint = rrb.Blueprint(
    rrb.Spatial2DView(origin='/people', name='people', background=[110, 110, 110]),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint, interval=1)
simulation = Simulation(create(MODEL, reach=9.0, random_state=0), observers=[observer], random_state=0).run(until=STEPS)

# %%
# Percolation of the social circles
# ---------------------------------
# A circle of radius `r` holds on average `k = N pi r^2 / L^2` others. Random disks link into a spanning network at the
# critical mean degree `k_c ~ 4.51` of continuum percolation, so by infection the final share of adopters should jump
# from the few innovators to everyone around `k_c`. By influence, adopters must outnumber a threshold share of each
# circle, and the spread should stall at a small share for every circle size.

reaches = np.linspace(4.0, 30.0, 14)
degrees = N_AGENTS * np.pi * reaches**2 / SIZE**2
fig, (series, sweep) = plt.subplots(1, 2, figsize=(10, 4))
trajectory = simulation.trajectory_
series.plot(
    trajectory.times,
    [trajectory.agents('people', frame).values['adopted'].mean() for frame in range(STEPS + 1)],
)
series.set_xlabel('step')
series.set_ylabel('share of adopters')
series.set_title('Infection with circles of radius 9')
for mode, color in (('infection', 'C0'), ('influence', 'C1')):
    finals = []
    for reach in reaches:
        finals.append(
            np.mean(
                [
                    Simulation(create(MODEL, reach=reach, mode=mode, random_state=seed), random_state=seed)
                    .run(until=STEPS)
                    .trajectory_.agents('people', STEPS)
                    .values['adopted']
                    .mean()
                    for seed in range(3)
                ],
            ),
        )
    sweep.plot(degrees, finals, 'o-', color=color, label=mode)
sweep.axvline(4.51, color='grey', linestyle='--', label='k_c ~ 4.51')
sweep.set_xscale('log')
sweep.set_xlabel('mean number of others in a circle k')
sweep.set_ylabel('final share of adopters')
sweep.set_title('Infection percolates, influence stalls')
sweep.legend()
fig.tight_layout()
plt.show()
