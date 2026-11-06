"""
# Rice food balance

Land suitable for rice is converted into rice fields until it runs out, while the fields lose a small fraction every
year to housing, roads, and industry. The population keeps growing, so the food security level, the rice available
over the rice required, rises while new land is cultivated and then falls.
"""

# %%
# The region starts with 68000 hectares of rice out of 120000 suitable, a planning time of 15 years, and 2.8 million
# people, for 100 years. The blueprint shows the area cultivated above the population.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'rice_food_balance'

blueprint = rrb.Blueprint(
    rrb.Vertical(
        rrb.TimeSeriesView(origin='/region/area_cultivated', name='area cultivated'),
        rrb.TimeSeriesView(origin='/region/population', name='population'),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint=blueprint, interval=1)
simulation = create_simulation(SYSTEM).set_params(dt=0.25, observers=[observer]).prepare().run(until=100)

# %%
# Food security against time
# --------------------------
# The area cultivated approaches the balance `L / (1 + T f)` between the conversion of the remaining land and the
# land lost, and the population grows exponentially at the birth fraction minus one over the life expectancy. The food
# security should rise for about twenty years, while the area grows faster than the population, and then fall below
# its starting level, as in Fig. 4.21 of Bala et al.

trajectory = simulation.trajectory_
times = np.asarray(trajectory.times)
fig, (land, security) = plt.subplots(1, 2, figsize=(10, 4))
land.plot(times, trajectory.values('region.area_increase_rate'), label='area increase rate')
land.plot(times, trajectory.values('region.land_lost'), '--', label='land lost')
land.set_xlabel('years')
land.set_ylabel('hectares per year')
land.set_title('The flows of rice land')
land.legend()
security.plot(times, trajectory.values('region.food_security'))
security.axhline(1.0, color='black', linestyle=':', lw=0.8)
security.set_xlabel('years')
security.set_ylabel('food available / food required')
security.set_title('The food security level')
fig.tight_layout()
plt.show()
