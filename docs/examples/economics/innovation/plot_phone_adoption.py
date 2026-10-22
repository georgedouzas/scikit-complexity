"""
# Phone adoption

Hamill and Gilbert's model of how British households took up fixed-line phones from 1951. A household buys a phone
once its income passes a threshold, but only if a neighbour already has one to call. Incomes grow every year, so the
phones spread outwards from the rich upper-class quarter in an S-shaped adoption curve, lagging behind the households
that could already afford one.
"""

# %%
# Nine hundred households live on a 30 by 30 torus, the upper and the lower class each in one quadrant and the middle
# class in the other two. Incomes grow by 2 percent a year, and a phone costs an income of 1.2. The blueprint shows
# the households in 1969, eighteen years in, when the phones (yellow) have filled the upper-class quarter and are
# spreading through the middle class.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create, describe

MODEL = 'phone_adoption'
THRESHOLD, YEARS = 1.2, 50

observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=1)
model_system = create(MODEL, random_state=0)
simulation = Simulation(
    model_system,
    observers=[observer],
    random_state=0,
    observables=describe(MODEL).observables_of(model_system),
).run(until=18)

# %%
# Adoption against affordability
# ------------------------------
# Over fifty years, a household needs both the income and a neighbour with a phone, so in every class the share with
# a phone should rise along an S-curve below the share that can afford one, starting with the upper class, and the lag
# should be longest for the lower class, whose quarter the phones reach last.

model_system = create(MODEL, random_state=0)
trajectory = (
    Simulation(model_system, random_state=0, observables=describe(MODEL).observables_of(model_system))
    .run(until=YEARS)
    .trajectory_
)
frames = range(len(trajectory.times))
households = [trajectory.agents('households', frame).values for frame in frames]
classes = households[0]['social_class']
fig, (by_class, overall) = plt.subplots(1, 2, figsize=(10, 4))
for index, name in enumerate(np.unique(classes)):
    members = classes == name
    owners = [values['phone'][members].mean() for values in households]
    affording = [(values['income'][members] >= THRESHOLD).mean() for values in households]
    by_class.plot(1951 + trajectory.times, owners, color=f'C{index}', label=f'class {int(name)} with a phone')
    by_class.plot(1951 + trajectory.times, affording, '--', color=f'C{index}', lw=0.8)
by_class.set_xlabel('year')
by_class.set_ylabel('share of households')
by_class.set_title('Phones (solid) lag affordability (dashed)')
by_class.legend(fontsize=8)
overall.plot(1951 + trajectory.times, trajectory['share'], label='with a phone')
overall.plot(
    1951 + trajectory.times,
    [(values['income'] >= THRESHOLD).mean() for values in households],
    '--',
    label='can afford one',
)
overall.set_xlabel('year')
overall.set_ylabel('share of all households')
overall.set_title('An S-shaped adoption curve')
overall.legend()
fig.tight_layout()
plt.show()
