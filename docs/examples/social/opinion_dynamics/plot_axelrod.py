"""
# Axelrod culture dissemination

Villages on a lattice hold cultures of `F` features with `q` possible traits each. Neighbours who share more features
are more likely to interact, and an interaction makes them share one more. With few traits the whole lattice converges
to one culture, while with many traits neighbours too often share nothing and the lattice freezes into a mosaic of
small cultural regions.
"""

# %%
# A 20 by 20 lattice of villages with `F = 3` features and `q = 5` traits starts from random cultures. Every step,
# each village picks a random one of its four neighbours and, with probability equal to the fraction of features they
# share, copies one trait on which they differ. The default view colours the villages by their culture code.

import matplotlib.pyplot as plt

from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system
from skcomplex.systems.social.opinion_dynamics import culture_observables

SYSTEM = 'axelrod'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=5)
system = create_system(SYSTEM, width=20, height=20, features=3, traits=5, random_state=0)
observables = culture_observables(20, 20)
simulation = Simulation(system, observers=[observer], observables=observables, random_state=0)
simulation.prepare().run(until=150)

# %%
# Order-disorder transition
# -------------------------
# Castellano, Marsili, and Vespignani found that the share of the largest cultural region in the absorbing state drops
# from nearly one to nearly zero as `q` crosses a critical value `q_c`, which grows with `F`. On a small lattice the
# drop is smoothed, but the final share of the largest region for `F = 3` should fall from most of the lattice at
# small `q` to a few percent at large `q`.

traits = [2, 4, 8, 16, 32]
shares = []
for q in traits:
    run = Simulation(
        create_system(SYSTEM, width=8, height=8, features=3, traits=q, random_state=0),
        observables=culture_observables(8, 8),
        random_state=0,
    ).prepare()
    shares.append(run.run(until=300).trajectory_.values('largest_region')[-1])
fig, (series, transition) = plt.subplots(1, 2, figsize=(10, 4))
series.plot(simulation.trajectory_.times, simulation.trajectory_.values('largest_region'))
series.set_xlabel('step')
series.set_ylabel('share of the largest region')
series.set_title('Growth of the largest region, F = 3, q = 5')
transition.semilogx(traits, shares, 'o-')
transition.set_xticks(traits, [str(q) for q in traits])
transition.set_ylim(0, 1.05)
transition.set_xlabel('traits per feature q')
transition.set_ylabel('final share of the largest region')
transition.set_title('Order for few traits, disorder for many (8 by 8)')
fig.tight_layout()
plt.show()
