"""
# Voter model

Starting from random opinions on a square lattice, each voter copies the opinion of a random neighbour. Agreement
spreads locally, so the opinions coarsen into growing domains of purple and yellow voters.
"""

# %%
# The voters sit on the nodes of a 20 by 20 lattice. The nodes keep their lattice coordinates as their `pos`
# attribute, so the graph view draws them on the lattice. The blueprint gives the graph view a grey background, on
# which both opinions show.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'voter_model'

lattice = nx.convert_node_labels_to_integers(nx.grid_2d_graph(20, 20), label_attribute='pos')
system = create(MODEL, graph=lattice, random_state=0)
blueprint = rrb.Blueprint(
    rrb.GraphView(origin='/voters', name='voters', background=rrb.archetypes.GraphBackground(color=[110, 110, 110])),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=60)

# %%
# Magnetization and interface density
# -----------------------------------
# The voter model conserves the mean opinion on average, so the fraction of voters with opinion 1 should wander
# around its initial value. On a square lattice the share of edges joining voters who disagree, the interface
# density, should decay slowly as the domains coarsen.

edges = np.array(lattice.edges)
fractions, interfaces = [], []
for frame in range(len(simulation.trajectory_.times)):
    voters = simulation.trajectory_.agents('voters', frame)
    opinion = np.empty(len(voters))
    opinion[np.ravel(voters.locations).astype(int)] = voters.values['opinion']
    fractions.append(opinion.mean())
    interfaces.append(np.mean(opinion[edges[:, 0]] != opinion[edges[:, 1]]))
fig, (share, interface) = plt.subplots(1, 2, figsize=(10, 4))
share.plot(simulation.trajectory_.times, fractions)
share.set_ylim(0, 1)
share.set_xlabel('step')
share.set_ylabel('fraction with opinion 1')
share.set_title('The mean opinion is conserved on average')
interface.loglog(simulation.trajectory_.times[1:], interfaces[1:])
interface.set_xlabel('step')
interface.set_ylabel('interface density')
interface.set_title('Domains coarsen')
fig.tight_layout()
plt.show()
