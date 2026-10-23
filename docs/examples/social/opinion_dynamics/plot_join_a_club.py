"""
# Join-a-club spins

Agents decide to be in a club (yellow) or out of it (purple) by a heat bath in the field of their acquaintances,
while acquaintances across the club boundary dissolve and reform on one side. At coupling 0.5 the members and the
outsiders end in separate communities with few acquaintances between them.
"""

# %%
# The 200 agents start on an Erdős-Rényi graph of mean degree 6 at inverse temperature 0.5, reconsider their choice
# with probability 0.1 per step, and rewire acquaintances faster. The graph view places the agents with its force
# layout and colors them by their choice.

import matplotlib.pyplot as plt
import numpy as np
import rerun.blueprint as rrb

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'join_a_club'

system = create(MODEL)
# The graph view lays out the nodes of a changing network by forces, which these settings keep compact and centred.
blueprint = rrb.Blueprint(
    rrb.GraphView(
        origin='/voters/opinion',
        name='opinion',
        background=rrb.archetypes.GraphBackground(color=[40, 40, 40]),
        visual_bounds=rrb.VisualBounds2D(x_range=(-330, 330), y_range=(-330, 330)),
        force_link=rrb.archetypes.ForceLink(distance=40.0),
        force_many_body=rrb.archetypes.ForceManyBody(strength=-60.0),
        force_position=rrb.archetypes.ForcePosition(strength=0.05),
    ),
    rrb.TimePanel(state='collapsed', timeline='sim_time', play_state='following'),
    rrb.TopPanel(state='hidden'),
    collapse_panels=True,
)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint=blueprint)
simulation = Simulation(system, observers=[observer], random_state=0).run(until=150)

# %%
# Order against coupling
# ----------------------
# Without coupling the share of edges across the club boundary should stay near its value `2 m (1 - m)` for random
# acquaintances, lowered by the rewiring, and it should fall to zero as the coupling grows, when the network splits
# into a club and its outsiders, while the share of members stays near its initial value.

betas = np.linspace(0.0, 0.6, 7)
order, crossing, mixed = [], [], []
for beta in betas:
    run = Simulation(create(MODEL, beta=beta), dt=10.0, random_state=2).run(until=150).trajectory_
    members = run['voters.opinion'][-1]
    snapshot = run.network(-1)
    order.append(abs(2 * members.mean() - 1))
    crossing.append(np.mean(members[snapshot.sources] != members[snapshot.targets]))
    mixed.append(2 * members.mean() * (1 - members.mean()))
fig, axis = plt.subplots(figsize=(6, 4))
axis.plot(betas, order, 'o-', label='imbalance |2 m - 1|')
axis.plot(betas, crossing, 's-', label='edges across the boundary')
axis.plot(betas, mixed, 'k--', label='2 m (1 - m) for random edges')
axis.set_xlabel('inverse temperature β')
axis.set_title('Coupling separates the club from its outsiders')
axis.legend()
fig.tight_layout()
plt.show()
