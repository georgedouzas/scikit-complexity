"""
# DNA computation

In 1994 Adleman solved a Hamiltonian path problem in a test tube. Every node of a directed graph was an oligonucleotide
and every link an oligo that joins the end of one node to the start of the next, so ligation built random paths in
massive parallel. Filters kept the strands that start and end at the right nodes, have the right length, and visit
every node, and what was left was the solution.
"""

# %%
# The tube holds 2000 strands, which start as edge oligos of Adleman's seven-node graph and meet in random pairs for
# 60 steps. The default blueprint shows the number of strands that pass every filter.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_simulation

SYSTEM = 'dna_computation'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = create_simulation(SYSTEM).set_params(observers=[observer]).prepare().run()

# %%
# The filters
# -----------
# Each filter should leave fewer strands: those from node 0 to node 6, those of seven nodes among them, and those that
# visit every node. The last ones should all spell the only Hamiltonian path, 0, 1, 2, 3, 4, 5, 6.

trajectory = simulation.trajectory_
counts = []
for frame in range(len(trajectory.times)):
    paths = np.asarray(trajectory.agents('strands', frame).values['path'], dtype=int)
    lengths = (paths >= 0).sum(axis=1)
    ends = (paths[:, 0] == 0) & (paths[np.arange(len(paths)), lengths - 1] == 6)
    sized = ends & (lengths == 7)
    covered = sized & np.array([np.unique(path[path >= 0]).size == 7 for path in paths])
    counts.append((ends.sum(), sized.sum(), covered.sum()))
counts = np.array(counts)
fig, ax = plt.subplots(figsize=(7, 4))
for column, label in enumerate(('start at 0 and end at 6', 'and pass 7 nodes', 'and visit every node')):
    ax.plot(trajectory.times, counts[:, column], label=label)
ax.set_yscale('symlog')
ax.set_xlabel('step')
ax.set_ylabel('strands left')
final = np.asarray(trajectory.agents('strands', len(trajectory.times) - 1).values['path'], dtype=int)
solved = {tuple(path) for path, ok in zip(final, covered, strict=True) if ok}
ax.set_title(f'Solutions: {sorted(solved)}')
ax.legend()
plt.show()
