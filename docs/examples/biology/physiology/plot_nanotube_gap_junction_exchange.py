"""
# Nanotube and gap junction exchange

A hundred cells in ten modules exchange a substance with their neighbours through gap junctions, metabolize it, and
pass extra substance through tunnelling nanotubes. With nanotubes 20% stronger than the critical strength, the
exchange is no longer stable: the substance accumulates in the densest module and spreads to its neighbours, while
it fades elsewhere. The view shows the network at the end of the run, bright in the module where the substance grows
and dark where it has faded.
"""

# %%
# The model is created by name from the catalog on a modular Erdős-Rényi network of 100 cells like the default, with
# the metabolism `alpha = 1`, `beta = 10`, and the gap junctions `g = 0.1`. The critical strength `xi_crit` of the
# nanotubes is where the largest eigenvalue of `xi A - g L` reaches `alpha / beta`, found by root finding, and the run
# sets `xi = 1.2 xi_crit`. The observer saves every second frame with the default blueprint, whose primary view is the
# network colored by the concentrations.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import brentq

from skcomplex.core.runtime import Simulation
from skcomplex.core.viz import RerunObserver
from skcomplex.models import create

MODEL = 'nanotube_gap_junction_exchange'

graph = nx.random_partition_graph([10] * 10, 0.297, 0.0033, seed=1)
adjacency = nx.to_numpy_array(graph)
laplacian = np.diag(adjacency.sum(axis=1)) - adjacency


def leading_rate(xi):
    """Return the largest eigenvalue of the linearised exchange around the state without substance."""
    return np.linalg.eigvalsh(xi * adjacency - 0.1 * laplacian).max() - 1.0 / 10.0


critical = brentq(leading_rate, 0.0, 1.0)
system = create(MODEL, graph=graph, xi=1.2 * critical)
observer = RerunObserver(MODEL, save_path=f'{MODEL}.rrd', blueprint='default', interval=2)
simulation = Simulation(system, dt=0.5, observers=[observer], random_state=0).run(until=100)

# %%
# Distance to instability
# -----------------------
# Around the state without substance, the exchange is linear with the Jacobian `-alpha / beta + xi A - g L`. Its
# leading eigenvalue should cross zero at `xi_crit`, and the total substance should decay below it and grow above it at
# that rate, as in the idealized, stable, critical, and unstable cases of Mihailović et al.

strengths = np.linspace(0.0, 1.5 * critical, 31)
fig, (spectrum, totals) = plt.subplots(1, 2, figsize=(10, 4))
spectrum.plot(strengths, [leading_rate(xi) for xi in strengths])
spectrum.axhline(0.0, color='gray', linewidth=0.8)
spectrum.axvline(critical, color='black', linestyle=':', label=f'xi_crit = {critical:.4f}')
spectrum.set_xlabel('nanotube strength xi')
spectrum.set_ylabel('leading eigenvalue of the Jacobian')
spectrum.set_title('Distance to instability')
spectrum.legend()
for share in (0.0, 0.8, 1.0, 1.2):
    run = Simulation(create(MODEL, graph=graph, xi=share * critical, x=0.01), dt=0.5).run(until=100)
    totals.semilogy(run.trajectory_.times, run.trajectory_['cells.x'].sum(axis=1), label=f'xi = {share} xi_crit')
totals.set_xlabel('time')
totals.set_ylabel('total substance')
totals.set_title('Decay below and growth above the threshold')
totals.legend()
fig.tight_layout()
plt.show()
