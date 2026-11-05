"""
# Fuzzy oil drop

The fuzzy oil drop system of Konieczny et al. (2023 §1.3.1) compares the hydrophobicity that the residues of a protein
observe with an idealised micelle, a three-dimensional Gaussian over the chain. The relative distance `RD` compares the
divergence of the observed distribution from the Gaussian and from a uniform distribution, and the factor `K` measures
how far the environment departs from water.
"""

# %%
# The system replays the 38 NMR systems of the Trp-cage 1L2Y, one per frame. The default blueprint shows the 20
# effective atoms of the chain, the mean atom of each residue, joined in sequence order.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.analysis import ShapeAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems import create_system, describe
from skcomplex.systems.biology.systems_biology import BackboneGeometry, FuzzyOilDropAnalysis, read_structure

SYSTEM = 'fuzzy_oil_drop'

system = create_system(SYSTEM)
observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
simulation = Simulation(
    system,
    observers=[observer],
    observables=describe(SYSTEM).observables_of(system),
    random_state=0,
)
trajectory = simulation.prepare().run(until=37).trajectory_

# %%
# Distributions of one system
# --------------------------
# The first panel holds the theoretical and the observed distribution of the first system per residue, with the uniform
# distribution as the line. The second holds `RD` and `K` of every system of the ensemble, with means of 0.77 and 2.2.
# The book expects `RD` below 0.5 and `K` near 0 of a fast-folding protein. It gives no formula for the width of its
# Gaussian, and this difference comes from the width that the system chooses, so `RD` and `K` here do not compare with
# its values.

structure = read_structure('1L2Y')
profile = FuzzyOilDropAnalysis(structure, 'profile').prepare().run()
residues = np.arange(1, 21)
fig, (distributions, ensemble) = plt.subplots(1, 2, figsize=(11, 4))
distributions.plot(residues, profile.theoretical_[0], marker='o', label='T')
distributions.plot(residues, profile.observed_[0], marker='o', label='O')
distributions.plot(residues, profile.uniform_, color='0.5', label='R')
distributions.set_xlabel('residue')
distributions.set_ylabel('share of hydrophobicity')
distributions.legend()
ensemble.plot(trajectory.times, trajectory.values('RD'), label='RD')
ensemble.plot(trajectory.times, trajectory.values('K'), label='K')
ensemble.set_xlabel('NMR system')
ensemble.legend()
fig.tight_layout()
plt.show()

# %%
# Folding from an extended chain
# ------------------------------
# With `dynamics='folding'`, the chain starts extended and each unit of time proposes one rotation of a dihedral angle,
# which temperature zero accepts when the score does not grow, `RD` plus one for every clash. Lowering `RD` also makes
# the chain somewhat more compact: the radius of gyration falls from 19.7 Å to 17.4 Å, far from the 7.3 Å of the first
# NMR system.

folding = create_system(SYSTEM, dynamics='folding')
folded = Simulation(folding, random_state=0).prepare().run(until=200).trajectory_
geometry = BackboneGeometry(structure).prepare()
angles = folded.lattice('chain.angle').astype(float)[:, 0]
chains = np.array([geometry.run(row).read_effective_atoms() for row in angles[::20]])
radii = ShapeAnalysis().prepare(chains).run().values_

fig, (score, size) = plt.subplots(1, 2, figsize=(11, 4))
score.plot(folded.times, folded.values('folding.score')[:, 0])
score.set_xlabel('time')
score.set_ylabel('RD')
size.plot(folded.times[::20], radii, marker='o')
size.set_xlabel('time')
size.set_ylabel('radius of gyration (Å)')
fig.tight_layout()
plt.show()
