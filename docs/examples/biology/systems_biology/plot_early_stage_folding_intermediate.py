"""
# Early-stage folding intermediate

In the early stage of folding, the structure of a chain depends on its backbone alone, and the dihedral angles of each
residue lie on an ellipse path of the Ramachandran map (Konieczny et al. 2023 §3.7.1). The path links the areas of the
secondary structures, and it is the limited conformational space of the early stage.
"""

# %%
# The residues of BPTI sample the cells of the path by heat bath. The default blueprint shows one row of colors, with
# the cell of every residue at the current frame.

import matplotlib.pyplot as plt
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.viz import RerunObserver
from skcomplex.systems.biology.systems_biology import (
    AMINO_ACIDS,
    EarlyStageFoldingIntermediateSystem,
    EllipsePath,
    read_cell_probabilities,
    read_native_pairs,
    read_structure,
)

SYSTEM = 'early_stage_folding_intermediate'

observer = RerunObserver(SYSTEM, save_path=f'{SYSTEM}.rrd', blueprint='default', interval=1)
trajectory = simulate(EarlyStageFoldingIntermediateSystem(), observers=[observer], random_state=0, until=299)

# %%
# Projection and sampled cells
# ----------------------------
# The first panel moves the native angles of BPTI to the closest point of the path by the shortest periodic distance.
# The second compares, for glycine, the probability of each cell with the frequency of the cell over the last 200
# frames of the first glycine of the chain.

structure = read_structure('5PTI')
names, native = read_native_pairs(structure)
path = EllipsePath().prepare()
projected = path.run(native).angles_
curve = path.read_points(np.linspace(0, 2 * np.pi, 400))
probabilities = read_cell_probabilities()
residues = [residue for _, _, residue in structure.residues]
site = residues.index('GLY')
cells = trajectory.get_lattice('residues.cell').astype(int)[-200:, 0, site]
frequencies = np.bincount(cells, minlength=72) / 200

fig, (ramachandran, sampled) = plt.subplots(1, 2, figsize=(11, 4.5))
ramachandran.plot(*np.degrees(curve).T, color='0.5', label='ellipse path')
ramachandran.scatter(*np.degrees(native).T, s=14, label='native')
ramachandran.scatter(*np.degrees(projected).T, s=14, label='projected')
ramachandran.set_xlabel('phi (degrees)')
ramachandran.set_ylabel('psi (degrees)')
ramachandran.legend()
sampled.bar(np.arange(72), frequencies, color='C1', label='frequency')
sampled.plot(np.arange(72), probabilities[AMINO_ACIDS.index('GLY')], color='C0', label='probability')
sampled.set_xlabel('cell of the path')
sampled.legend()
fig.tight_layout()
plt.show()
