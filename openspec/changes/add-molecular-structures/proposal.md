# Proposal

## Why

The survey of Raman (2021) and Konieczny et al. (2023) recorded three models as out of scope, because each maps a
sequence to a structure, or scores a structure, rather than integrating an equation in time:
- genotype spaces and neutral networks of RNA folds (Raman 2021 §13.2–13.3), where each RNA sequence folds into a
  secondary structure, point mutations connect sequences, and robustness and evolvability are counted over the
  neighbours of a sequence;
- the fuzzy oil drop model (Konieczny et al. 2023 §1.3.1, Eqs. 1.1–1.9), which compares the hydrophobicity a protein's
  residues observe (Levitt's function, Eq. 1.2) with an idealised 3D Gaussian (Eq. 1.1) by divergence entropy, and
  summarises the comparison as the relative distance `RD` and the environment factor `K`;
- the early-stage folding intermediate (Konieczny et al. 2023 §3.7.1, Eq. 3.1), which limits each residue's dihedral
  angles to an ellipse path on the Ramachandran map.

Each one becomes a dynamics once it is read through the framework's Monte Carlo form. A neutral walk is a sequence of
point mutations accepted when the fold does not change. A folding simulation is a sequence of dihedral moves accepted
by a score of the whole chain. A structure ensemble is replayed one member per frame. The framework already has
categorical lattice states, moves, and Metropolis and heat-bath samplers, but its energies are local to a site. It has
no chain whose internal coordinates give Cartesian positions, no score of a whole chain or sequence, no structure
files, and no maps from sequences to structures. The same pieces serve structural biology, evolution on
genotype-phenotype maps, and polymer physics.

## What Changes

- Add sequences: categorical states over a declared alphabet along the last axis of a lattice, one row per copy, with
  point mutation moves.
- Add chains: elements on a one-dimensional lattice whose sites hold internal coordinates, bond lengths, bond angles,
  and dihedral angles, and whose Cartesian positions are derived from them, with dihedral rotation moves.
- Add global scores: Monte Carlo acceptance by the change of a function of a whole chain or sequence, including
  temperature zero, which accepts only moves that do not raise the score. Neutral walks and adaptive walks are the
  temperature zero case.
- Add sequence-to-structure maps: memoised functions from a sequence to a phenotype key and numeric features. The
  first is RNA secondary-structure folding, with the minimum free energy, the base-pair distance to a target, and a
  plane layout.
- Add structures: reading and writing PDB and mmCIF files, models of an ensemble, residues, atoms, effective atoms,
  and hydrophobicity scales, and replaying an ensemble or a trajectory file one member per frame.
- Add analyses: neutral networks of a map, enumerated or sampled, genotype and phenotype robustness and evolvability,
  dihedral angles, radius of gyration, end-to-end distance, RMSD after superposition, and the fuzzy oil drop
  distributions with `RD` and `K`.
- Add a chain visual for Rerun: sites as points and bonds as line strips in three dimensions, or in the plane for a
  secondary-structure layout, coloured by a site value.
- Add the models `rna_neutral_networks`, `fuzzy_oil_drop`, and `early_stage_folding_intermediate`, with gallery
  examples, and move their inventory entries from out of scope to pending this change.
- Add the optional extras `rna`, for ViennaRNA, and `trajectories`, for MDTraj, with a named error that names the
  extra when it is missing.

## Capabilities

### New Capabilities

- `molecular-structures`: sequences and chains as lattice states, global scores, sequence-to-structure maps,
  structure files and ensembles, and their analyses.

### Modified Capabilities

- `model-library`: genotype-phenotype and protein structure models.
- `rerun-visualization`: chains drawn as points and bonds.

## Impact

- `skcomplex.core.model` gains `Chain`, `Score`, and the `mutation` move. `Sampler` accepts a temperature of zero.
- `skcomplex.core.backends` gains `Structure`, `read_structure`, `write_structure`, `read_trajectory`, `rna_fold`, and
  `rna_layout`. `skcomplex.core.analysis` gains the neutral network, chain geometry, and fuzzy oil drop functions.
  `skcomplex.core.viz` gains `ChainVisual`.
- Biopython becomes a runtime dependency. ViennaRNA goes in the extra `rna` and MDTraj in the extra `trajectories`.
- `skcomplex.core.base` gains `MissingExtraError` unless `add-constraint-based-models` has added it first.
- The models register under `biology/evolution` and `biology/systems_biology`, and `docs/overview/inventory.toml`
  records them as pending this change.
- Uses the gallery of `add-model-gallery`.
