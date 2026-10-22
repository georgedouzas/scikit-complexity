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

Each one becomes a dynamics once it is read through the framework's Monte Carlo sampling. A neutral walk is a sequence
of point mutations accepted when the fold does not change. A folding simulation is a sequence of dihedral moves
accepted by a score of the whole chain. A structure ensemble is replayed one member per frame. The framework already
has categorical lattice states and the `Sample` law with Metropolis and heat-bath acceptance, but its energies are local
to a site. It has no chain whose internal coordinates give Cartesian positions, no score of a whole chain or sequence,
no structure files, and no maps from sequences to structures. The same general pieces serve structural biology,
evolution on genotype-phenotype maps, and polymer physics.

**Rebased on `redesign-system-model`.** The redesign absorbed `Energy`, `Move`, and `Sampler` into the `Sample` law,
restricted `core.backends` to execution backends, moved observables to simulations and catalog entries, and requires
every specific realization to live in `skcomplex.models`. This change therefore no longer adds a `Score` law: a global
score is an energy of `Sample` over a whole lattice row, and the mutation move is a move kind of `Sample`. The general
pieces stay in the core: chains with a bead geometry, global scores and temperature zero in `Sample`, the replay law,
chain geometry analyses, and the chain visual. The biological pieces move to the library: the protein backbone
geometry, structure files and ensembles, effective atoms, hydrophobicity and the fuzzy oil drop go to
`skcomplex.models.biology.realizations`, and RNA folding, sequence-to-phenotype maps, and neutral network analyses go
to the `biology/evolution` model package.

## What Changes

- Add sequences: categorical states over a declared alphabet in a site group of a lattice of shape `(copies, length)`,
  one row per copy, with the point mutation move kind of `Sample`.
- Add chains: a site group on a one-dimensional `LatticeSpace` whose sites hold internal coordinates, bond lengths,
  bond angles, and dihedral angles, and whose Cartesian positions are derived from them with a generic bead geometry,
  with dihedral rotation moves. The protein backbone geometry is a biology realization.
- Extend `Sample` with global scores: acceptance by the change of a function of a whole lattice row, including
  temperature zero, which accepts only moves that do not raise the score. Neutral walks and adaptive walks are the
  temperature zero case.
- Add the replay law, which sets a state to member `k` of an ensemble at step `k`.
- Add the general chain geometry analyses: radius of gyration, end-to-end distance, and RMSD after superposition.
- Add, in `skcomplex.models.biology.realizations`: reading and writing PDB and mmCIF files, models of an ensemble,
  residues, atoms, effective atoms, hydrophobicity scales, the protein backbone geometry, backbone dihedral and peptide
  plane angles, and the fuzzy oil drop distributions with `RD` and `K`.
- Add, in the `biology/evolution` model package: memoised sequence-to-phenotype maps, RNA secondary-structure folding
  with the minimum free energy, the base-pair distance to a target, and a plane layout, and the neutral network,
  robustness, and evolvability analyses.
- Add a chain visual for Rerun: sites as points and bonds as line strips in three dimensions, or in the plane for a
  secondary-structure layout, coloured by a site value.
- Declare the requirements of chains, global scores, and the replay law, with the validation issues of their misuse.
- Add the models `rna_neutral_networks`, `fuzzy_oil_drop`, and `early_stage_folding_intermediate`, with gallery
  examples, and move their inventory entries from out of scope to pending this change.
- Add the optional extras `rna`, for ViennaRNA, and `trajectories`, for MDTraj, with a named error that names the
  extra when it is missing.

## Capabilities

### New Capabilities

- `molecular-structures`: sequences and chains as lattice states, global scores, replay, chain geometry, and the
  library's sequence-to-structure maps, structure files and ensembles, and their analyses.

### Modified Capabilities

- `model-library`: genotype-phenotype and protein structure models, and the biology realizations they use.
- `rerun-visualization`: chains drawn as points and bonds.

## Impact

- `skcomplex.core.components` gains `Chain`, `skcomplex.core.laws` gains the global score, temperature zero, and the
  `'mutation'` move kind of `Sample`, and the replay law, `skcomplex.core.analysis` gains the chain geometry functions,
  and `skcomplex.core.viz` gains `ChainVisual`.
- `skcomplex.models.biology.realizations` gains `Structure`, `read_structure`, `write_structure`, `read_trajectory`,
  effective atoms, hydrophobicity scales, the protein backbone geometry, and the fuzzy oil drop functions. The
  `biology/evolution` package gains `GenotypeMap`, `rna_fold`, `rna_layout`, and the neutral network analyses.
- Biopython becomes a runtime dependency. ViennaRNA goes in the extra `rna` and MDTraj in the extra `trajectories`.
- `skcomplex.core.base` gains `MissingExtraError` unless `add-constraint-based-models` has added it first.
- The models register under `biology/evolution` and `biology/systems_biology`, and `docs/overview/inventory.toml`
  records them as pending this change.
- Depends on `redesign-system-model`, and uses the gallery of `add-model-gallery`.
