# Proposal

## Why

The inventory records three models of Raman (2021) and Konieczny et al. (2023) as blocked on this change, because each
maps a sequence to a structure, or scores a whole structure, rather than integrating an equation in time:
- `rna_neutral_networks` in `biology/evolution` (Raman 2021 §13.2–13.3): each RNA sequence folds into a secondary
  structure, point mutations connect sequences, and robustness and evolvability are counted over the neighbours of a
  sequence.
- `fuzzy_oil_drop` in `biology/systems_biology` (Konieczny et al. 2023 §1.3.1, Eqs. 1.1–1.9): the hydrophobicity a
  protein's residues observe is compared with an idealised 3D Gaussian by divergence entropy, summarised as the
  relative distance `RD` and the environment factor `K`.
- `early_stage_folding_intermediate` in `biology/systems_biology` (Konieczny et al. 2023 §3.7.1, Eq. 3.1): each
  residue's dihedral angles lie on an ellipse path on the Ramachandran map.

Each one becomes a dynamics once read through Monte Carlo sampling. A neutral walk is a sequence of point mutations
accepted when the fold does not change, a folding simulation is a sequence of dihedral moves accepted by a score of the
whole chain, and a structure ensemble is replayed one member per frame.

The current core already provides most of this. A `LatticeElement` on a `LatticeSpace((copies, length))` holds one
categorical sequence per row, with labels from a `FiniteEnumeratedSet` domain. `SamplingInteraction` samples a lattice
state with Metropolis, Glauber, or heat-bath acceptance, and its `move` may be a function `move(values, random)`, which
gives a point mutation or a dihedral rotation with no new move kind. `DriveLaw` changes a whole lattice state through a
function, `PointsVisual` and `ImageVisual` draw positions and lattices, and `EntropyAnalysis` gives divergences. What is
missing is a score of a whole lattice row with acceptance at temperature zero, a drive that reads the step it runs in,
the geometry analysis of point chains, line strips through points, and a named error for a missing extra. The protein
geometry, structure files, the fuzzy oil drop, RNA folding, and the neutral network analyses are biology and live in the
model packages that use them.

## What Changes

- Add `RowScoreSpec(function, proposals=1)` to `skcomplex.specs`, which `SamplingInteraction` takes as its `objective`: a
  function of the whole row of a lattice state and the target's parameters that returns one number, accepted by the
  change of that number per row. Let the intensity of `SamplingInteraction` be infinite, the zero temperature of its thin
  subclass in physics, which accepts only moves that do not raise the objective.
- Let the function of `DriveLaw` take the keyword `time`, the time of the step, when it names it, so a drive replays one
  member of an ensemble per step.
- Add `ShapeAnalysis` to `skcomplex.analysis`, with the kinds `'gyration'`, `'end-to-end'`, and `'rmsd'`, the last after
  optimal superposition through SciPy's `Rotation.align_vectors`.
- Add the setting `strip` to `PointsVisual`, a line strip through the points in member order.
- Add `MissingExtraError` to `skcomplex.errors`, raised by a model whose optional extra is not installed, unless
  `add-constraint-based-models` adds it first.
- Add to the package `skcomplex.models.biology.systems_biology`: structure files and ensembles over Biopython, the
  protein backbone geometry, effective atoms, the ellipse path, `FuzzyOilDropAnalysis`, `BackboneVisual`, and the
  models `fuzzy_oil_drop` and `early_stage_folding_intermediate`, with the PDB entries 1L2Y, 1UBQ, and 5PTI bundled.
- Add to the package `skcomplex.models.biology.evolution`: RNA folding over ViennaRNA, `NeutralNetworkAnalysis`, and the
  model `rna_neutral_networks`.
- Add the extras `structures`, for Biopython, `rna`, for ViennaRNA, and `trajectories`, for MDTraj. Biopython 1.88
  publishes no macOS x86_64 wheels, and only the two protein models use it, so it stays out of the default install.

## Capabilities

### New Capabilities

- `molecular-structures`: row scores and zero temperature in sampling, drives that read the step, chain geometry, and
  the named error of a missing extra.

### Modified Capabilities

- `model-library`: the three models, their structures, maps, and analyses in their packages.
- `rerun-visualization`: line strips through points.

## Impact

- `skcomplex.specs` gains `RowScoreSpec`, and `SamplingInteraction` accepts it and an infinite intensity.
- `skcomplex.core.laws.DriveLaw` passes `time` to a function that names it.
- `skcomplex.analysis` gains `ShapeAnalysis`, and `skcomplex.engine.viz.PointsVisual` gains `strip`.
- `skcomplex.errors` gains `MissingExtraError`.
- `skcomplex.models.biology.systems_biology` and `skcomplex.models.biology.evolution` gain the models and their
  realizations, and `docs/overview/inventory.toml` marks the three entries implemented.
- `pyproject.toml` gains the extras `structures = ["biopython>=1.88"]`, `rna = ["ViennaRNA>=2.7.2"]`, and
  `trajectories = ["mdtraj>=1.11.2"]`. The Project Profile lists them.
