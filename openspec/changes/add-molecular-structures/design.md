# Design

## Context

The parts of the current code this change builds on:
- `LatticeElement(label, element=, space=LatticeSpace(shape))` in `skcomplex.core.elements` holds one value per site.
  A state with the domain `FiniteEnumeratedSet(['A', 'C', 'G', 'U'])` is categorical, and `EntrySpec` names its labels.
- `SamplingInteraction(target, state, energy, move, temperature, rule, schedule, neighborhood, values, width)` in
  `skcomplex.core.interactions` lowers to a `SampleLaw` with an `EnergySpec`, a `MoveSpec`, and a `SamplerSpec`, and
  runs on `MonteCarloSolver` under `DiscreteTime`. Its energy is a local function of the namespace, its move may be a
  function `move(values, random)`, and its temperature is a parameter in `(0, oo)`.
- `DriveLaw(target, function)` changes a whole lattice state by `function(values, *, random, **parameters)`.
- `PointsVisual(position, color_by, color_range, trail)` and `ImageVisual` in `skcomplex.engine.viz`.
- `EntropyAnalysis` and `DistributionAnalysis` in `skcomplex.analysis`, and `register_model(field=, observables=)` in
  `skcomplex.models`.
- The errors of `skcomplex.errors`, which hold no error for a missing extra yet.

## Goals / Non-Goals

**Goals:**
- Fit the three models into `System`, `Simulation`, and `Trajectory` without a new runtime: walks and folding are Monte
  Carlo time, and ensembles are replayed one member per frame.
- Keep the core free of molecular biology. A row score is any function of a row, a chain is any sequence of points, and
  a replay is any drive that reads its step.
- Take parsing, internal coordinates, superposition, hydrophobicity scales, RNA folding, and trajectory formats from
  mature libraries.

**Non-Goals:**
- Molecular dynamics with force fields, which `LammpsSolver` covers.
- Tertiary structure prediction, DSSP, RNA partition functions, and RNA design.
- Population genetics on genotype networks.

## Decisions

### Library survey

Checked on 2026-10-19 against PyPI, GitHub, and spike installs on macOS arm64 with Python 3.13:

| Library | Licence | Role | Decision |
|---|---|---|---|
| Biopython 1.88 | permissive, partly BSD-3-Clause | PDB and mmCIF, internal coordinates, superposition, hydrophobicity scales | adopt as a runtime dependency |
| ViennaRNA 2.7.2 | free for research and commercial use, no redistribution for a fee, not OSI approved | MFE folding, base-pair distance, plane layouts | adopt behind the extra `rna` |
| MDTraj 1.11.2 | LGPL-2.1-or-later | DCD, XTC, TRR, NetCDF, and HDF5 trajectories | adopt behind the extra `trajectories` |
| MDAnalysis 2.10 | LGPL-3.0-or-later | trajectories | reject: no manylinux aarch64 wheel, heavier than MDTraj |
| gemmi 0.7.5 | MPL-2.0 | fast mmCIF and PDB I/O | reject: Biopython also gives internal coordinates and superposition |
| seqfold 0.10.2 | MIT | MFE folding | reject: its folds differ from ViennaRNA's, which the literature on neutral networks uses |
| PeptideBuilder 1.1 | MIT | peptides from dihedrals | reject: inactive since 2020, Biopython does the same |
| SciPy | BSD-3-Clause | `Rotation.align_vectors`, bounded minimisation | already a dependency |
| NetworkX | BSD-3-Clause | neutral networks and their components | already a dependency |

Biopython is permissive and needs only NumPy, so it is a runtime dependency, which the Project Profile allows for a
library a model alone uses. ViennaRNA's licence cannot be promised to the users of an MIT package, and MDTraj is LGPL,
so both live behind extras. A model that needs a missing extra raises `MissingExtraError`, which names the model, the
extra, and the install command. The import is lazy with a suppression that gives the reason, as the constitution asks.

The spike measured that ViennaRNA folds `GGGGAAACCCC` into `((((...))))` at −4.5 kcal/mol, that a neutral walk of 8
walkers of length 30 for 300 steps with robustness and evolvability every 10 steps takes 3.0 s, with a Spearman
correlation of robustness and evolvability between −0.69 and −0.77, and that Biopython parses the 38 models of 1L2Y in
0.18 s with an `RD` between 0.69 and 0.85.

### How the models fit the runtime

| Model | A unit of time | Core parts |
|---|---|---|
| `rna_neutral_networks` | one proposed point mutation per walker | `SamplingInteraction` with a function move and a `RowScoreSpec` |
| `fuzzy_oil_drop`, replay | one member of the ensemble | `DriveLaw` reading `time` |
| `fuzzy_oil_drop`, folding | one sweep of dihedral moves | `SamplingInteraction` with a rotation move and a `RowScoreSpec` |
| `early_stage_folding_intermediate` | one heat-bath sweep of path cells | `SamplingInteraction` with the `'heat-bath'` rule and a local energy |

Every model runs under `DiscreteTime()`.

### Row scores

`RowScoreSpec(function, proposals=1)` is a frozen spec of `skcomplex.specs` whose `function(row, **parameters)` takes
the values of one row of a lattice state, the last axis, and the parameters of the target that it names, and returns
one number. `SamplingInteraction` takes it as its `energy` and runs it as a private method of its own: each step it
proposes `proposals` moves per row through its move, evaluates the function on the proposed row, and accepts by its
rule on the difference. The rows are independent copies, each with its own draws from the simulation's stream. The
score of every row is recorded as the observable `<label>.score`.

The temperature domain of `SamplingInteraction` becomes `[0, oo)`. At temperature zero the Metropolis rule accepts a
move only when the energy does not grow, which gives neutral walks, with a score of 0 on the target phenotype and 1 off
it, and adaptive walks, with the base-pair distance. A `RowScoreSpec` whose function is not callable, or whose target is
not on a `LatticeSpace`, raises a `ValidationError` that the compiler reports with the existing code `law-space` or
`value`.

### Drives that read the step

`DriveLaw` passes `time`, the time of the step, to a function that names it in its signature, as it passes the
parameters it names. A replay is then a model function, `lambda values, *, time: ensemble[min(int(time), last)]`, which
holds the last member after the end. No replay law is added.

### Chain geometry

`ShapeAnalysis(kind, group=None, reference=None)` reads the positions of the members of a group from a trajectory, or an
array of frames by points by dimensions:
- `'gyration'` sets `values_`, the radius of gyration per frame.
- `'end-to-end'` sets `values_`, the distance between the first and last point per frame.
- `'rmsd'` sets `values_`, the root mean square deviation from `reference` after optimal superposition with SciPy's
  `Rotation.align_vectors`.

### Visualization

`PointsVisual(strip=True)` logs the positions of a group as one line strip in member order under `<path>/strip`, in two
or three dimensions, beside the points. Sequences use the existing `ImageVisual` with one color per label.

### Where the biology lives

The two protein models share the package `skcomplex.models.biology.systems_biology`, so their realizations live there,
and the RNA pieces live in `skcomplex.models.biology.evolution`:
- `systems_biology` holds the structures, the geometry, and the analysis of the two protein models:
  - `read_structure` and `write_structure` over Biopython's PDB and mmCIF parsers and writers, which return a frozen
    `StructureSpec` with coordinates of shape `(models, atoms, 3)` and the atom, residue, and chain names, and
    `read_trajectory` over MDTraj with the extra.
  - The protein backbone geometry, which writes `phi`, `psi`, and `omega` into Biopython's `IC_Chain` and rebuilds the
    atoms, and effective atoms, the mean atom of each residue.
  - `FuzzyOilDropAnalysis(BaseAnalysis)` with the kinds `'profile'`, the distributions `T`, `O`, and `R`,
    `'relative-distance'`, and `'environment-factor'`, after Eqs. 1.1–1.9.
  - The ellipse path of Brylinski et al. (2004, Bioinformatics 20:199) with its projection.
  - `BackboneVisual(PointsVisual)`, which rebuilds the backbone from the dihedral states of every frame.
  - The bundled entries, gzipped mmCIF files read with `importlib.resources`.
- `evolution`: the fold of a sequence through `RNA.fold_compound`, memoised per sequence, the base-pair distance, and
  the plane layout, and `NeutralNetworkAnalysis(BaseAnalysis)` with the kinds `'network'`, `'genotype-robustness'`,
  `'genotype-evolvability'`, `'phenotype-robustness'`, and `'phenotype-evolvability'`, after Wagner's definitions as
  Raman (2021 §13.3) states them, over any function from a sequence to a phenotype key.

The fuzzy oil drop takes each `σ` as a third of the largest absolute coordinate along its principal axis, which the
book leaves unstated, and the Kyte-Doolittle scale of Biopython rescaled to `[0, 1]` by default. `K` minimises
`D_KL(O|M)` over `[0, 5]` with SciPy's bounded scalar minimiser. The early-stage model samples the cells of the path,
72 by default, with the heat-bath rule and the local energy `-T log p(cell | amino acid)`, whose probabilities come from
the projected native angles of the bundled structures with add-one smoothing.

### Library models

- `rna_neutral_networks` (`biology/evolution`): a `LatticeElement` of walkers on `LatticeSpace((walkers, length))`, and
  a `SamplingInteraction` with the point mutation move, a `RowScoreSpec` that folds through ViennaRNA, and temperature
  zero. It raises `MissingExtraError` without the extra `rna`.
- `fuzzy_oil_drop` (`biology/systems_biology`): replays 1L2Y by default through a `DriveLaw`, and with
  `dynamics='folding'` samples the dihedral angles with a `RowScoreSpec` of `RD` and a clash penalty. Its catalog
  entry declares `RD` and `K` as observables.
- `early_stage_folding_intermediate` (`biology/systems_biology`): samples the path cells of BPTI's residues.

All three stay within the gallery budget of 300 frames and 60 s.

## Risks / Trade-offs

- [ViennaRNA's licence is not OSI approved] → It is never installed by default, and the extra's documentation states
  the licence. The tests, docs, and gallery sessions install it explicitly.
- [Row scores of Python functions are slow on long rows] → The memo removes repeated folds, and neutral walks stay at
  lengths where ViennaRNA folds in under a millisecond.
- [The book does not fix `σ`, the scale, or the ellipse parameters] → Each is a parameter whose default and source the
  docstring names, and the scenarios test properties that hold for any choice.
- [Biopython's internal coordinates warn on NumPy's `where` without `out`] → The package filters this one warning by
  category and module, and a test checks that no other warning is hidden.

## Migration Plan

Additive. Rollback removes the new names and the extras, and returns the three inventory entries to blocked.
