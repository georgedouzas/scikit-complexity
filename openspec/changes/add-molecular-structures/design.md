# Design

## Context

The parts of the framework that this change builds on, after `redesign-system-model`:
- A system is `System(label, components=, interactions=, space=, time=)`. A `Group(on='sites')` on a `LatticeSpace`
  holds categorical or real states per site.
- The `Sample` law, in the discrete class, holds an energy, a move, and an acceptance: Metropolis by default, with
  Glauber and heat-bath as the general alternatives, at a live temperature. Its energy is local: it reads a site and its
  neighbourhood, and its change is derived symbolically. The internal form is `Form.MONTE_CARLO` with the `MonteCarlo`
  stepper, under `DiscreteTime()`.
- `Drive` changes a whole lattice state through a Python function, and `Call` and `NumericFunction` evaluate functions
  inside laws, on NumPy from Python and on Numba when the function is compiled.
- Every primitive declares `Requires`, every solver and backend `Supports`, and `validate()` reports misuse as issues
  with a code, path, message, and fix before compilation.
- Observables are given to simulations and catalog entries and recorded per frame in trajectories, and
  `Trajectory.lattice` returns a lattice state per frame.
- `skcomplex.core.analysis` already has `kl_divergence`, `shannon_entropy`, and `as_graph` on NetworkX.
- `PointsVisual` draws group positions, and `ImageVisual` draws lattices.
- `core.backends` holds only execution backends, and specific realizations live in `skcomplex.models.<field>`.

What is missing is a score of a whole row of a lattice, the geometry that turns internal coordinates into positions,
structure files, and sequence-to-structure maps. Raman (2021 §13.2–13.3) and Konieczny et al. (2023 §1.3.1, §3.7.1)
give the three models. The survey rejected them as static. The proposal explains why each one is a Monte Carlo dynamics
or a replay once a row can be scored as a whole.

## Goals / Non-Goals

**Goals:**

- Fit the three models into the existing System, Simulation, and Trajectory interface without a new runtime: walks
  and folding are Monte Carlo time, and ensembles are replayed one member per frame.
- Keep the core domain-agnostic. A sequence is a categorical row over any alphabet, a chain is any polymer with
  internal coordinates, and a score is any function of a row. RNA folding, protein files, the protein backbone,
  hydrophobicity, and genotype-phenotype analyses are realizations in `skcomplex.models.biology`.
- Declare the requirements of every new core primitive, so validation reports misuse before compilation.
- Take parsing, internal coordinates, superposition, hydrophobicity scales, RNA folding, and trajectory formats from
  mature libraries, and keep only the adapters in the codebase.
- Keep the default install under licences compatible with MIT, with ViennaRNA and MDTraj behind extras.

**Non-Goals:**

- Molecular dynamics with force fields. OpenMM and LAMMPS do that, and the LAMMPS backend already exists. This change
  reads their trajectories.
- Tertiary structure prediction, such as AlphaFold or Rosetta.
- Secondary structure of proteins from DSSP. MDTraj computes it with the extra, and no model of this change needs it.
- Partition functions, suboptimal structures, and RNA design. ViennaRNA provides them, and they can be added to the
  adapter later.
- Population genetics with selection coefficients and drift, such as Wright-Fisher on genotype networks. A later change
  can add it on the same sequences and maps.

## Decisions

### Library survey

Checked on 2026-10-19 against PyPI, GitHub, and spike installs on macOS arm64 with Python 3.13:

| Library | Version (date) | Licence | Wheels for Linux and macOS, 3.12–3.13 | Maintenance | Role | Decision |
|---|---|---|---|---|---|---|
| Biopython | 1.88 (2026-08-12) | Biopython License Agreement, permissive, partly dual BSD-3-Clause | cp312 and cp313 for manylinux x86_64 and aarch64 and macOS arm64, and an sdist that builds on macOS x86_64. Needs only NumPy | very active, 5.2k stars, last push 2026-10-18 | PDB and mmCIF parsing and writing, internal coordinates and rebuilding from them, SVD superposition, hydrophobicity scales | adopt as a runtime dependency |
| ViennaRNA | 2.7.2 (2025-12-30) | ViennaRNA licence: free for research, education, and commercial use, but no redistribution for a fee and inclusion in a commercial product needs the authors' permission. Not OSI approved | cp312 and cp313 for manylinux x86_64 and aarch64 and macOS x86_64 and arm64 | active, last push 2026-10-16 | MFE folding with the Turner 2004 parameters, base-pair distance, plane layouts | adopt behind the extra `rna` |
| MDTraj | 1.11.2 (2026-10-14) | LGPL-2.1-or-later | cp312 and cp313 for manylinux and macOS universal | active, last push 2026-10-14 | DCD, XTC, TRR, NetCDF, and HDF5 trajectories, DSSP, dihedrals | adopt behind the extra `trajectories` |
| MDAnalysis | 2.10.0 (2025-10-17) | LGPL-3.0-or-later | cp312 and cp313, but no manylinux aarch64 wheel | active | trajectory formats and analyses | reject, MDTraj covers the formats with lighter dependencies and an aarch64 wheel |
| gemmi | 0.7.5 (2026-03-02) | MPL-2.0 | cp312 and cp313 for all targets | active | fast mmCIF and PDB I/O | reject, Biopython also gives internal coordinates and superposition, and the bundled structures are small |
| seqfold | 0.10.2 (2026-06-14) | MIT | abi3 wheels for all targets, Rust core | one maintainer, 111 stars | MFE folding with older nearest-neighbour parameters | reject, its folds differ from ViennaRNA's for the same sequence, and the literature on neutral networks uses ViennaRNA |
| PeptideBuilder | 1.1.0 (2020-04-05) | MIT | pure Python | inactive since 2020 | peptides from dihedral angles | reject, Biopython's internal coordinates do the same |
| NetworkX | 3.7 (2026-09-21) | BSD-3-Clause | pure Python | very active | neutral networks and their components | already a runtime dependency |
| RNAstructure, NUPACK, LinearFold | — | GPL-2.0, or non-commercial licences | no wheels on PyPI | — | RNA folding | reject |

The spike measured:
- ViennaRNA folds a 30-mer in 0.14 ms, and seqfold in 0.13 ms. On a random 30-mer they returned
  `...(((((...))))).((((...)))).. ` at −3.5 kcal/mol and `.((.((..((....)).)).))........` at −3.8 kcal/mol.
- ViennaRNA folds `GGGGAAACCCC` into `((((...))))` at −4.5 kcal/mol, and the base-pair distance of `((((...))))` and
  `.(((...))).` is 1.
- A neutral walk of 8 walkers of length 30 for 300 steps, with robustness and evolvability over all 90 neighbours every
  10 steps and a memo of folds, takes 3.0 s. Over three seeds the Spearman correlation of genotype robustness and
  evolvability is between −0.69 and −0.77, and the mean Hamming distance from the start is between 12.8 and 14.1.
- Biopython parses the 38 models of 1L2Y in 0.18 s and computes the fuzzy oil drop of one model in 0.33 ms. Its `RD`
  ranges over 0.69–0.85 with a mean of 0.77, with the Kyte-Doolittle scale. Native ubiquitin (1UBQ) has `RD` 0.70, and
  the chain rebuilt with `phi = −120°` and `psi = 130°` has 0.87. Rebuilding 76 residues from internal coordinates
  takes 21 ms.

### Licensing: Biopython in the core, ViennaRNA and MDTraj behind extras

Biopython's licence is permissive, so it is a runtime dependency, as `highspy` is for constraint networks. ViennaRNA's
licence forbids redistribution for a fee and asks for permission to include it in a commercial product, which an MIT
package cannot promise its users, so it lives behind the extra `rna`. MDTraj is LGPL, so it lives behind the extra
`trajectories`, as the constitution requires of copyleft libraries. The adapters import them lazily, with a suppression
that gives the reason, and raise `MissingExtraError` naming the capability and the extra. `MissingExtraError` is added
to the errors module as a `BackendError` unless `add-constraint-based-models` has added it first. The project manifest
gains the two extras, deptry learns the optional imports, and the Project Profile lists them.

### How the models fit System, Simulation, and Trajectory

None of the three models integrates an equation, and each keeps its own meaning of time:

| Model | What a unit of time is | Form | What `run` returns | What the trajectory holds |
|---|---|---|---|---|
| `rna_neutral_networks` | one proposed point mutation per walker | Monte Carlo with a global score | a `Simulation` whose last frame holds the walkers' sequences | per frame, the sequences as a `(walkers, length)` lattice, and the catalog entry's observables for structure index, free energy, base-pair distance, Hamming distance, and, every `measure_every` frames, robustness and evolvability |
| `fuzzy_oil_drop`, replay | one member of the ensemble | discrete map driven by a replay law | a `Simulation` that has replayed the ensemble | per frame, the effective-atom positions and the catalog entry's observables for `T`, `O`, `R`, the divergences, `RD`, and `K` |
| `fuzzy_oil_drop`, folding | one sweep of dihedral moves | Monte Carlo with a global score | a `Simulation` at the last conformation | per frame, the dihedral angles, positions, `RD`, and radius of gyration |
| `early_stage_folding_intermediate` | one heat-bath sweep of ellipse cells | Monte Carlo | a `Simulation` at the last sampled conformation | per frame, the cell, the dihedral angles, the positions, the angles between peptide bond planes, and the radius of gyration |

A replay is not forced onto a time axis. The member index becomes the frame index, which is what the viewer and the
analysis plots step through. Static results, such as the native projection of the early-stage model or the neutral
network of a toy map, are analysis functions that the examples call on the model's inputs or on its trajectory, not
frames. Every model runs under `DiscreteTime()`.

### Where each piece lives

| Piece | Home | Reason |
|---|---|---|
| `Chain` with the bead geometry | `core.components` | any polymer with internal coordinates |
| global scores, temperature zero, the `'mutation'` move kind | `core.laws`, in `Sample` | the redesign's sampling law, extended |
| the replay law | `core.laws` | any ensemble replayed one member per step |
| radius of gyration, end-to-end distance, superposed RMSD | `core.analysis` | geometry of any point chain |
| `ChainVisual` | `core.viz` | draws any chain |
| `MissingExtraError` | `core.base` | a general error |
| `Structure`, `read_structure`, `write_structure`, `read_trajectory`, effective atoms, hydrophobicity scales, the protein backbone geometry, backbone dihedral and peptide plane angles, the fuzzy oil drop | `skcomplex.models.biology.realizations` | molecular biology; shared by the two `systems_biology` models |
| `GenotypeMap`, `rna_fold`, `rna_layout`, the neutral network analyses | the `biology/evolution` model package | used only by `rna_neutral_networks` |

### Sequences

A sequence is a categorical state of a `Group(on='sites')` on a `LatticeSpace` of shape `(copies, length)`. The
`Categorical` domain already names the letters. The alphabet is the domain's labels, and the core ships no biological
alphabet. The models declare `ACGU` and the twenty amino acids themselves. `Sample` gains the move kind `'mutation'`,
which draws a different value uniformly. Strings convert through the domain's labels, and a letter outside them raises
`ValidationError`.

### Chains and internal coordinates

`Chain` is a site group on a one-dimensional `LatticeSpace`. Its site states are bond lengths, bond angles, and
dihedral angles. Its positions are a derived vector state that the stepper recomputes after an accepted move and the
visuals read. The core builds in one geometry, a generic bead chain rebuilt by the natural extension reference frame
on NumPy and Numba, and takes any other geometry as an object that maps internal coordinates to positions. The
protein backbone is such a geometry, in `skcomplex.models.biology.realizations`: its `phi`, `psi`, and `omega` per
residue are written into Biopython's `IC_Chain` and rebuilt with `internal_to_atom_coordinates`. Bond lengths and angles
stay at the values Biopython reads or its defaults.

The NeRF of the bead chain is about twenty lines of vector algebra, which Biopython does not expose for arbitrary beads.
It is kept in house for that reason and checked against Biopython on a backbone.

`Chain` declares `Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE}, domains={DomainKind.REAL,
DomainKind.ANGLE})`. `validate()` reports a chain on a lattice of more than one dimension, or outside a lattice, with
the fix of a one-dimensional `LatticeSpace`.

### Global scores

`Sample(score=...)` takes, in place of a local energy, a `NumericFunction` of the whole row of a lattice group and the
component's parameters that returns one number. The stepper proposes a move on a row, evaluates the score of the
proposed row, and accepts by the Metropolis acceptance of `Sample` on the difference. The acceptance takes a
temperature of zero, which accepts only non-increasing scores. Neutral walks use the score `0` on the target phenotype
and `1` off it, and adaptive walks use the base-pair distance. Each row has its own random draws from the simulation's
stream, so rows are independent copies. `proposals` sets the number of moves per row and unit of time, 1 for walks, and
the row length for sweeps. The score of every row is recorded in the trajectory.

A score that wraps a Python function, such as an RNA fold, runs on the NumPy backend. A score whose function is
Numba-compiled, such as the bead-chain energy or the fuzzy oil drop distributions, runs inside the Numba Monte Carlo
kernel. The temperature is a parameter symbol, so it is a Sage expression lowered like any other parameter and changes
live.

`Sample` keeps `LawKind.SAMPLE` and its requirements, `Requires(time={TimeKind.DISCRETE}, space={SpaceKind.LATTICE})`.
`validate()` reports:
- a `Sample` that gives both an energy and a score, or a local energy and a score on one lattice state, since one
  acceptance step cannot combine a symbolic local difference with an opaque global one; the fix names the two laws;
- a score of an uncompiled Python function with the Numba backend, with the fix of the NumPy backend or a
  Numba-compiled function. The NumPy and Numba backends extend their `Supports` declarations accordingly.

### Replay

The replay law is a whole-state law of the discrete class, lowered like `Drive`, whose function returns member `k` of an
ensemble at step `k`. It sets a chain's positions and holds the last member after the end. It declares
`Requires(time={TimeKind.DISCRETE})`, and validation reports it under another time. The bundled structures it replays
by default live in the `fuzzy_oil_drop` model package.

### Sequence-to-structure maps

`GenotypeMap`, in the `biology/evolution` model package, wraps a function from a sequence string to a phenotype key and
a mapping of named numbers, with an LRU cache whose size is a parameter and whose hits and misses it reports.
`rna_fold` is the ViennaRNA map: it calls `RNA.fold_compound` with a model of a given temperature, returns the
dot-bracket structure, the free energy, and, with a target, `RNA.bp_distance`. `rna_layout` returns plane coordinates
from `RNA.simple_xy_coordinates` and the base pairs from `RNA.ptable`. Any Python function is a map, so a user can map
digital circuits or Boolean networks to their functions, as Raman's Table 13.1 lists.

### Neutral network analyses

These live in the `biology/evolution` model package beside the maps. `neutral_network(map, alphabet, length)`
enumerates all sequences when `len(alphabet) ** length` is at most 2**20, and
otherwise takes sequences sampled by neutral walks, and returns a NetworkX graph with point-mutation edges.
`genotype_robustness`, `genotype_evolvability`, `phenotype_robustness`, and `phenotype_evolvability` follow Wagner's
definitions as Raman (2021 §13.3) states them, with each neighbour equally likely. They count neighbours through the
map, so the memo makes repeated neighbourhoods free.

### Structures, ensembles, and replay

These live in `skcomplex.models.biology.realizations`. `Structure` holds the models of an ensemble as arrays:
coordinates of shape `(models, atoms, 3)`, and atom names, residue names and numbers, and chain identifiers.
`read_structure` reads PDB and mmCIF with Biopython, and `write_structure` writes them with `PDBIO` and `MMCIFIO`.
`effective_atoms` averages the atoms of each residue. `read_trajectory` reads molecular dynamics formats with MDTraj
and returns a `Structure` with one model per frame.

The replay law of the core replays a `Structure`'s models. The bundled structures are the PDB entries 1L2Y (Trp-cage, NMR, 38 models), 1UBQ
(ubiquitin), and 5PTI (BPTI), stored as gzipped mmCIF in the model package. PDB data is CC0, so it can ship with an
MIT package, and the tests never reach the network.

### Fuzzy oil drop

`fuzzy_oil_drop_profile(structure, scale, cutoff=9.0)`, in `skcomplex.models.biology.realizations`, follows Konieczny et al. (2023) Eqs. 1.1–1.6:
- The effective atoms are centred and rotated onto their principal axes. Each `σ` is a third of the largest absolute
  coordinate along its axis, so the drop holds the molecule within three standard deviations. The book leaves `σ`
  unstated, and the docstring records this choice.
- `O` uses Levitt's function of Eq. 1.2 with the cut-off `c`.
- `T`, `O`, and `R` are normalised, and the divergences use `kl_divergence` in bits.
- `K` minimises `D_KL(O|M)` over `[0, 5]` with SciPy's bounded scalar minimiser, with `M` from Eq. 1.9.

The default scale is Biopython's Kyte-Doolittle scale (`Bio.SeqUtils.ProtParamData.kd`), rescaled to `[0, 1]`, since
Eq. 1.2 adds the intrinsic hydrophobicities of two residues and needs them non-negative. Biopython's other scales and a
user mapping are accepted. The profile has a Numba kernel for the folding variant's score.

### Early-stage ellipse path

The path is an ellipse on the Ramachandran torus with a centre, two semi-axes, and a tilt, all parameters. The default
values are those that Brylinski et al. (2004, Bioinformatics 20:199) fitted, transcribed with a reference in the
model's docstring. Refitting the path to Eq. 3.1 was rejected. The book does not define the curvature radius `R`
precisely. A spike that took `V` as the angle between successive C=O vectors and `R` as the circumradius of three
successive peptide bond centres of a regular chain gave `V = 23.5°` and `ln R = 0.78` at the helix, against 0.56 from
Eq. 3.1, and `V = 164.6°` and `ln R = 5.28` at the strand, against 6.75. The trend agrees, but the values differ. The
example draws this comparison as a qualitative check.

The projection of a native pair is the nearest point of the path in the periodic metric, found by sampling the path's
parameter and refining with SciPy's bounded minimiser. The path is divided into cells, 72 by default (5° of the
parameter). The probabilities of each amino acid over the cells come from the projected native angles of the bundled
structures, with add-one smoothing. They are computed once when the model is created. Sampling uses
`Sample` with heat-bath acceptance over the cells, with a site energy of `−T log p(cell | amino acid)` so that heat-bath draws
follow the probabilities at any temperature `T`.

### Visuals

`ChainVisual` logs a chain's positions as `Points3D` or `Points2D` under `<chain>/sites`, with the bonds as one
`LineStrips3D` or `LineStrips2D` under `<chain>/bonds` and optional pairs under `<chain>/pairs`. Points are coloured by
a site value with a range fixed at construction, so frames compare. Sequence lattices use the existing `ImageVisual`
with a categorical colormap. Scores and the simulation's observables stream as scalars, as they do already. `default_blueprint` puts the
chain view first, so it is the gallery thumbnail.

### Library models and gallery examples

- `rna_neutral_networks` (`biology/evolution`) declares the population lattice and a `Sample` with the mutation move
  and a neutral or adaptive score through `rna_fold`. Its example streams the target layout coloured by mutation counts, then plots
  robustness against evolvability, and the size of the sampled neutral network against the walk length.
- `fuzzy_oil_drop` (`biology/systems_biology`) replays 1L2Y by default. Its example plots `T`, `O`, and `R` per residue
  for model 1, and `RD` and `K` per model.
- `early_stage_folding_intermediate` (`biology/systems_biology`) samples BPTI's cells. Its example plots the
  Ramachandran map with native, path, and projected angles, and the comparison with Eq. 3.1.

All three stay within the gallery budget of 300 frames and 60 s. The neutral walk takes 3 s in the spike, and the other
two take under a second for their default sizes. The inventory entries move from `out of scope` to
`pending add-molecular-structures`, and from pending to `implemented` with the code.

## Risks / Trade-offs

- [ViennaRNA's licence is not OSI approved] → It is never installed by default. The extra's documentation states the
  licence, and the tests, docs, and gallery sessions install it explicitly.
- [Scores of Python functions are slow on large rows] → The memo removes repeated folds, and the Numba path covers
  compiled scores. Neutral walks stay at lengths where ViennaRNA folds in under a millisecond.
- [The book does not fix `σ`, the hydrophobicity scale, or the ellipse parameters] → Each one is a parameter whose
  default and source the docstring names. The scenarios test properties that hold for any choice, such as `RD = 0` for
  an ideal micelle and native below extended, rather than published values.
- [Bundled structures add data to the wheel] → Three mmCIF files add a few hundred KB once gzipped,
  and they are read with `importlib.resources`.
- [Biopython's internal coordinates warn on NumPy's `where` without `out`] → The adapter filters this one warning by
  category and module, and a test checks that no other warning is hidden.

## Migration Plan

Additive. Install the extras where RNA folding or MD trajectories are needed. Rollback removes the new names and the
extras, and returns the three inventory entries to out of scope.

## Open Questions

- **Sequences as lattice rows or as the sequence domain.** The redesign added a sequence state domain over an alphabet,
  used by rewrite laws. This change keeps fixed-length sequences as categorical rows of a lattice site group, which
  gives vectorised per-site moves and the existing lattice kernels. Whether fixed-length sequences should instead use
  the sequence domain, with `Sample` targeting it, needs a product decision.
- **Biopython as a runtime dependency.** After the rebase, Biopython is used only by `skcomplex.models.biology`, not by
  the core. The project's rule keeps a library the core uses as a runtime dependency and allows extras only for
  model-only dependencies. Whether Biopython stays a runtime dependency, as this change first decided, or moves to an
  extra like ViennaRNA and MDTraj needs a product decision.
- **General sequence maps in the core.** `GenotypeMap` and the neutral network, robustness, and evolvability analyses
  apply to any map from sequences to keys, such as circuits or Boolean networks, but are named for biology. This
  change places them in the `biology/evolution` package under their current names; promoting them to the core under
  neutral names would be a new design.
